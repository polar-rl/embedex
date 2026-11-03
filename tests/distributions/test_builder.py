"""Building distribution embeddings and their generators from a config."""

import json
from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.distributions.builder import build_from_config, create_model
from embedex.distributions.conditioning import ConcatMLP, FiLMMLP, HyperMLP
from embedex.distributions.embedding import (
    AttentionPool,
    FlowEmbedding,
    PointwiseBlock,
    SetTransformerBlock,
)
from embedex.distributions.flows.autoregressive import AutoregressiveModel
from embedex.distributions.flows.dequantization import Dequantize, UniformDequantize
from embedex.distributions.flows.discrete_flow_matching import (
    AttentionDenoiserNet,
    DiscreteDenoiserNet,
    DiscreteFlowMatchingModel,
)
from embedex.distributions.flows.flow_matching import FlowMatchingModel
from embedex.distributions.flows.normalizing_flow import Coupling, NormalizingFlowModel

BLOCK_TYPES = {'pointwise': PointwiseBlock, 'set_transformer': SetTransformerBlock}
FLOW_TYPES = {
    'nf': NormalizingFlowModel,
    'fm': FlowMatchingModel,
    'dfm': DiscreteFlowMatchingModel,
    'ar': AutoregressiveModel,
}
CONDITIONED_MLPS = {'concat': ConcatMLP, 'film': FiLMMLP, 'hyper': HyperMLP}

# Every generator over categorical samples: (flow_type, extra generator fields).
DISCRETE_GENERATORS = {
    'nf': {},
    'dfm_mlp': {'flow_type': 'dfm', 'dfm_denoiser': 'mlp'},
    'dfm_attention': {'flow_type': 'dfm', 'dfm_denoiser': 'attention'},
    'ar': {'flow_type': 'ar'},
}


def test_embedding_has_one_block_per_round(make_model, block_type):
    model = make_model(block_type=block_type, num_rounds=3)

    assert len(model.embedding_blocks) == 3
    assert all(isinstance(b, BLOCK_TYPES[block_type]) for b in model.embedding_blocks)


def test_attention_readout_has_one_pool_per_round(make_model):
    assert make_model(pool_type='mean').pools is None
    pools = make_model(pool_type='attention', num_rounds=3).pools
    assert len(pools) == 3 and all(isinstance(p, AttentionPool) for p in pools)


def test_attention_readout_supports_one_query_only(make_model):
    with pytest.raises(NotImplementedError, match='pool_num_queries'):
        make_model(pool_type='attention', pool_num_queries=2)


@pytest.mark.parametrize('flow_type', list(FLOW_TYPES))
def test_generator_follows_the_config(make_model, flow_type):
    model = make_model(flow_type=flow_type)

    assert isinstance(model, FlowEmbedding)
    assert isinstance(model.generator, FLOW_TYPES[flow_type])


def test_normalizing_flow_stacks_the_configured_couplings(make_model):
    flow = make_model(flow_type='nf', nf_num_coupling_layers=3).generator

    assert len(flow.flows) == 3
    assert all(isinstance(f, Coupling) and f.dual for f in flow.flows)


@pytest.mark.parametrize(
    'fields, layer_type, conditioning_dim',
    [
        ({}, Dequantize, 8),
        ({'nf_condition_dequant': False}, Dequantize, 0),
        ({'nf_dequant_type': 'uniform'}, UniformDequantize, None),
    ],
    ids=['conditioned', 'unconditioned', 'uniform'],
)
def test_discrete_normalizing_flow_starts_by_dequantizing(
    make_model, fields, layer_type, conditioning_dim
):
    flow = make_model(flow_type='nf', dequantize=True, **fields).generator

    assert isinstance(flow.flows[0], layer_type)
    assert all(isinstance(f, Coupling) for f in flow.flows[1:])
    if conditioning_dim is not None:
        assert flow.flows[0].conditioning_dim == conditioning_dim


@pytest.mark.parametrize(
    'fields, value',
    [
        ({'block_type': 'transformer'}, 'transformer'),
        ({'pool_type': 'max'}, 'max'),
        ({'flow_type': 'ar', 'relational_mode': 'gnn'}, 'gnn'),
        ({'flow_type': 'diffusion'}, 'diffusion'),
        ({'conditioning': 'cross'}, 'cross'),
        ({'flow_type': 'dfm', 'dfm_denoiser': 'transformer'}, 'transformer'),
        ({'dequantize': True, 'nf_dequant_type': 'gaussian'}, 'gaussian'),
    ],
)
def test_unknown_components_are_rejected(make_model, fields, value):
    with pytest.raises(ValueError, match=value):
        make_model(**fields)


def test_flow_matching_cannot_dequantize(make_model):
    with pytest.raises(NotImplementedError, match='DFM'):
        make_model(flow_type='fm', dequantize=True)


@pytest.mark.parametrize('conditioning', list(CONDITIONED_MLPS))
@pytest.mark.parametrize('flow_type', ['nf', 'fm', 'dfm'])
def test_generator_networks_read_the_embedding_as_configured(make_model, conditioning, flow_type):
    generator = make_model(flow_type=flow_type, conditioning=conditioning).generator

    if flow_type == 'nf':
        net = generator.flows[0].transform_param_net
    elif flow_type == 'fm':
        net = generator.velocity_net.net
    else:
        assert isinstance(generator.denoiser_net, DiscreteDenoiserNet)
        net = generator.denoiser_net.net
    assert isinstance(net, CONDITIONED_MLPS[conditioning])


def test_attention_denoiser_follows_the_config(make_model):
    generator = make_model(flow_type='dfm', dfm_denoiser='attention').generator

    assert isinstance(generator.denoiser_net, AttentionDenoiserNet)


def test_relational_stream_is_only_built_when_selected(make_model):
    assert make_model(flow_type='ar').relational is None
    assert make_model(flow_type='ar', relational_mode='none').relational is None
    assert make_model(flow_type='ar', relational_mode='learned').relational is not None


@pytest.mark.parametrize('relational_mode', ['none', 'learned', 'raw_agree', 'raw_edge'])
@pytest.mark.parametrize('generator', list(DISCRETE_GENERATORS))
def test_every_discrete_generator_reads_every_embedding(
    make_model, generator, relational_mode, key
):
    # The relational streams widen the embedding by one value per pair of variables, which the
    # autoregressive model and the attention denoiser deliver to each variable's own token.
    fields = {'flow_type': 'nf', **DISCRETE_GENERATORS[generator]}
    model = make_model(
        input_dim=4,
        num_classes=3,
        dequantize=fields['flow_type'] == 'nf',
        relational_mode=relational_mode,
        **fields,
    )
    xs = jax.random.randint(key, (16, 4), 0, 3).astype(jnp.float32)

    z, _ = model.embed(xs)
    loss, _, _, _ = model.loss(xs, key=key)
    generated = model.generate(z, 8, key=key)

    assert z.shape == (8 + (0 if relational_mode == 'none' else 16),)
    assert jnp.isfinite(loss)
    assert generated.shape == (8, 4)
    if fields['flow_type'] != 'dfm':  # discrete flow matching has no log-probability
        log_probs, _, _ = model.log_prob(xs, z, key=key)
        assert log_probs.shape == (16,) and jnp.all(jnp.isfinite(log_probs))


@pytest.mark.parametrize('conditioning', list(CONDITIONED_MLPS))
def test_every_continuous_generator_reads_the_embedding(
    make_model, continuous_flow, conditioning, key
):
    model = make_model(flow_type=continuous_flow, conditioning=conditioning)
    xs = jax.random.normal(key, (16, 2))

    z, _ = model.embed(xs)
    loss, _, _, _ = model.loss(xs, key=key)
    log_probs, _, _ = model.log_prob(xs, z)

    assert jnp.isfinite(loss)
    assert log_probs.shape == (16,) and jnp.all(jnp.isfinite(log_probs))
    assert model.generate(z, 8, key=key).shape == (8, 2)


def test_attention_denoiser_needs_square_relational_nodes(make_model):
    with pytest.raises(NotImplementedError, match='num_variables'):
        make_model(
            flow_type='dfm',
            dfm_denoiser='attention',
            input_dim=4,
            relational_mode='learned',
            rel_node_dim=3,
        )


@pytest.mark.parametrize('flow_type', list(FLOW_TYPES))
def test_building_from_a_saved_config_reproduces_the_model(make_model_config, flow_type, key):
    # Checkpoints store the config as JSON and rebuild the model from it.
    cfg = make_model_config(flow_type=flow_type, relational_mode='learned')
    saved = json.loads(json.dumps({'model': asdict(cfg)}))

    original = create_model(cfg, dequantize=False, key=key)
    rebuilt = build_from_config(saved, dequantize=False, key=key)

    params = [jax.tree.leaves(eqx.filter(m, eqx.is_array)) for m in (original, rebuilt)]
    for a, b in zip(*params, strict=True):
        np.testing.assert_array_equal(a, b)
