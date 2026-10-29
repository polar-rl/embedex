"""Building networks and optimizers from a config."""

from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from embedex.graphs.builder import (
    build_from_config,
    create_embedding_network,
    create_medians_network,
    create_optimizer,
    create_transition_network,
)
from embedex.graphs.embedding import PointwiseBlock, SetTransformerBlock

BLOCK_TYPES = {'pointwise': PointwiseBlock, 'set_transformer': SetTransformerBlock}


def test_embedding_network_has_one_block_per_stage(make_config, stage_block, key):
    cfg = make_config(stage_block=stage_block)
    cfg.model.embedding.num_stages = 3

    network = create_embedding_network(cfg, key)

    assert len(network.embedding_blocks) == 3
    assert all(isinstance(b, BLOCK_TYPES[stage_block]) for b in network.embedding_blocks)


def test_unknown_stage_block_is_rejected(make_config, key):
    cfg = make_config(stage_block='transformer')

    with pytest.raises(ValueError, match='stage_block'):
        create_embedding_network(cfg, key)


def test_allo_hyperparameters_reach_the_embedding(make_config, key):
    cfg = make_config()
    cfg.model.allo.penalty_coeff, cfg.model.allo.dual_reg = 3.0, 0.5

    network = create_embedding_network(cfg, key)

    assert (network.penalty_coeff, network.dual_reg) == (3.0, 0.5)
    assert network.betas.num_dims == cfg.model.allo.num_dims


def test_building_from_a_saved_config_dict_reproduces_the_network(make_config, stage_block, key):
    # Checkpoints store the config as a dict and rebuild the network from it.
    cfg = make_config(stage_block=stage_block)

    original = create_embedding_network(cfg, key)
    rebuilt = build_from_config(asdict(cfg), key)

    assert eqx.tree_equal(original, rebuilt)


def test_medians_network_predicts_a_positive_radius_per_vertex(make_config, key):
    cfg = make_config()
    network = create_medians_network(cfg, key)
    z = jnp.ones(cfg.model.embedding.embedding_size)

    radii = jax.vmap(network, in_axes=(0, None))(jnp.eye(cfg.model.input_size), z)

    assert radii.shape == (cfg.model.input_size,)
    assert jnp.all(radii > 0)


def test_transition_network_scores_every_pair_of_vertices(make_config, key):
    cfg = make_config()
    network = create_transition_network(cfg, key)
    n = cfg.model.input_size

    weights = network(jnp.ones((n, n)), jnp.ones(cfg.model.embedding.embedding_size))

    assert weights.shape == (n, n)


@pytest.mark.parametrize('schedule', ['constant', 'cosine'])
@pytest.mark.parametrize('grad_clip_norm', [0.0, 1.0])
def test_optimizer_takes_finite_steps(make_config, schedule, grad_clip_norm):
    cfg = make_config()
    cfg.training.schedule, cfg.training.grad_clip_norm = schedule, grad_clip_norm
    params = {'w': jnp.zeros(3)}

    opt = create_optimizer(cfg, num_steps=10)
    updates, _ = opt.update({'w': jnp.array([1.0, -2.0, 1e6])}, opt.init(params), params)

    new_params = optax.apply_updates(params, updates)
    assert np.all(np.isfinite(new_params['w']))
    assert not np.allclose(new_params['w'], 0.0)


def test_unknown_schedule_is_rejected(make_config):
    cfg = make_config()
    cfg.training.schedule = 'linear'

    with pytest.raises(ValueError, match='schedule'):
        create_optimizer(cfg, num_steps=10)
