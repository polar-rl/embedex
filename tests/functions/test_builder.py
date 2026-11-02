"""Building networks, losses and optimizers from a config."""

from dataclasses import asdict
from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from embedex.functions import builder
from embedex.functions.builder import (
    build_from_config,
    create_embedding_network,
    create_operator_network,
    create_optimizer,
)
from embedex.functions.embedding import (
    DotProductBlock,
    PointwiseConcatBlock,
    SetTransformerBlock,
    SetTransformerDotProductBlock,
)
from embedex.functions.evaluator import AttentionDecoder, ConcatDecoder
from embedex.functions.operator import MlpOperator, ResidualOperator

BLOCK_TYPES = {
    'concat': PointwiseConcatBlock,
    'dot_product': DotProductBlock,
    'set_transformer': SetTransformerBlock,
    'set_transformer_dot_product': SetTransformerDotProductBlock,
}
DECODER_TYPES = {'concat': ConcatDecoder, 'attention': AttentionDecoder}


def test_embedding_network_has_one_block_per_stage(make_config, stage_block, key):
    cfg = make_config(stage_block=stage_block)
    cfg.model.embedding.num_stages = 3

    network = create_embedding_network(cfg, key)

    assert len(network.embedding_blocks) == 3
    assert all(isinstance(b, BLOCK_TYPES[stage_block]) for b in network.embedding_blocks)


def test_decoder_follows_the_config(make_config, decoder, key):
    network = create_embedding_network(make_config(decoder=decoder), key)

    assert isinstance(network.evaluator, DECODER_TYPES[decoder])


@pytest.mark.parametrize(
    'kind, operator_type', [('mlp', MlpOperator), ('residual', ResidualOperator)]
)
def test_operator_reads_the_familys_input_and_outputs_an_embedding(
    family, kind, operator_type, make_config, key
):
    cfg = make_config(family)
    cfg.model.operator.kind = kind
    generator = family.generator()

    operator = create_operator_network(cfg, generator, key)

    assert isinstance(operator.net, operator_type)
    source = jnp.ones(generator.operator_input_size(cfg))
    assert operator(source).shape == (cfg.model.embedding.embedding_size,)


@pytest.mark.parametrize(
    'section, name, value',
    [
        ('embedding', 'stage_block', 'transformer'),
        ('embedding', 'activation', 'tanh'),
        ('evaluator', 'decoder', 'mlp'),
        ('evaluator', 'activation', 'tanh'),
        ('training', 'loss_fn', 'l1'),
    ],
)
def test_unknown_embedding_components_are_rejected(section, name, value, make_config, key):
    cfg = make_config()
    setattr(cfg.training if section == 'training' else getattr(cfg.model, section), name, value)

    with pytest.raises(ValueError, match=value):
        create_embedding_network(cfg, key)


def test_unknown_operator_is_rejected(family, make_config, key):
    cfg = make_config(family)
    cfg.model.operator.kind = 'fno'

    with pytest.raises(ValueError, match='fno'):
        create_operator_network(cfg, family.generator(), key)


def test_building_from_a_saved_config_dict_reproduces_the_network(make_config, stage_block, key):
    # Checkpoints store the config as a dict and rebuild the network from it. Each build makes a
    # new loss closure, so only the parameters can be compared.
    cfg = make_config(stage_block=stage_block)

    original = create_embedding_network(cfg, key)
    rebuilt = build_from_config(asdict(cfg), key)

    params = [jax.tree.leaves(eqx.filter(n, eqx.is_array)) for n in (original, rebuilt)]
    for a, b in zip(*params, strict=True):
        np.testing.assert_array_equal(a, b)


PREDICTION = jnp.array([[0.5], [0.0], [0.0]])
TARGET = jnp.array([[0.5], [2.0], [-1.0]])  # errors 0, -2 and 1


@pytest.mark.parametrize(
    'name, expected',
    [
        ('mse', (0 + 4 + 1) / 3),
        ('huber', (0 + 1.5 + 0.5) / 3),  # quadratic below an error of 1, linear above
        ('rel_l2', np.sqrt(5 / 5.25)),
    ],
)
def test_losses_match_their_definitions(name, expected):
    loss = builder._get_loss(name)(PREDICTION, TARGET)

    assert float(loss) == pytest.approx(expected, rel=1e-6)


def test_softmax_cross_entropy_compares_distributions_over_samples():
    # The target [0, 0] is uniform; the prediction [0, ln 3] puts 1/4 and 3/4 on the samples.
    loss = builder._get_loss('softmax_ce')(jnp.array([0.0, np.log(3.0)]), jnp.zeros(2))

    assert float(loss) == pytest.approx(-0.5 * (np.log(0.25) + np.log(0.75)), rel=1e-6)


def test_relative_l2_loss_survives_a_zero_target():
    loss = builder._get_loss('rel_l2')(jnp.ones((3, 1)), jnp.zeros((3, 1)))

    assert jnp.isfinite(loss)


def run_optimizer(opt: optax.GradientTransformation, grads: dict, num_steps: int) -> list[dict]:
    params = {'w': jnp.ones(3)}
    state = opt.init(params)
    history = []
    for _ in range(num_steps):
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        history.append(updates)

    return history


@pytest.mark.parametrize('schedule', ['constant', 'cosine'])
@pytest.mark.parametrize('grad_clip_norm', [0.0, 1.0])
def test_optimizer_takes_finite_descent_steps(make_config, schedule, grad_clip_norm):
    cfg = make_config()
    cfg.training.lr_schedule, cfg.training.grad_clip_norm = schedule, grad_clip_norm
    grads = {'w': jnp.array([1.0, -2.0, 1e6])}

    (updates,) = run_optimizer(create_optimizer(cfg, 0.0, num_steps=10), grads, 1)

    assert np.all(np.isfinite(updates['w']))
    np.testing.assert_array_equal(np.sign(updates['w']), -np.sign(grads['w']))


def test_cosine_schedule_decays_the_learning_rate_to_zero(make_config):
    cfg = make_config()
    cfg.training.lr_schedule = 'cosine'

    history = run_optimizer(create_optimizer(cfg, 0.0, num_steps=10), {'w': jnp.ones(3)}, 11)

    magnitudes = [float(jnp.abs(u['w']).max()) for u in history]
    assert all(a > b for a, b in pairwise(magnitudes[:10]))
    assert magnitudes[10] == 0.0


def test_weight_decay_shrinks_parameters_without_gradient(make_config):
    cfg = make_config()

    (decayed,) = run_optimizer(create_optimizer(cfg, 0.1, num_steps=10), {'w': jnp.zeros(3)}, 1)
    (still,) = run_optimizer(create_optimizer(cfg, 0.0, num_steps=10), {'w': jnp.zeros(3)}, 1)

    assert np.all(np.asarray(decayed['w']) < 0.0)
    np.testing.assert_array_equal(still['w'], 0.0)


def test_unknown_schedule_is_rejected(make_config):
    cfg = make_config()
    cfg.training.lr_schedule = 'linear'

    with pytest.raises(ValueError, match='schedule'):
        create_optimizer(cfg, 0.0, num_steps=10)


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match='tanh'):
        builder._get_activation('tanh')


def test_activations_are_relu_and_gelu():
    x = jnp.array([-1.0, 2.0])

    np.testing.assert_array_equal(builder._get_activation('relu')(x), jax.nn.relu(x))
    np.testing.assert_array_equal(builder._get_activation('gelu')(x), jax.nn.gelu(x))
