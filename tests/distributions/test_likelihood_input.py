"""embed(states, likelihood): the likelihood features reach the moment stream as columns and
the learned relational stream as a gate, with the states' dtype never touching them."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from embedex.distributions.builder import create_model
from embedex.distributions.config import EmbeddingConfig, GeneratorConfig, ModelConfig
from embedex.distributions.relational import LikelihoodGate, pair_weights

N, CLASSES, MOMENT_DIM = 9, 2, 16


def _model(likelihood_dim):
    cfg = ModelConfig(
        input_dim=N,
        likelihood_dim=likelihood_dim,
        embedding=EmbeddingConfig(
            block_type='set_transformer',
            embedding_dim=MOMENT_DIM,
            num_classes=CLASSES,
            relational_mode='learned',
        ),
        generator=GeneratorConfig(flow_type='ar', ar_num_classes=CLASSES),
    )
    return create_model(cfg, dequantize=False, key=jax.random.key(0))


def test_bool_states_do_not_truncate_the_likelihood():
    model = _model(1)
    rng = np.random.default_rng(0)
    xs = rng.integers(0, 2, size=(32, N)).astype(bool)
    lik = jnp.asarray(np.where(rng.random(32) < 0.5, 0.0, -2.5).astype(np.float32))[:, None]
    z_bool, _ = model.embed(jnp.asarray(xs), lik)
    z_float, _ = model.embed(jnp.asarray(xs, dtype=jnp.float32), lik)
    z_zero, _ = model.embed(jnp.asarray(xs), jnp.zeros_like(lik))
    assert np.allclose(np.asarray(z_bool), np.asarray(z_float), atol=1e-6)
    assert not np.allclose(np.asarray(z_bool), np.asarray(z_zero), atol=1e-3)


def test_embed_requires_the_declared_likelihood_shape():
    model = _model(1)
    xs = jnp.zeros((8, N))
    try:
        model.embed(xs)
    except ValueError as e:
        assert 'likelihood' in str(e)
    else:
        raise AssertionError('embed() accepted a missing likelihood')


def test_gate_starts_uniform_and_pair_weights_reduce_to_sample_weights():
    gate = LikelihoodGate(N)
    w = gate(16, jnp.linspace(-30.0, 0.0, 16)[:, None])
    assert w.shape == (16, N)
    assert np.allclose(np.asarray(w), 1.0 / 16, atol=1e-6)
    ws = jax.nn.softmax(jnp.linspace(-3.0, 0.0, 16))
    p = pair_weights(jnp.broadcast_to(ws[:, None], (16, N)))
    assert np.allclose(np.asarray(p), np.asarray(ws)[:, None, None], atol=1e-6)
    assert np.allclose(np.asarray(p.sum(0)), 1.0, atol=1e-6)


def test_closed_gate_gives_posterior_pair_masses():
    model = _model(1)
    a = jnp.full(N, 1.0)
    b = jnp.full(N, 15.0)
    stream = eqx.tree_at(lambda s: (s.gate.a, s.gate.b), model.relational, (a, b))
    rng = np.random.default_rng(1)
    xs = jnp.asarray(rng.integers(0, 2, size=(64, N)).astype(np.float32))
    lik = jnp.asarray(np.where(rng.random(64) < 0.5, 0.0, -30.0).astype(np.float32))[:, None]
    kept = np.asarray(lik[:, 0] == 0.0)
    z_gated = stream(xs, lik)
    z_kept = stream(xs[kept], jnp.zeros((int(kept.sum()), 1)))
    assert np.allclose(np.asarray(z_gated), np.asarray(z_kept), atol=1e-4)


def test_gate_index_maps_variables_to_their_factor():
    # three variables per factor: an explicit index routes each variable to its column
    index = (0, 0, 0, 1, 1, 1, 2, 2, 2)
    gate = LikelihoodGate(N, index)
    lik = jnp.zeros((8, 3)).at[:4, 1].set(-30.0)  # factor 1 rules out the first 4 samples
    w = np.asarray(
        eqx.tree_at(lambda g: (g.a, g.b), gate, (jnp.ones(N), jnp.full(N, 15.0)))(8, lik)
    )
    assert np.allclose(w[:, [0, 1, 2, 6, 7, 8]], 1 / 8, atol=1e-6)
    assert np.allclose(w[:4, 3:6], 0.0, atol=1e-6) and np.allclose(w[4:, 3:6], 0.25, atol=1e-6)


def test_gate_rejects_unmapped_feature_width():
    gate = LikelihoodGate(N)
    try:
        gate(4, jnp.zeros((4, 3)))
    except ValueError as e:
        assert 'likelihood_index' in str(e)
    else:
        raise AssertionError('a 3-wide feature for 9 variables needs an index')
