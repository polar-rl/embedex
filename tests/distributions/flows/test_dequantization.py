"""Dequantization layers, which let a normalizing flow model categorical data by spreading each
state over the unit cell around it."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.distributions.flows.dequantization import Dequantize, UniformDequantize

STATES = jnp.array([0.0, 2.0])


@pytest.fixture
def learned(make_model) -> Dequantize:
    """The learned dequantization the builder puts in front of a discrete normalizing flow, with
    a noise distribution that is neither centered nor of unit scale."""

    layer = make_model(flow_type='nf', dequantize=True).generator.flows[0]
    assert isinstance(layer, Dequantize)

    last = layer.net.layers[-1]
    location_and_log_scale = jnp.array([0.3, -0.4, 0.4, -0.3])
    return eqx.tree_at(
        lambda d: (d.net.layers[-1].weight, d.net.layers[-1].bias),
        layer,
        (jnp.zeros_like(last.weight), location_and_log_scale),
    )


@pytest.fixture
def sideinfo(key) -> jax.Array:
    return jax.random.normal(key, (8,))


@pytest.mark.parametrize('layer', ['learned', 'uniform'])
def test_noise_stays_in_the_cell_and_rounds_back(layer, learned, sideinfo, key):
    layer = learned if layer == 'learned' else UniformDequantize()

    noisy = jax.vmap(lambda k: layer._forward(STATES, 0.0, sideinfo=sideinfo, key=k)[0])(
        jax.random.split(key, 1000)
    )
    restored = jax.vmap(lambda x: layer._inverse(x, 0.0)[0])(noisy)

    assert np.all(np.abs(noisy - STATES) < 0.5)
    np.testing.assert_array_equal(restored, jnp.broadcast_to(STATES, restored.shape))


def test_uniform_noise_leaves_the_log_det_alone(key):
    _, ldj = UniformDequantize()._forward(STATES, 0.0, key=key)

    assert float(ldj) == 0.0


def test_learned_noise_log_density_integrates_to_one_over_the_cell(learned, sideinfo, key):
    # The layer subtracts log q(u | x) from the log-det, and E_q[1 / q(u)] is the volume q covers.
    # The noise is clipped to |logit(u + 1/2)| < 5, so that volume is (sigmoid(5) - sigmoid(-5))^2.
    def forward(k):
        noisy, ldj = learned._forward(STATES, 0.0, sideinfo=sideinfo, key=k)
        return noisy - STATES, -ldj

    noise, log_q = jax.vmap(forward)(jax.random.split(key, 50000))

    unclipped = jnp.all(jnp.abs(jax.scipy.special.logit(noise + 0.5)) < 4.99, axis=-1)
    volume = (jax.nn.sigmoid(5.0) - jax.nn.sigmoid(-5.0)) ** 2
    estimate = jnp.mean(jnp.where(unclipped, jnp.exp(-log_q), 0.0))
    assert float(estimate) == pytest.approx(float(volume), rel=0.02)
    assert float(log_q.std()) > 0.1  # a uniform q would pass trivially


def test_learned_noise_depends_on_the_embedding(make_model, sideinfo, key):
    layer = make_model(flow_type='nf', dequantize=True).generator.flows[0]

    noise, _ = layer._forward(STATES, 0.0, sideinfo=sideinfo, key=key)
    other, _ = layer._forward(STATES, 0.0, sideinfo=-sideinfo, key=key)

    assert not np.allclose(noise, other)
