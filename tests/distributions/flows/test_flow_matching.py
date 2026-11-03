"""Flow matching: the training objective, and the ODEs that sample and score with the learned
velocity field.

Velocity fields with a known flow pin down the conventions: noise at t = 0, data at t = 1, the
interpolant the loss regresses on, and the continuous change of variables behind `log_prob`.
"""

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import expm
from jax.scipy.stats import multivariate_normal

from embedex.distributions.flows.flow_matching import FlowMatchingModel, sinusoidal_embedding

TARGET = jnp.array([1.5, -0.5])


class PointMassVelocity(eqx.Module):
    """The exact velocity field of the flow-matching path to a point mass at `target`."""

    target: jax.Array
    sigma_min: float

    def __call__(self, x_t, t, sideinfo=None):
        s = 1.0 - self.sigma_min
        return self.target - s * (x_t - t * self.target) / (1.0 - s * t)


class LinearVelocity(eqx.Module):
    """v(x) = A x, whose flow from t = 0 to 1 maps x to expm(A) x."""

    A: jax.Array

    def __call__(self, x_t, t, sideinfo=None):
        return self.A @ x_t


def flow_matching(velocity_net, num_steps: int, sigma_min: float = 0.0) -> FlowMatchingModel:
    model = FlowMatchingModel(velocity_net, input_dim=2, num_steps=num_steps, sigma_min=sigma_min)
    return model.add_prior(distrax.Normal(0.0, 1.0), (2,))


A = jnp.array([[0.3, -0.4], [0.2, -0.1]])


def exact_log_prob(x: jax.Array) -> jax.Array:
    """log p(x) for the noise pushed through the flow of `LinearVelocity(A)`."""

    return multivariate_normal.logpdf(expm(-A) @ x, jnp.zeros(2), jnp.eye(2)) - jnp.trace(A)


def test_time_embedding_is_sines_then_cosines():
    embedding = sinusoidal_embedding(jnp.array(0.0), 8)

    np.testing.assert_array_equal(embedding, [0, 0, 0, 0, 1, 1, 1, 1])


@pytest.mark.parametrize('sigma_min', [0.0, 0.1])
def test_the_exact_velocity_field_has_zero_loss(sigma_min, key):
    model = flow_matching(PointMassVelocity(TARGET, sigma_min), num_steps=8, sigma_min=sigma_min)

    losses = jax.vmap(lambda k: model.loss(TARGET, key=k)[0])(jax.random.split(key, 64))

    np.testing.assert_allclose(losses, 0.0, atol=1e-8)


def test_the_loss_penalizes_a_wrong_velocity_field(key):
    model = flow_matching(PointMassVelocity(TARGET, 0.0), num_steps=8)

    losses = jax.vmap(lambda k: model.loss(TARGET + 1.0, key=k)[0])(jax.random.split(key, 64))

    assert float(losses.min()) > 0.0


@pytest.mark.parametrize('sigma_min', [0.0, 0.1])
def test_sampling_carries_noise_to_the_data(sigma_min, key):
    # The path ends at sigma_min * noise + target.
    model = flow_matching(PointMassVelocity(TARGET, sigma_min), num_steps=8, sigma_min=sigma_min)

    samples = jax.vmap(lambda k: model.sample(None, None, k))(jax.random.split(key, 256))

    np.testing.assert_allclose(samples.mean(0), TARGET, atol=3 * sigma_min / 16 + 1e-5)
    np.testing.assert_allclose(samples.std(0), sigma_min, atol=0.02)


def test_log_prob_follows_the_change_of_variables(key):
    model = flow_matching(LinearVelocity(A), num_steps=2000)
    xs = jax.random.normal(key, (16, 2))

    log_probs, _, _ = jax.vmap(lambda x: model.log_prob(x))(xs)

    np.testing.assert_allclose(log_probs, jax.vmap(exact_log_prob)(xs), atol=2e-3)


def test_the_stochastic_trace_estimate_is_unbiased(key):
    model = flow_matching(LinearVelocity(A), num_steps=16)
    x = jnp.array([0.5, -1.0])
    exact, _, _ = model.log_prob(x)

    estimates, _, _ = jax.vmap(lambda k: model.log_prob(x, key=k))(jax.random.split(key, 8192))

    assert not np.allclose(estimates[0], estimates[1])
    assert float(estimates.mean()) == pytest.approx(float(exact), abs=0.05)


class TestLearnedVelocity:
    """The velocity network the builder makes, conditioned on an embedding."""

    @pytest.fixture
    def model(self, make_model) -> FlowMatchingModel:
        return make_model(flow_type='fm', fm_num_steps=64).generator

    @pytest.fixture
    def sideinfo(self, key) -> jax.Array:
        return jax.random.normal(jax.random.fold_in(key, 2), (8,))

    def test_encoding_inverts_sampling(self, model, sideinfo, key):
        noise = jax.random.normal(key, (2,))

        data, _ = model._inverse(noise, 0.0, sideinfo=sideinfo)
        back = model.encode(data, sideinfo)

        np.testing.assert_allclose(back, noise, atol=1e-3)

    def test_density_integrates_to_one(self, model, sideinfo):
        grid = jnp.linspace(-8.0, 8.0, 161)
        points = jnp.stack(jnp.meshgrid(grid, grid), -1).reshape(-1, 2)

        log_probs, _, _ = jax.vmap(lambda x: model.log_prob(x, sideinfo))(points)

        mass = float(jnp.sum(jnp.exp(log_probs)) * (grid[1] - grid[0]) ** 2)
        assert mass == pytest.approx(1.0, abs=0.02)
