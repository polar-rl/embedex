"""The Lotka-Volterra predator-prey system, integrated numerically."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

BATCH = 8


@pytest.fixture
def generator(families):
    return families['lotka_volterra'].generator()


@pytest.fixture
def params(generator, key) -> jax.Array:
    return generator.sample(BATCH, 16, key)[2]


def test_trajectories_start_at_the_initial_populations(generator, params):
    for p in params:
        np.testing.assert_allclose(generator.trajectory(jnp.zeros(1), p)[0], p[:2], rtol=1e-6)


def test_trajectories_follow_the_vector_field(generator, params):
    # dx/dt = alpha x - beta x y and dy/dt = delta x y - gamma y, by central differences.
    ts, h = jnp.linspace(0.5, 9.5, 10), 1e-2

    for x0, y0, alpha, beta, gamma, delta in params:
        p = jnp.array([x0, y0, alpha, beta, gamma, delta])
        x, y = generator.trajectory(ts, p).T
        derivative = (generator.trajectory(ts + h, p) - generator.trajectory(ts - h, p)) / (2 * h)
        expected = jnp.stack([alpha * x - beta * x * y, delta * x * y - gamma * y], axis=1)

        np.testing.assert_allclose(derivative, expected, rtol=1e-2, atol=1e-2)


def test_populations_stay_positive_and_conserve_the_first_integral(generator, params):
    # V = delta x - gamma ln x + beta y - alpha ln y is constant along every trajectory.
    ts = jnp.linspace(0.0, 10.0, 200)

    for p in params:
        _, _, alpha, beta, gamma, delta = p
        x, y = generator.trajectory(ts, p).T
        invariant = delta * x - gamma * jnp.log(x) + beta * y - alpha * jnp.log(y)

        assert jnp.all(x > 0) and jnp.all(y > 0)
        np.testing.assert_allclose(invariant, invariant[0], rtol=1e-4)


def test_states_come_back_in_the_order_the_times_were_given(generator, params, key):
    ts = jnp.linspace(0.0, 10.0, 32)
    order = jax.random.permutation(key, 32)

    np.testing.assert_allclose(
        generator.trajectory(ts[order], params[0]),
        generator.trajectory(ts, params[0])[order],
        rtol=1e-5,
    )
