"""A chain of coupled damped harmonic oscillators, solved in closed form."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.functions.families.damped_oscillator import DampedOscillatorGenerator

NUM_SAMPLES = 64


def make_generator(num_masses: int) -> DampedOscillatorGenerator:
    return DampedOscillatorGenerator(
        (0.0, 10.0), (-2.0, 2.0), (-2.0, 2.0), (0.3, 2.5), (0.5, 3.0), (0.5, 1.5), num_masses
    )


def system_matrix(params: np.ndarray, n: int) -> np.ndarray:
    """A = [[0, I], [-K, -2 gamma I]], where the chain's stiffness K ties every mass to its rest
    position (omega^2) and to each neighbour (coupling k)."""

    gamma, omega, k = params[2 * n :]
    neighbours = np.eye(n, k=1) + np.eye(n, k=-1)
    stiffness = np.diag(omega**2 + k * neighbours.sum(1)) - k * neighbours

    return np.block([[np.zeros((n, n)), np.eye(n)], [-stiffness, -2 * gamma * np.eye(n)]])


@pytest.mark.parametrize('num_masses', [1, 3])
def test_trajectories_solve_the_linear_system(num_masses, key):
    generator = make_generator(num_masses)
    _, _, params = generator.sample(4, NUM_SAMPLES, key)
    ts = jnp.array([0.0, 0.5, 2.0, 7.0])

    for p in params:
        states = generator.trajectory(ts, p)
        velocities = jax.jacfwd(lambda t, p=p: generator.trajectory(t, p))(ts)
        velocities = jnp.einsum('tdt->td', velocities)

        np.testing.assert_allclose(states[0], p[: 2 * num_masses], rtol=1e-6)  # z(0) = z0
        expected = states @ system_matrix(np.asarray(p, np.float64), num_masses).T
        np.testing.assert_allclose(velocities, expected, rtol=1e-3, atol=1e-4)


def test_damping_never_adds_energy(key):
    # E = (v^2 + omega^2 x^2) / 2 changes at the rate -2 gamma v^2.
    generator = make_generator(1)
    _, _, params = generator.sample(16, NUM_SAMPLES, key)
    ts = jnp.linspace(0.0, 10.0, 200)

    for x0, v0, gamma, omega, _ in params:
        x, v = generator.trajectory(ts, jnp.array([x0, v0, gamma, omega, 0.0])).T
        energy = np.asarray(v**2 + omega**2 * x**2) / 2

        assert np.all(np.diff(energy) <= 1e-6 * energy[0])
        assert energy[-1] < energy[0]


def test_coupling_carries_motion_along_the_chain():
    # Only the first mass starts displaced; its neighbour moves only through the coupling.
    generator = make_generator(3)
    ts = jnp.linspace(0.0, 5.0, 50)
    x0, v0 = [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    coupled = generator.trajectory(ts, jnp.array([*x0, *v0, 0.3, 1.0, 1.0]))
    uncoupled = generator.trajectory(ts, jnp.array([*x0, *v0, 0.3, 1.0, 0.0]))

    assert np.abs(coupled[:, 1:3]).max() > 0.1
    np.testing.assert_allclose(uncoupled[:, 1:3], 0.0, atol=1e-6)


def test_parameters_lie_in_their_ranges(key):
    n = 3
    _, _, params = make_generator(n).sample(256, NUM_SAMPLES, key)
    params = np.asarray(params)

    for columns, (low, high) in [
        (slice(0, 2 * n), (-2.0, 2.0)),  # positions, then velocities
        (2 * n, (0.3, 2.5)),  # gamma
        (2 * n + 1, (0.5, 3.0)),  # omega
        (2 * n + 2, (0.5, 1.5)),  # coupling
    ]:
        assert np.all((params[:, columns] >= low) & (params[:, columns] <= high))
