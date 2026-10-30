from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import minfx.neptune_v2 as minfx

from embedex.checkpointing import save_ckpt
from embedex.functions.embedding import FunctionEmbedding
from embedex.functions.evaluation_ode import evaluate_embeddings, evaluate_solution_operator
from embedex.functions.families.config import DampedOscillatorConfig, parse_args
from embedex.functions.generator import Generator
from embedex.functions.training import run_training_pipeline
from embedex.functions.visualize import plot_oscillator_reconstruction
from embedex.logger import log_cfg


class DampedOscillatorGenerator(Generator):
    """Generator for a chain of `num_masses` coupled damped harmonic oscillators.

    The governing equation at `num_masses=1` is:
        x'' + 2 * gamma * x' + omega^2 * x = 0.

    Rewritten as a first-order system, this becomes:
        dx/dt = v
        dv/dt = -omega^2 * x - 2 * gamma * v

    Unit masses on a line, each tied to its rest position by an on-site spring
    (frequency `omega`) and to its neighbours by coupling springs (stiffness `k`),
    under uniform velocity damping `gamma`. Stacking positions and velocities
    into z = [x, v] gives the linear system:

        z' = A * z,  A = [[0, I], [-K, -2 * gamma * I]],

    with K the tridiagonal stiffness matrix and a closed-form solution z(t) = expm(A * t) @ z0.
    """

    def __init__(
        self,
        t_range: tuple[float, float],
        x0_range: tuple[float, float],
        v0_range: tuple[float, float],
        gamma_range: tuple[float, float],
        omega_range: tuple[float, float],
        coupling_range: tuple[float, float],
        num_masses: int,
    ) -> None:
        self.t_range = t_range
        self.x0_range = x0_range
        self.v0_range = v0_range
        self.gamma_range = gamma_range
        self.omega_range = omega_range
        self.coupling_range = coupling_range
        self.num_masses = num_masses

    def _system_matrix(self, gamma: jax.Array, omega: jax.Array, k: jax.Array) -> jax.Array:
        """Assemble A = [[0, I], [-K, -2 * gamma * I]] for the coupled chain."""

        n = self.num_masses
        identity = jnp.eye(n)

        if n == 1:
            stiffness = jnp.reshape(omega**2, (1, 1))
        else:
            adjacency = jnp.eye(n, k=1) + jnp.eye(n, k=-1)
            stiffness = jnp.diag(omega**2 + k * jnp.sum(adjacency, axis=1)) - k * adjacency

        return jnp.block([[jnp.zeros((n, n)), identity], [-stiffness, -2.0 * gamma * identity]])

    def trajectory(self, t: jax.Array, params: jax.Array) -> jax.Array:
        n = self.num_masses

        x0, v0 = params[:n], params[n : 2 * n]
        gamma, omega, k = params[2 * n], params[2 * n + 1], params[2 * n + 2]

        A = self._system_matrix(gamma, omega, k)
        z0 = jnp.concatenate([x0, v0])

        return jax.vmap(lambda ti: jax.scipy.linalg.expm(A * ti) @ z0)(t)

    def _sample_one(
        self, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

        n = self.num_masses
        t = jax.random.uniform(k1, (num_samples,), minval=self.t_range[0], maxval=self.t_range[1])
        x0 = jax.random.uniform(k2, (n,), minval=self.x0_range[0], maxval=self.x0_range[1])
        v0 = jax.random.uniform(k3, (n,), minval=self.v0_range[0], maxval=self.v0_range[1])
        gamma = jax.random.uniform(k4, minval=self.gamma_range[0], maxval=self.gamma_range[1])
        omega = jax.random.uniform(k5, minval=self.omega_range[0], maxval=self.omega_range[1])
        k = jax.random.uniform(k6, minval=self.coupling_range[0], maxval=self.coupling_range[1])

        params = jnp.concatenate([x0, v0, jnp.stack([gamma, omega, k])])

        return t[:, jnp.newaxis], self.trajectory(t, params), params

    @eqx.filter_jit
    def sample(
        self, batch_size: int, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jax.vmap(self._sample_one, in_axes=(None, 0))(
            num_samples, jax.random.split(key, batch_size)
        )

    def operator_input_size(self, cfg: DampedOscillatorConfig) -> int:
        return 2 * self.num_masses + 3  # (x0, v0) per mass, plus (gamma, omega, coupling)

    def operator_input(
        self, embedding_network: FunctionEmbedding, x: jax.Array, y: jax.Array, params: jax.Array
    ) -> jax.Array:
        return params

    def operator_ground_truth(self, x: jax.Array, y: jax.Array, params: jax.Array) -> jax.Array:
        return y


def main(cfg: DampedOscillatorConfig) -> None:
    cfg.model.output_size = 2 * cfg.data.num_masses  # positions and velocities of every mass

    cfg.run = minfx.init_run(
        project=cfg.project,
        name=cfg.run_name,
        tags=['function_embeddings', 'damped_oscillator'],
        mode='async' if not cfg.debug_mode else 'debug',
    )
    log_cfg(cfg.run, cfg)

    generator = DampedOscillatorGenerator(
        cfg.data.t_range,
        cfg.data.x0_range,
        cfg.data.v0_range,
        cfg.data.gamma_range,
        cfg.data.omega_range,
        cfg.data.coupling_range,
        cfg.data.num_masses,
    )
    embedding_network, operator_network = run_training_pipeline(
        cfg,
        generator,
        generator,
        partial(
            evaluate_embeddings,
            plot_fn=partial(
                plot_oscillator_reconstruction,
                num_masses=cfg.data.num_masses,
                suptitle='Damped Harmonic Oscillator Embeddings',
                filename='damped_oscillator_embedding',
            ),
        ),
        partial(
            evaluate_solution_operator,
            plot_fn=partial(
                plot_oscillator_reconstruction,
                num_masses=cfg.data.num_masses,
                suptitle='Damped Harmonic Oscillator Solution Operator',
                filename='damped_oscillator_solution_operator',
            ),
        ),
    )

    save_ckpt((embedding_network, operator_network), cfg, step=cfg.training.num_iterations)
    cfg.run.stop()


if __name__ == '__main__':
    main(parse_args(DampedOscillatorConfig))
