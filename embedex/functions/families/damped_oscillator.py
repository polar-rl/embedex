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
from embedex.functions.visualize import plot_phase_portrait
from embedex.logger import log_cfg


class DampedOscillatorGenerator(Generator):
    """Generator for the Damped Harmonic Oscillator ODE: x'' + 2 * gamma * x' + omega^2 * x = 0.

    Rewritten as a first-order system:
      dx/dt = v
      dv/dt = -omega^2 * x - 2 * gamma * v

    Closed-form solution: [x(t), v(t)]^T = expm(A * t) @ [x0, v0]^T,
        where A = [[0, 1], [-omega^2, -2 * gamma]]
    """

    param_size: int = 4  # (x0, v0, gamma, omega)

    def __init__(
        self,
        t_range: tuple[float, float],
        x0_range: tuple[float, float],
        v0_range: tuple[float, float],
        gamma_range: tuple[float, float],
        omega_range: tuple[float, float],
    ) -> None:
        self.t_range = t_range
        self.x0_range = x0_range
        self.v0_range = v0_range
        self.gamma_range = gamma_range
        self.omega_range = omega_range

    def _sample_one(
        self, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        t = jax.random.uniform(k1, (num_samples,), minval=self.t_range[0], maxval=self.t_range[1])
        x0 = jax.random.uniform(k2, minval=self.x0_range[0], maxval=self.x0_range[1])
        v0 = jax.random.uniform(k3, minval=self.v0_range[0], maxval=self.v0_range[1])
        gamma = jax.random.uniform(k4, minval=self.gamma_range[0], maxval=self.gamma_range[1])
        omega = jax.random.uniform(k5, minval=self.omega_range[0], maxval=self.omega_range[1])

        A = jnp.array([[0.0, 1.0], [-(omega**2), -2.0 * gamma]])
        xv = jax.vmap(lambda ti: jax.scipy.linalg.expm(A * ti) @ jnp.stack([x0, v0]))(t)
        params = jnp.stack([x0, v0, gamma, omega])

        return t[:, jnp.newaxis], xv, params

    @eqx.filter_jit
    def sample(
        self, batch_size: int, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jax.vmap(self._sample_one, in_axes=(None, 0))(
            num_samples, jax.random.split(key, batch_size)
        )

    def operator_input_size(self, cfg: DampedOscillatorConfig) -> int:
        return self.param_size

    def operator_input(
        self, embedding_network: FunctionEmbedding, x: jax.Array, y: jax.Array, params: jax.Array
    ) -> jax.Array:
        return params

    def operator_ground_truth(self, x: jax.Array, y: jax.Array, params: jax.Array) -> jax.Array:
        return y


def main(cfg: DampedOscillatorConfig) -> None:
    cfg.model.output_size = 2  # positions and velocities

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
    )
    embedding_network, operator_network = run_training_pipeline(
        cfg,
        generator,
        partial(
            evaluate_embeddings,
            plot_fn=partial(
                plot_phase_portrait,
                suptitle='Damped Harmonic Oscillator Embeddings',
                filename='damped_oscillator_embedding',
            ),
        ),
        partial(
            evaluate_solution_operator,
            plot_fn=partial(
                plot_phase_portrait,
                suptitle='Damped Harmonic Oscillator Solution Operator',
                filename='damped_oscillator_solution_operator',
            ),
        ),
    )

    save_ckpt((embedding_network, operator_network), cfg, step=cfg.training.num_iterations)
    cfg.run.stop()


if __name__ == '__main__':
    main(parse_args(DampedOscillatorConfig))
