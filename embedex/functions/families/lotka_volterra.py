from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import minfx.neptune_v2 as minfx

from embedex.checkpointing import save_ckpt
from embedex.functions.embedding import FunctionEmbedding
from embedex.functions.evaluation_ode import evaluate_embeddings, evaluate_solution_operator
from embedex.functions.families.config import LotkaVolterraConfig, parse_args
from embedex.functions.generator import Generator
from embedex.functions.training import run_training_pipeline
from embedex.functions.visualize import plot_phase_portrait
from embedex.logger import log_cfg


class LotkaVolterraGenerator(Generator):
    """Generator for the Lotka-Volterra predator-prey ODE system.

    Rewritten as a first-order system:
      dx/dt = alpha * x - beta * x * y   (prey)
      dy/dt = delta * x * y - gamma * y  (predator)

    The system has no elementary closed-form solution, so each trajectory
    is integrated numerically with a Dormand-Prince (Dopri5) solver.
    """

    param_size: int = 6  # (x0, y0, alpha, beta, gamma, delta)

    def __init__(
        self,
        t_range: tuple[float, float],
        x0_range: tuple[float, float],
        y0_range: tuple[float, float],
        alpha_range: tuple[float, float],
        beta_range: tuple[float, float],
        gamma_range: tuple[float, float],
        delta_range: tuple[float, float],
    ) -> None:
        self.t_range = t_range
        self.x0_range = x0_range
        self.y0_range = y0_range
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        self.delta_range = delta_range

    def _sample_one(
        self, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

        t = jax.random.uniform(k1, (num_samples,), minval=self.t_range[0], maxval=self.t_range[1])
        x0 = jax.random.uniform(k2, minval=self.x0_range[0], maxval=self.x0_range[1])
        y0 = jax.random.uniform(k3, minval=self.y0_range[0], maxval=self.y0_range[1])
        alpha = jax.random.uniform(k4, minval=self.alpha_range[0], maxval=self.alpha_range[1])
        beta = jax.random.uniform(k5, minval=self.beta_range[0], maxval=self.beta_range[1])
        gamma = jax.random.uniform(k6, minval=self.gamma_range[0], maxval=self.gamma_range[1])
        delta = jax.random.uniform(k7, minval=self.delta_range[0], maxval=self.delta_range[1])

        def vector_field(t, state, args):
            (x, y), (alpha, beta, gamma, delta) = state, args
            return jnp.stack([alpha * x - beta * x * y, delta * x * y - gamma * y])

        # diffeqsolve saves at increasing times; sort to query, then restore order.
        order = jnp.argsort(t)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Dopri5(),
            t0=self.t_range[0],
            t1=self.t_range[1],
            dt0=None,
            y0=jnp.stack([x0, y0]),
            args=(alpha, beta, gamma, delta),
            saveat=diffrax.SaveAt(ts=t[order]),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
        )
        states = solution.ys[jnp.argsort(order)]
        params = jnp.stack([x0, y0, alpha, beta, gamma, delta])

        return t[:, jnp.newaxis], states, params

    @eqx.filter_jit
    def sample(
        self, batch_size: int, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jax.vmap(self._sample_one, in_axes=(None, 0))(
            num_samples, jax.random.split(key, batch_size)
        )

    def operator_input_size(self, cfg: LotkaVolterraConfig) -> int:
        return self.param_size

    def operator_input(
        self, embedding_network: FunctionEmbedding, x: jax.Array, y: jax.Array, params: jax.Array
    ) -> jax.Array:
        return params

    def operator_ground_truth(self, x: jax.Array, y: jax.Array, params: jax.Array) -> jax.Array:
        return y


def main(cfg: LotkaVolterraConfig) -> None:
    cfg.model.output_size = 2  # prey and predator populations

    cfg.run = minfx.init_run(
        project=cfg.project,
        name=cfg.run_name,
        tags=['function_embeddings', 'lotka_volterra'],
        mode='async' if not cfg.debug_mode else 'debug',
    )
    log_cfg(cfg.run, cfg)

    generator = LotkaVolterraGenerator(
        cfg.data.t_range,
        cfg.data.x0_range,
        cfg.data.y0_range,
        cfg.data.alpha_range,
        cfg.data.beta_range,
        cfg.data.gamma_range,
        cfg.data.delta_range,
    )
    embedding_network, operator_network = run_training_pipeline(
        cfg,
        generator,
        partial(
            evaluate_embeddings,
            plot_fn=partial(
                plot_phase_portrait,
                suptitle='Lotka-Volterra Embeddings',
                filename='lotka_volterra_embedding',
                component_labels=('Prey $x(t)$', 'Predator $y(t)$'),
                phase_labels=('Prey $x$', 'Predator $y$'),
            ),
        ),
        partial(
            evaluate_solution_operator,
            plot_fn=partial(
                plot_phase_portrait,
                suptitle='Lotka-Volterra Solution Operator',
                filename='lotka_volterra_solution_operator',
                component_labels=('Prey $x(t)$', 'Predator $y(t)$'),
                phase_labels=('Prey $x$', 'Predator $y$'),
            ),
        ),
    )

    save_ckpt((embedding_network, operator_network), cfg, step=cfg.training.num_iterations)
    cfg.run.stop()


if __name__ == '__main__':
    main(parse_args(LotkaVolterraConfig))
