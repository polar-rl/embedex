from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import minfx.neptune_v2 as minfx

from embedex.checkpointing import save_ckpt
from embedex.functions.embedding import FunctionEmbedding
from embedex.functions.evaluation_ode import evaluate_embeddings, evaluate_solution_operator
from embedex.functions.families.config import LogisticConfig, parse_args
from embedex.functions.generator import Generator
from embedex.functions.training import run_training_pipeline
from embedex.functions.visualize import plot_reconstruction
from embedex.logger import log_cfg


class LogisticGenerator(Generator):
    """Generator for the Logistic ODE: y' = r * y * (1 - y / K), y(0) = y0.

    Closed-form solution: y(t) = K * y0 / (y0 + (K - y0) * exp(-r * t))
    """

    param_size: int = 3  # (y0, r, K)

    def __init__(
        self,
        t_range: tuple[float, float],
        y0_range: tuple[float, float],
        r_range: tuple[float, float],
        K_range: tuple[float, float],
    ) -> None:
        self.t_range = t_range
        self.y0_range = y0_range
        self.r_range = r_range
        self.K_range = K_range

    def _sample_one(
        self, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        k1, k2, k3, k4 = jax.random.split(key, 4)

        t = jax.random.uniform(k1, (num_samples,), minval=self.t_range[0], maxval=self.t_range[1])
        y0 = jax.random.uniform(k2, minval=self.y0_range[0], maxval=self.y0_range[1])
        r = jax.random.uniform(k3, minval=self.r_range[0], maxval=self.r_range[1])
        K = jax.random.uniform(k4, minval=self.K_range[0], maxval=self.K_range[1])

        y = K * y0 / (y0 + (K - y0) * jnp.exp(-r * t))
        params = jnp.stack([y0, r, K])

        return t[:, jnp.newaxis], y[:, jnp.newaxis], params

    @eqx.filter_jit
    def sample(
        self, batch_size: int, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jax.vmap(self._sample_one, in_axes=(None, 0))(
            num_samples, jax.random.split(key, batch_size)
        )

    def operator_input_size(self, cfg: LogisticConfig) -> int:
        return self.param_size

    def operator_input(
        self, embedding_network: FunctionEmbedding, x: jax.Array, y: jax.Array, params: jax.Array
    ) -> jax.Array:
        return params

    def operator_ground_truth(self, x: jax.Array, y: jax.Array, params: jax.Array) -> jax.Array:
        return y


def main(cfg: LogisticConfig) -> None:
    cfg.run = minfx.init_run(
        project=cfg.project,
        name=cfg.run_name,
        tags=['function_embeddings', 'logistic'],
        mode='async' if not cfg.debug_mode else 'debug',
    )
    log_cfg(cfg.run, cfg)

    generator = LogisticGenerator(
        cfg.data.t_range, cfg.data.y0_range, cfg.data.r_range, cfg.data.K_range
    )
    embedding_network, operator_network = run_training_pipeline(
        cfg,
        generator,
        generator,
        partial(
            evaluate_embeddings,
            plot_fn=partial(
                plot_reconstruction,
                suptitle='Logistic ODE Embeddings',
                filename='logistic_embeddings',
            ),
        ),
        partial(
            evaluate_solution_operator,
            plot_fn=partial(
                plot_reconstruction,
                suptitle='Logistic ODE Solution Operator',
                filename='logistic_solution_operator',
            ),
        ),
    )

    save_ckpt((embedding_network, operator_network), cfg, step=cfg.training.num_iterations)
    cfg.run.stop()


if __name__ == '__main__':
    main(parse_args(LogisticConfig))
