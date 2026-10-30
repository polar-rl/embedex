import equinox as eqx
import jax
import jax.numpy as jnp
import minfx.neptune_v2 as minfx

from embedex.checkpointing import save_ckpt
from embedex.functions.embedding import FunctionEmbedding
from embedex.functions.families.config import DerivativesConfig, parse_args
from embedex.functions.generator import Generator
from embedex.functions.metrics import (
    evaluate_operator_rel_l2,
    evaluate_operator_rel_max,
    evaluate_operator_rmse,
    evaluate_rel_l2,
    evaluate_rel_max,
    evaluate_rmse,
)
from embedex.functions.operator import OperatorMapping
from embedex.functions.training import run_training_pipeline
from embedex.functions.visualize import plot_operator_reconstruction, plot_samples_reconstruction
from embedex.logger import log_cfg, safe_log


class PolynomialGenerator:
    """Generator for random polynomials of varying degrees."""

    def __init__(
        self,
        x_range: tuple[float, float],
        coeff_range: tuple[float, float],
        min_degree: int,
        max_degree: int,
    ) -> None:
        self.x_range = x_range
        self.coeff_range = coeff_range
        self.min_degree = min_degree
        self.max_degree = max_degree

    def _random_coeffs(self, degree: jax.Array, key: jax.Array) -> jax.Array:
        """Sample coefficients for a polynomial of the given degree."""

        low, high = self.coeff_range
        coeffs = jax.random.uniform(key, (self.max_degree + 1,), minval=low, maxval=high)
        coeffs = coeffs.at[degree].set(jnp.where(coeffs[degree] == 0, 1.0, coeffs[degree]))
        mask = jnp.arange(self.max_degree + 1) <= degree

        return coeffs * mask

    def evaluate(self, x: jax.Array, coeffs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Polynomial values and derivatives at points `x` for the given coeffs."""

        def _polynomial(xi, c):
            return jnp.sum(c * jnp.power(xi, jnp.arange(c.shape[0])))

        y = jax.vmap(_polynomial, in_axes=(0, None))(x, coeffs)
        dy = jax.vmap(jax.grad(_polynomial), in_axes=(0, None))(x, coeffs)

        return y, dy

    def _sample_one(
        self, degree: jax.Array, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample a single polynomial of the given degree at random points."""

        low, high = self.x_range

        coeffs_key, sample_key = jax.random.split(key)
        coeffs = self._random_coeffs(degree, coeffs_key)

        x = jax.random.uniform(sample_key, (num_samples,), minval=low, maxval=high)
        y, dy = self.evaluate(x, coeffs)

        return x[:, jnp.newaxis], y[:, jnp.newaxis], dy[:, jnp.newaxis]

    @eqx.filter_jit
    def sample(
        self, batch_size: int, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample polynomials with degrees drawn uniformly from [min_degree, max_degree]."""

        key, degree_key = jax.random.split(key, 2)
        degrees = jax.random.randint(
            degree_key, (batch_size,), minval=self.min_degree, maxval=self.max_degree + 1
        )

        return jax.vmap(self._sample_one, in_axes=(0, None, 0))(
            degrees, num_samples, jax.random.split(key, batch_size)
        )

    @eqx.filter_jit
    def sample_degree(
        self, degree: int, batch_size: int, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jax.vmap(self._sample_one, in_axes=(None, None, 0))(
            degree, num_samples, jax.random.split(key, batch_size)
        )


class PolynomialDerivativeGenerator(PolynomialGenerator, Generator):
    """Data generator for learning the differential operator: embed(f) -> embed(f')."""

    def operator_input_size(self, cfg: DerivativesConfig) -> int:
        return cfg.model.embedding_size

    def operator_input(
        self, embedding_network: FunctionEmbedding, x: jax.Array, y: jax.Array, dy: jax.Array
    ) -> jax.Array:
        return embedding_network.embed(x, y)

    def operator_ground_truth(self, x: jax.Array, y: jax.Array, dy: jax.Array) -> jax.Array:
        return dy


def evaluate_embeddings(
    cfg: DerivativesConfig,
    embedding_network: FunctionEmbedding,
    generator: PolynomialGenerator,
    key: jax.Array,
) -> None:
    num_degrees = generator.max_degree + 1

    def _eval_degree(degree, key):
        xs, ys, *_ = generator.sample_degree(
            degree, cfg.evaluation.num_eval_functions, cfg.evaluation.num_eval_samples, key
        )

        return (
            evaluate_rmse(embedding_network, xs, ys),
            evaluate_rel_l2(embedding_network, xs, ys),
            evaluate_rel_max(embedding_network, xs, ys),
        )

    eval_keys = jax.random.split(key, num_degrees)
    rmse_values, rel_l2_values, rel_max_values = jax.vmap(_eval_degree)(
        jnp.arange(num_degrees), eval_keys
    )

    rmses = {}
    for degree in range(num_degrees):
        rmses[f'Degree {degree}'] = rmse_values[degree]
        print(
            f'Degree {degree}: RMSE: {rmse_values[degree]:.8g}; '
            f'Rel L2: {rel_l2_values[degree]:.8g}; Rel Max: {rel_max_values[degree]:.8g}'
        )

        if hasattr(cfg, 'run'):
            safe_log(cfg.run, f'eval/embedding/rmse/degree_{degree}', rmse_values[degree])
            safe_log(cfg.run, f'eval/embedding/rel_l2/degree_{degree}', rel_l2_values[degree])
            safe_log(cfg.run, f'eval/embedding/rel_max/degree_{degree}', rel_max_values[degree])

    overall_rmse = jnp.mean(rmse_values)
    overall_rel_l2 = jnp.mean(rel_l2_values)
    overall_rel_max = jnp.mean(rel_max_values)
    print(
        f'Embedding: RMSE: {overall_rmse:.8g}; '
        f'Rel L2: {overall_rel_l2:.8g}; Rel Max: {overall_rel_max:.8g}'
    )

    if hasattr(cfg, 'run'):
        safe_log(cfg.run, 'eval/embedding/rmse/overall', overall_rmse)
        safe_log(cfg.run, 'eval/embedding/rel_l2/overall', overall_rel_l2)
        safe_log(cfg.run, 'eval/embedding/rel_max/overall', overall_rel_max)

    num_plots = min(4, num_degrees)

    key, degree_key = jax.random.split(key, 2)
    plot_degrees = jnp.sort(
        jax.random.choice(degree_key, jnp.arange(num_degrees), (num_plots,), replace=False)
    )

    plot_keys = jax.random.split(key, num_plots)
    xs, ys, _ = jax.vmap(generator.sample_degree, in_axes=(0, None, None, 0))(
        plot_degrees, cfg.evaluation.num_eval_plots, cfg.evaluation.num_eval_samples, plot_keys
    )

    plot_xs = {f'Degree {int(deg)}': xs[i] for i, deg in enumerate(plot_degrees)}
    plot_ys = {f'Degree {int(deg)}': ys[i] for i, deg in enumerate(plot_degrees)}

    plot_samples_reconstruction(
        cfg,
        embedding_network,
        plot_xs,
        plot_ys,
        rmses,
        generator.x_range,
        'Polynomial Embeddings',
        'polynomial_embeddings_by_degree',
    )


def evaluate_differential_operator(
    cfg: DerivativesConfig,
    embedding_network: FunctionEmbedding,
    operator_network: OperatorMapping,
    generator: PolynomialDerivativeGenerator,
    key: jax.Array,
) -> None:
    degrees = jnp.arange(generator.min_degree, generator.max_degree + 1)
    num_degrees = int(degrees.shape[0])

    def _eval_degree(degree, key):
        xs, ys, dys = generator.sample_degree(
            degree, cfg.evaluation.num_eval_functions, cfg.evaluation.num_eval_samples, key
        )
        rmse = evaluate_operator_rmse(embedding_network, operator_network, generator, xs, ys, dys)
        rel_l2 = evaluate_operator_rel_l2(
            embedding_network, operator_network, generator, xs, ys, dys
        )
        rel_max = evaluate_operator_rel_max(
            embedding_network, operator_network, generator, xs, ys, dys
        )

        return rmse, rel_l2, rel_max

    eval_keys = jax.random.split(key, num_degrees)
    rmse_values, rel_l2_values, rel_max_values = jax.vmap(_eval_degree)(degrees, eval_keys)

    rmses = {}
    for i in range(num_degrees):
        degree = int(degrees[i])
        rmses[f'Degree {degree}'] = rmse_values[i]
        print(
            f'Degree {degree}: RMSE: {rmse_values[i]:.8g}; '
            f'Rel L2: {rel_l2_values[i]:.8g}; Rel Max: {rel_max_values[i]:.8g}'
        )

        if hasattr(cfg, 'run'):
            safe_log(cfg.run, f'eval/operator/rmse/degree_{degree}', rmse_values[i])
            safe_log(cfg.run, f'eval/operator/rel_l2/degree_{degree}', rel_l2_values[i])
            safe_log(cfg.run, f'eval/operator/rel_max/degree_{degree}', rel_max_values[i])

    overall_rmse = jnp.mean(rmse_values)
    overall_rel_l2 = jnp.mean(rel_l2_values)
    overall_rel_max = jnp.mean(rel_max_values)
    print(
        f'Operator: RMSE: {overall_rmse:.8g}; '
        f'Rel L2: {overall_rel_l2:.8g}; Rel Max: {overall_rel_max:.8g}'
    )

    if hasattr(cfg, 'run'):
        safe_log(cfg.run, 'eval/operator/rmse/overall', overall_rmse)
        safe_log(cfg.run, 'eval/operator/rel_l2/overall', overall_rel_l2)
        safe_log(cfg.run, 'eval/operator/rel_max/overall', overall_rel_max)

    num_plots = min(4, num_degrees)

    key, degree_key = jax.random.split(key, 2)
    plot_degrees = jnp.sort(jax.random.choice(degree_key, degrees, (num_plots,), replace=False))

    x_low, x_high = generator.x_range
    eval_grid = jnp.linspace(x_low, x_high, 256)[:, jnp.newaxis]
    half = cfg.evaluation.num_eval_samples // 2

    plot_keys = jax.random.split(key, num_plots)
    xs, ys, dys = jax.vmap(generator.sample_degree, in_axes=(0, None, None, 0))(
        plot_degrees, cfg.evaluation.num_eval_plots, cfg.evaluation.num_eval_samples, plot_keys
    )

    plot_xs, plot_dys, plot_preds = {}, {}, {}
    for i, deg in enumerate(plot_degrees):
        name = f'Degree {int(deg)}'
        plot_xs[name] = xs[i]
        plot_dys[name] = dys[i]
        zs = jax.vmap(embedding_network.embed)(xs[i][:, :half], ys[i][:, :half])
        plot_preds[name] = jax.vmap(embedding_network.eval, in_axes=(None, 0))(
            eval_grid, jax.vmap(operator_network)(zs)
        )

    plot_operator_reconstruction(
        cfg,
        plot_xs,
        plot_dys,
        eval_grid,
        plot_preds,
        rmses,
        generator.x_range,
        "Learned Differential Operator: $D(\\mathrm{embed}(f)) \\approx \\mathrm{embed}(f')$",
        "Ground Truth $f'$",
        'Predicted $D(\\mathrm{embed}(f))$',
        '$x$',
        "$f'(x)$",
        'polynomial_differential_operator_by_degree',
    )


def main(cfg: DerivativesConfig) -> None:
    cfg.run = minfx.init_run(
        project=cfg.project,
        name=cfg.run_name,
        tags=['function_embeddings', 'derivatives'],
        mode='async' if not cfg.debug_mode else 'debug',
    )
    log_cfg(cfg.run, cfg)

    def _polynomials(min_degree):
        return PolynomialDerivativeGenerator(
            cfg.data.x_range, cfg.data.coeff_range, min_degree, cfg.data.max_degree
        )

    embedding_network, operator_network = run_training_pipeline(
        cfg,
        _polynomials(cfg.data.min_degree),
        _polynomials(cfg.data.operator_min_degree),
        evaluate_embeddings,
        evaluate_differential_operator,
    )

    save_ckpt((embedding_network, operator_network), cfg, step=cfg.training.num_iterations)
    cfg.run.stop()


if __name__ == '__main__':
    main(parse_args(DerivativesConfig))
