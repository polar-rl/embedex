import jax
import jax.numpy as jnp
import minfx.neptune_v2 as minfx

from embedex.checkpointing import save_ckpt
from embedex.functions.embedding import FunctionEmbedding
from embedex.functions.families.config import AntiderivativesConfig, parse_args
from embedex.functions.families.derivatives import PolynomialGenerator, evaluate_embeddings
from embedex.functions.generator import Generator
from embedex.functions.metrics import (
    evaluate_operator_rel_l2,
    evaluate_operator_rel_max,
    evaluate_operator_rmse,
)
from embedex.functions.operator import OperatorMapping
from embedex.functions.training import run_training_pipeline
from embedex.functions.visualize import plot_operator_reconstruction
from embedex.logger import log_cfg, safe_log


class PolynomialAntiderivativeGenerator(PolynomialGenerator, Generator):
    """Data generator for the antidifferential operator: (embed(f'), x0, f(x0)) -> embed(f).

    The sample (x0, f(x0)) fixes the integration constant, which f' alone leaves undetermined.
    """

    def operator_input_size(self, cfg: AntiderivativesConfig) -> int:
        return cfg.model.embedding.embedding_size + 2

    def operator_input(
        self, embedding_network: FunctionEmbedding, x: jax.Array, y: jax.Array, dy: jax.Array
    ) -> jax.Array:
        return jnp.concatenate([embedding_network.embed(x, dy), x[0], y[0]])

    def operator_ground_truth(self, x: jax.Array, y: jax.Array, dy: jax.Array) -> jax.Array:
        return y


def evaluate_antiderivative_operator(
    cfg: AntiderivativesConfig,
    embedding_network: FunctionEmbedding,
    operator_network: OperatorMapping,
    generator: PolynomialAntiderivativeGenerator,
    key: jax.Array,
) -> None:
    num_degrees = generator.max_degree + 1

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
            safe_log(cfg.run, f'eval/operator/rmse/degree_{degree}', rmse_values[degree])
            safe_log(cfg.run, f'eval/operator/rel_l2/degree_{degree}', rel_l2_values[degree])
            safe_log(cfg.run, f'eval/operator/rel_max/degree_{degree}', rel_max_values[degree])

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
    plot_degrees = jnp.sort(
        jax.random.choice(degree_key, jnp.arange(num_degrees), (num_plots,), replace=False)
    )

    x_low, x_high = generator.x_range
    eval_grid = jnp.linspace(x_low, x_high, 256)[:, jnp.newaxis]
    half = cfg.evaluation.num_eval_samples // 2

    plot_keys = jax.random.split(key, num_plots)
    xs, ys, dys = jax.vmap(generator.sample_degree, in_axes=(0, None, None, 0))(
        plot_degrees, cfg.evaluation.num_plot_functions, cfg.evaluation.num_eval_samples, plot_keys
    )

    plot_xs, plot_ys, plot_preds = {}, {}, {}
    for i, deg in enumerate(plot_degrees):
        name = f'Degree {int(deg)}'
        plot_xs[name] = xs[i]
        plot_ys[name] = ys[i]
        sources = jax.vmap(generator.operator_input, in_axes=(None, 0, 0, 0))(
            embedding_network, xs[i][:, :half], ys[i][:, :half], dys[i][:, :half]
        )
        plot_preds[name] = jax.vmap(embedding_network.eval, in_axes=(None, 0))(
            eval_grid, jax.vmap(operator_network)(sources)
        )

    plot_operator_reconstruction(
        cfg,
        plot_xs,
        plot_ys,
        eval_grid,
        plot_preds,
        rmses,
        generator.x_range,
        "Learned Antidifferential Operator: $A(\\mathrm{embed}(f'), x_0, f(x_0)) \\approx "
        '\\mathrm{embed}(F)$',
        'Ground Truth $F$',
        'Predicted $A(z)$',
        '$x$',
        '$F(x)$',
        'polynomial_antidifferential_operator_by_degree',
    )


def main(cfg: AntiderivativesConfig) -> None:
    cfg.run = minfx.init_run(
        project=cfg.project,
        name=cfg.run_name,
        tags=['function_embeddings', 'antiderivatives'],
        mode='async' if not cfg.debug_mode else 'debug',
    )
    log_cfg(cfg.run, cfg)

    generator = PolynomialAntiderivativeGenerator(
        cfg.data.x_range, cfg.data.coeff_range, cfg.data.min_degree, cfg.data.max_degree
    )
    embedding_network, operator_network = run_training_pipeline(
        cfg, generator, generator, evaluate_embeddings, evaluate_antiderivative_operator
    )

    save_ckpt((embedding_network, operator_network), cfg, step=cfg.training.num_iterations)
    cfg.run.stop()


if __name__ == '__main__':
    main(parse_args(AntiderivativesConfig))
