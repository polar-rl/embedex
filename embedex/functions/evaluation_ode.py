from collections.abc import Callable

import jax
import jax.numpy as jnp

from embedex.functions.config import Config
from embedex.functions.embedding import FunctionEmbedding
from embedex.functions.generator import Generator
from embedex.functions.metrics import (
    evaluate_mse,
    evaluate_operator_mse,
    evaluate_operator_rel_l2,
    evaluate_rel_l2,
)
from embedex.functions.operator import OperatorMapping
from embedex.logger import safe_log


def evaluate_embeddings(
    cfg: Config,
    embedding_network: FunctionEmbedding,
    generator: Generator,
    plot_fn: Callable,
    key: jax.Array,
) -> None:
    """Empirical reconstruction metrics over many functions, plus a few plotted examples."""

    eval_key, plot_key = jax.random.split(key, 2)

    ts, ys, *_ = generator.sample(
        cfg.evaluation.num_eval_functions, cfg.evaluation.num_eval_samples, eval_key
    )
    mse = evaluate_mse(embedding_network, ts, ys)
    rel_l2 = evaluate_rel_l2(embedding_network, ts, ys)
    report_metrics(cfg, 'embedding', mse, rel_l2)

    ts_plot, ys_plot, *_ = generator.sample(
        cfg.evaluation.num_eval_plots, cfg.evaluation.num_eval_samples, plot_key
    )
    ts_plot_embed = ts_plot[:, : cfg.evaluation.num_eval_samples // 2]
    ys_plot_embed = ys_plot[:, : cfg.evaluation.num_eval_samples // 2]

    eval_grid = jnp.linspace(*generator.t_range, 256)[:, jnp.newaxis]
    eval_preds = jax.vmap(embedding_network.eval, in_axes=(None, 0))(
        eval_grid, jax.vmap(embedding_network.embed)(ts_plot_embed, ys_plot_embed)
    )

    plot_fn(cfg, ts_plot, ys_plot, eval_grid, eval_preds, mse)


def evaluate_solution_operator(
    cfg: Config,
    embedding_network: FunctionEmbedding,
    operator_network: OperatorMapping,
    generator: Generator,
    plot_fn: Callable,
    key: jax.Array,
) -> None:
    """Empirical operator metrics over many functions, plus a few plotted examples."""

    eval_key, plot_key = jax.random.split(key, 2)

    ts, ys, params = generator.sample(
        cfg.evaluation.num_eval_functions, cfg.evaluation.num_eval_samples, eval_key
    )
    mse = evaluate_operator_mse(embedding_network, operator_network, generator, ts, ys, params)
    rel_l2 = evaluate_operator_rel_l2(
        embedding_network, operator_network, generator, ts, ys, params
    )
    report_metrics(cfg, 'operator', mse, rel_l2)

    ts_plot, ys_plot, params_plot = generator.sample(
        cfg.evaluation.num_eval_plots, cfg.evaluation.num_eval_samples, plot_key
    )
    sources = jax.vmap(generator.operator_input, in_axes=(None, 0, 0, 0))(
        embedding_network, ts_plot, ys_plot, params_plot
    )

    eval_grid = jnp.linspace(*generator.t_range, 256)[:, jnp.newaxis]
    eval_preds = jax.vmap(embedding_network.eval, in_axes=(None, 0))(
        eval_grid, jax.vmap(operator_network)(sources)
    )

    plot_fn(cfg, ts_plot, ys_plot, eval_grid, eval_preds, mse)


def report_metrics(cfg: Config, stage: str, mse: jax.Array, rel_l2: jax.Array) -> None:
    print(f'{stage.capitalize()}: MSE: {mse:.6f}; Rel L2: {rel_l2:.6f}')

    if hasattr(cfg, 'run'):
        safe_log(cfg.run, f'eval/{stage}/mse', mse)
        safe_log(cfg.run, f'eval/{stage}/rel_l2', rel_l2)
