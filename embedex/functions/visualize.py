import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from embedex.functions.config import Config
from embedex.functions.embedding import FunctionEmbedding
from embedex.plotting import save_figure


def plot_samples_reconstruction(
    cfg: Config,
    embedding_network: FunctionEmbedding,
    xs: dict[str, jax.Array],
    ys: dict[str, jax.Array],
    mses: dict[str, float],
    x_range: tuple[float, float],
    suptitle: str,
    filename: str,
) -> None:
    titles = list(xs.keys())
    num_sub = len(titles)

    fig, axes = plt.subplots(1, num_sub, figsize=(5 * num_sub, 4.5))

    x_low, x_high = x_range
    margin = (x_high - x_low) * 0.05
    eval_grid = jnp.linspace(x_low - margin, x_high + margin, 256)[:, jnp.newaxis]

    for ax, title in zip(axes, titles, strict=True):
        xi, yi = xs[title], ys[title]

        for i in range(cfg.evaluation.num_plots):
            zi = embedding_network.embed(
                xi[i, : cfg.evaluation.num_eval_samples // 2],
                yi[i, : cfg.evaluation.num_eval_samples // 2],
            )
            y_pred = embedding_network.eval(eval_grid, zi)

            color = f'C{i}'
            label_gt = 'Ground truth' if i == 0 else None
            label_ctx = 'Embedded points' if i == 0 else None
            label_pred = 'Prediction' if i == 0 else None

            order = jnp.argsort(xi[i, :, 0])
            ax.plot(
                np.asarray(xi[i, order, 0]),
                np.asarray(yi[i, order, 0]),
                ':',
                color=color,
                alpha=0.5,
                label=label_gt,
            )
            ax.plot(
                np.asarray(xi[i, : cfg.evaluation.num_eval_samples // 2, 0]),
                np.asarray(yi[i, : cfg.evaluation.num_eval_samples // 2, 0]),
                'x',
                color=color,
                markersize=2,
                label=label_ctx,
            )
            ax.plot(
                np.asarray(eval_grid[:, 0]),
                np.asarray(y_pred[:, 0]),
                '-',
                color=color,
                label=label_pred,
            )

        ax.set_title(title)
        ax.set_xlim(x_low - margin, x_high + margin)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')

        if title in mses:
            ax.text(
                0.95,
                0.05,
                f'MSE = {mses[title]:.4f}',
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize='small',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            )

    axes[0].legend(loc='best', fontsize='small')
    fig.suptitle(suptitle, fontsize=13)
    save_figure(fig, os.path.join(cfg.figure_dir, cfg.run_name, f'{filename}.png'))


def plot_operator_reconstruction(
    cfg: Config,
    xs: dict[str, jax.Array],
    ys: dict[str, jax.Array],
    eval_grid: jax.Array,
    preds: dict[str, jax.Array],
    mses: dict[str, float],
    x_range: tuple[float, float],
    suptitle: str,
    gt_label: str,
    pred_label: str,
    x_label: str,
    y_label: str,
    filename: str,
) -> None:
    titles = list(xs.keys())
    num_sub = len(titles)

    fig, axes = plt.subplots(1, num_sub, figsize=(5 * num_sub, 4.5))

    x_low, x_high = x_range

    for ax, title in zip(axes, titles, strict=True):
        xi, yi, pi = xs[title], ys[title], preds[title]

        for i in range(cfg.evaluation.num_plots):
            color = f'C{i}'

            order = jnp.argsort(xi[i, :, 0])
            ax.plot(
                np.asarray(xi[i, order, 0]),
                np.asarray(yi[i, order, 0]),
                ':',
                color=color,
                alpha=0.5,
                label=gt_label if i == 0 else None,
            )
            ax.plot(
                np.asarray(eval_grid[:, 0]),
                np.asarray(pi[i, :, 0]),
                '-',
                color=color,
                label=pred_label if i == 0 else None,
            )

        ax.set_title(title)
        ax.set_xlim(x_low, x_high)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if title in mses:
            ax.text(
                0.95,
                0.05,
                f'MSE = {mses[title]:.4f}',
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize='small',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            )

    axes[0].legend(loc='best', fontsize='small')
    fig.suptitle(suptitle, fontsize=13)
    save_figure(fig, os.path.join(cfg.figure_dir, cfg.run_name, f'{filename}.png'))
