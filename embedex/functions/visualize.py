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

        for i in range(cfg.evaluation.num_eval_plots):
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
            _annotate_mse(ax, mses[title])

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

        for i in range(cfg.evaluation.num_eval_plots):
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
            _annotate_mse(ax, mses[title])

    axes[0].legend(loc='best', fontsize='small')
    fig.suptitle(suptitle, fontsize=13)
    save_figure(fig, os.path.join(cfg.figure_dir, cfg.run_name, f'{filename}.png'))


def plot_reconstruction(
    cfg: Config,
    ts: jax.Array,
    ys: jax.Array,
    eval_grid: jax.Array,
    preds: jax.Array,
    mse: float,
    *,
    suptitle: str,
    filename: str,
    gt_label: str = 'Ground truth',
    pred_label: str = 'Prediction',
    x_label: str = '$t$',
    y_label: str = '$y(t)$',
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    t_pred = np.asarray(eval_grid[:, 0])

    for i in range(ts.shape[0]):
        color = f'C{i}'
        order = jnp.argsort(ts[i, :, 0])

        ax.plot(
            np.asarray(ts[i, order, 0]),
            np.asarray(ys[i, order, 0]),
            ':',
            color=color,
            alpha=0.6,
            label=gt_label if i == 0 else None,
        )
        ax.plot(
            t_pred,
            np.asarray(preds[i, :, 0]),
            '-',
            color=color,
            label=pred_label if i == 0 else None,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='best', fontsize='small')

    _annotate_mse(ax, mse)

    fig.suptitle(suptitle, fontsize=13)
    save_figure(fig, os.path.join(cfg.figure_dir, cfg.run_name, f'{filename}.png'))


def plot_phase_portrait(
    cfg: Config,
    ts: jax.Array,
    ys: jax.Array,
    eval_grid: jax.Array,
    preds: jax.Array,
    mse: float,
    *,
    suptitle: str,
    filename: str,
    gt_label: str = 'Ground Truth Solution',
    pred_label: str = 'Predicted Solution',
    component_labels: tuple[str, str] = ('$x(t)$', '$v(t)$'),
    phase_labels: tuple[str, str] = ('$x$', '$v$'),
) -> None:
    fig, (ax_0, ax_1, ax_phase) = plt.subplots(1, 3, figsize=(15, 4.5))
    t_pred = np.asarray(eval_grid[:, 0])

    for i in range(ts.shape[0]):
        color = f'C{i}'
        order = jnp.argsort(ts[i, :, 0])

        t_gt = np.asarray(ts[i, order, 0])
        c0_gt, c1_gt = np.asarray(ys[i, order, 0]), np.asarray(ys[i, order, 1])
        c0_pr, c1_pr = np.asarray(preds[i, :, 0]), np.asarray(preds[i, :, 1])

        ax_0.plot(t_gt, c0_gt, ':', color=color, alpha=0.5, label=gt_label if i == 0 else None)
        ax_0.plot(t_pred, c0_pr, '-', color=color, label=pred_label if i == 0 else None)

        ax_1.plot(t_gt, c1_gt, ':', color=color, alpha=0.5, label=gt_label if i == 0 else None)
        ax_1.plot(t_pred, c1_pr, '-', color=color, label=pred_label if i == 0 else None)

        ax_phase.plot(c0_gt, c1_gt, ':', color=color, alpha=0.5, label=gt_label if i == 0 else None)
        ax_phase.plot(c0_pr, c1_pr, '-', color=color, label=pred_label if i == 0 else None)

    ax_0.set_title(component_labels[0])
    ax_0.set_xlabel('$t$')
    ax_0.set_ylabel(component_labels[0])
    ax_0.legend(loc='best', fontsize='small')

    ax_1.set_title(component_labels[1])
    ax_1.set_xlabel('$t$')
    ax_1.set_ylabel(component_labels[1])
    ax_1.legend(loc='best', fontsize='small')

    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel(phase_labels[0])
    ax_phase.set_ylabel(phase_labels[1])
    ax_phase.legend(loc='best', fontsize='small')
    ax_phase.set_aspect('equal', adjustable='datalim')

    _annotate_mse(ax_0, mse)

    fig.suptitle(suptitle, fontsize=13)
    save_figure(fig, os.path.join(cfg.figure_dir, cfg.run_name, f'{filename}.png'))


def plot_oscillator_reconstruction(
    cfg: Config,
    ts: jax.Array,
    ys: jax.Array,
    eval_grid: jax.Array,
    preds: jax.Array,
    mse: float,
    *,
    num_masses: int,
    suptitle: str,
    filename: str,
) -> None:
    if num_masses == 1:
        plot_phase_portrait(
            cfg, ts, ys, eval_grid, preds, mse, suptitle=suptitle, filename=filename
        )

        return None

    num_shown = min(num_masses, 4)
    fig, axes = plt.subplots(1, num_shown, figsize=(5 * num_shown, 4.5))
    t_pred = np.asarray(eval_grid[:, 0])

    for mass, ax in enumerate(axes):
        for i in range(ts.shape[0]):
            color = f'C{i}'
            order = jnp.argsort(ts[i, :, 0])

            ax.plot(
                np.asarray(ts[i, order, 0]),
                np.asarray(ys[i, order, mass]),
                ':',
                color=color,
                alpha=0.5,
                label='Ground truth' if i == 0 else None,
            )
            ax.plot(
                t_pred,
                np.asarray(preds[i, :, mass]),
                '-',
                color=color,
                label='Prediction' if i == 0 else None,
            )

        ax.set_title(f'Mass {mass + 1}')
        ax.set_xlabel('$t$')
        ax.set_ylabel(f'$x_{{{mass + 1}}}(t)$')

    axes[0].legend(loc='best', fontsize='small')
    _annotate_mse(axes[0], mse)

    fig.suptitle(suptitle, fontsize=13)
    save_figure(fig, os.path.join(cfg.figure_dir, cfg.run_name, f'{filename}.png'))


def _annotate_mse(ax: plt.Axes, mse: float) -> None:
    ax.text(
        0.95,
        0.05,
        f'MSE = {float(mse):.4f}',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize='small',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )
