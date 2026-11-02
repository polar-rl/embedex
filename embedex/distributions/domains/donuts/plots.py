import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from embedex.distributions.domains.config import DonutsVisualizationConfig
from embedex.distributions.domains.donuts.donut import Donut, sample_batch
from embedex.distributions.embedding import FlowEmbedding
from embedex.plotting import apply_latex_style, save_figure

FIGURE_WIDTH = 9.5

METRICS = ['nll', 'swd', 'center_error', 'radius_error', 'width_error']
BASELINES = ['kde', 'oracle']


def is_main_grid_cell(cell: str) -> bool:
    """Main-grid cell (the intended 4) or a baseline — not an _d/_r/_w ablation variant."""
    return cell in BASELINES or re.search(r'_(d|r|w)\d+$', cell) is None


METRIC_LABELS = {
    'nll': 'Held-out NLL',
    'swd': 'Sliced Wasserstein-2',
    'center_error': 'Center error',
    'radius_error': 'Radius error',
    'width_error': 'Width error',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=str, default='results/donuts/results.jsonl')
    parser.add_argument('--figure-dir', type=str, default='figures/distributions/donuts/eval')
    parser.add_argument('--metric', nargs='*', default=METRICS, choices=METRICS)
    parser.add_argument('--format', type=str, default='pdf', choices=['pdf', 'png'])
    return parser.parse_args()


def load_curves(path: Path, metric: str) -> dict[str, dict[int, np.ndarray]]:
    curves: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if not is_main_grid_cell(row['cell']):
                continue  # ablation variants belong in the probe figures, not the grid curves
            curves[row['cell']][row['n_eval']].append(row['metrics'][metric]['mean'])

    return {
        cell: {n: np.array(means) for n, means in by_n.items()} for cell, by_n in curves.items()
    }


def plot_metric(curves: dict, metric: str, figure_dir: Path, fmt: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    all_n: set[int] = set()
    for cell in sorted(curves):
        by_n = curves[cell]
        n_vals = np.array(sorted(by_n))
        all_n.update(int(n) for n in n_vals)
        means = np.array([np.nanmean(by_n[n]) for n in n_vals])
        counts = np.array([int(np.sum(~np.isnan(by_n[n]))) for n in n_vals])
        # Drop points with no finite seeds so the line connects across the gap.
        valid = ~np.isnan(means)
        style = {'linestyle': '--', 'alpha': 0.8} if cell in BASELINES else {'linestyle': '-'}

        (line,) = ax.plot(
            n_vals[valid], means[valid], marker='o', markersize=3, label=cell, **style
        )

        ses = np.array(
            [
                np.nanstd(by_n[n], ddof=1) / np.sqrt(c) if c > 1 else 0.0
                for n, c in zip(n_vals, counts, strict=True)
            ]
        )
        if ses[valid].any():
            ax.fill_between(
                n_vals[valid],
                (means - ses)[valid],
                (means + ses)[valid],
                color=line.get_color(),
                alpha=0.15,
                lw=0,
            )

    if metric == 'nll':
        # The smallest n_eval produces wild NLL outliers; clip the view to the rest.
        inner = set(sorted(all_n)[1:])
        vals = [float(np.nanmean(by_n[n])) for by_n in curves.values() for n in by_n if n in inner]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            lo, hi = min(vals), max(vals)
            pad = 0.08 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)

    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(all_n))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel('Embedding sample count (n_eval)')
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    figure_dir.mkdir(parents=True, exist_ok=True)
    out = figure_dir / f'{metric}_vs_n_eval.{fmt}'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def view_box(points: np.ndarray, pad: float = 0.06) -> tuple[tuple, tuple]:
    """Padded (xlims, ylims) covering every point, shared across a figure's panels."""

    (x0, y0), (x1, y1) = points.min(0), points.max(0)
    px, py = pad * (x1 - x0), pad * (y1 - y0)

    return (x0 - px, x1 + px), (y0 - py, y1 + py)


def create_gradient_cmap(
    color: str | tuple[float, float, float], max_alpha: float = 1.0
) -> Colormap:
    if isinstance(color, str):
        color = mpl.colors.to_rgb(color)

    cmap_name = f'gradient_from_{color}'
    colors = [(1.0, 1.0, 1.0, 0.0), (*color, max_alpha)]

    return LinearSegmentedColormap.from_list(cmap_name, colors)


def density_on_grid(
    model: FlowEmbedding,
    embedding: jax.Array,
    xx: np.ndarray,
    yy: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Learned density on a meshgrid, floored to zero below `threshold`."""

    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    log_probs, _, _ = model.log_prob(grid, embedding)
    density = np.exp(np.asarray(log_probs)).reshape(xx.shape)

    return np.where(density >= threshold, density, 0.0)


def style_panel(
    ax: plt.Axes,
    xlims: tuple,
    ylims: tuple,
    title: str,
    xlabel: bool = True,
    ylabel: bool = True,
    label_size: float = 10.0,
    tick_size: float | None = None,
) -> None:
    ax.set_title(title, fontsize=8.4)
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_aspect('equal')

    if xlabel:
        ax.set_xlabel(r'$x_1$', fontsize=label_size)
    if ylabel:
        ax.set_ylabel(r'$x_2$', fontsize=label_size)
    if tick_size is not None:
        ax.tick_params(labelsize=tick_size)

    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


PAPER_DONUTS = [
    Donut(mean=jnp.array([-2.7, 0.5]), radius=jnp.float32(2.0), width=jnp.float32(0.45)),
    Donut(mean=jnp.array([0.25, 1.2]), radius=jnp.float32(1.4), width=jnp.float32(0.15)),
    Donut(mean=jnp.array([0.0, -0.5]), radius=jnp.float32(1.6), width=jnp.float32(0.3)),
]


@apply_latex_style
def build_donuts_figure(
    model: FlowEmbedding,
    config: DonutsVisualizationConfig,
    key: jax.Array,
) -> Figure:

    n_cond, n_gen = 32, 512
    color_cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']

    cond_key, gen_key = jax.random.split(key, 2)
    cond_all = np.asarray(sample_batch(PAPER_DONUTS, n_cond, cond_key))
    gen_keys = jax.random.split(gen_key, len(PAPER_DONUTS))

    # Embed each donut's conditioning set, then generate a fresh sample cloud from that embedding.
    embeddings, generated = [], []
    for cond, gk in zip(cond_all, gen_keys, strict=True):
        z, _ = model.embed(jnp.asarray(cond))
        gen = np.asarray(model.generate(z, n_gen, key=gk))
        embeddings.append(z)
        generated.append(gen[np.all(np.isfinite(gen), axis=1)])

    # Common, slightly padded view box covering every plotted point.
    xlims, ylims = view_box(np.concatenate([*cond_all, *generated]))

    gx = np.linspace(*xlims, config.grid_size)
    gy = np.linspace(*ylims, config.grid_size)
    xx, yy = np.meshgrid(gx, gy)

    fig, (ax_g, ax_d) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 4.7))

    for i, (cond, z, gen) in enumerate(zip(cond_all, embeddings, generated, strict=True)):
        color = color_cycler[i % len(color_cycler)]

        # The generated cloud, then the conditioning samples on top.
        ax_g.scatter(
            gen[:, 0],
            gen[:, 1],
            s=config.scatter_size,
            color=color,
            alpha=0.65,
            marker='.',
            linewidths=0,
        )
        ax_g.scatter(
            cond[:, 0],
            cond[:, 1],
            s=28,
            color=color,
            marker='X',
            edgecolor='black',
            linewidth=0.2,
            zorder=6,
        )

        # Learned density from the same embedding, with the conditioning samples overlaid.
        density = density_on_grid(model, z, xx, yy, config.density_threshold)
        ax_d.contourf(
            xx,
            yy,
            density,
            levels=max(config.num_contour_levels, 20),
            cmap=create_gradient_cmap(color, max_alpha=0.8),
            norm='log',
        )
        ax_d.scatter(
            cond[:, 0],
            cond[:, 1],
            s=28,
            color=color,
            marker='X',
            edgecolor='black',
            linewidth=0.2,
            zorder=6,
        )

    for ax, title in ((ax_g, '(a) Generated samples'), (ax_d, '(b) Learned densities')):
        style_panel(ax, xlims, ylims, title)

    handles = [
        Line2D(
            [],
            [],
            marker='X',
            color='C0',
            markeredgecolor='black',
            markeredgewidth=0.2,
            linestyle='None',
            markersize=5,
            label='Conditioning',
        ),
        Line2D([], [], marker='o', color='C0', linestyle='None', markersize=4, label='Generated'),
    ]
    ax_g.legend(
        handles=handles,
        loc='upper left',
        fontsize='xx-small',
        framealpha=0.9,
        borderpad=0.4,
        labelspacing=0.5,
        handletextpad=0.4,
    )

    return fig


def generate_paper_plots(
    model: FlowEmbedding,
    save_path: str,
    config: DonutsVisualizationConfig,
    key: jax.Array,
) -> None:
    print('Generating donut paper figure...')
    fig = build_donuts_figure(model, config, key)
    save_figure(fig, os.path.join(save_path, 'donuts_paper.pdf'))


def main() -> None:
    args = parse_args()

    results = Path(args.results)
    figure_dir = Path(args.figure_dir)

    for metric in args.metric:
        curves = load_curves(results, metric)
        if not curves:
            print(f'{metric}: no rows found, skipping')
            continue
        out = plot_metric(curves, metric, figure_dir, args.format)
        print(f'{metric}: wrote {out}')


if __name__ == '__main__':
    main()
