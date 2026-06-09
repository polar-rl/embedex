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
from matplotlib.ticker import ScalarFormatter
from numpy.typing import NDArray

from embedex.distributions.domains.donuts.config import VisualizationConfig
from embedex.distributions.domains.donuts.donut import PAPER_DONUTS, Donut, sample_batch
from embedex.distributions.embedding import FlowEmbedding
from embedex.plotting import save_figure

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


def create_gradient_cmap(color: str | tuple[float, float, float]) -> Colormap:
    if isinstance(color, str):
        color = mpl.colors.to_rgb(color)

    cmap_name = f'gradient_from_{color}'
    colors = [(1.0, 1.0, 1.0, 0.0), (*color, 1.0)]

    return LinearSegmentedColormap.from_list(cmap_name, colors)


def create_donut_samples_plot(
    donuts: list[Donut],
    num_samples: int,
    scatter_size: float,
    key: jax.Array,
) -> tuple[Figure, plt.Axes]:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    data = sample_batch(donuts, num_samples, key)

    for i in range(len(donuts)):
        ax.scatter(data[i, :, 0], data[i, :, 1], s=scatter_size, marker='.')

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax


def create_density_contours_plot(
    model: FlowEmbedding,
    samples: NDArray[np.floating],
    xlims: tuple[float, float],
    ylims: tuple[float, float],
    config: VisualizationConfig,
) -> tuple[Figure, plt.Axes]:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(*xlims, config.grid_size)
    y = np.linspace(*ylims, config.grid_size)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    color_cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for sample_set, color in zip(samples, color_cycler, strict=False):
        embedding, _ = model.embed(sample_set, jnp.ones(sample_set.shape[0]))
        log_probs, _, _ = model.log_prob(grid, embedding)

        density = np.exp(np.array(log_probs))
        density = np.where(density >= config.density_threshold, density, 0)
        density = density.reshape(config.grid_size, config.grid_size)

        cmap = create_gradient_cmap(color)

        ax.contourf(xx, yy, density, levels=config.num_contour_levels, cmap=cmap, norm='log')
        ax.scatter(
            sample_set[:, 0],
            sample_set[:, 1],
            s=config.scatter_size,
            color=color,
            marker='.',
        )

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax


def generate_paper_plots(
    model: FlowEmbedding,
    save_path: str,
    config: VisualizationConfig,
    key: jax.Array,
) -> None:
    sample_key, density_key = jax.random.split(key, 2)

    print('Generating example donuts...')
    fig, ax = create_donut_samples_plot(PAPER_DONUTS, 768, config.scatter_size, sample_key)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    save_figure(fig, os.path.join(save_path, 'donuts_example.pdf'))

    print('Generating density plots...')
    true_samples = sample_batch(PAPER_DONUTS, 256, density_key)
    fig, _ = create_density_contours_plot(model, true_samples, xlims, ylims, config)
    save_figure(fig, os.path.join(save_path, 'donuts_densities.pdf'))


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
