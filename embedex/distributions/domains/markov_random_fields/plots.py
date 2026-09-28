import argparse
import os
from collections import defaultdict
from pathlib import Path

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from embedex.distributions.domains.experiment_utils import (
    apply_plot_style,
    load_rows,
    mean_se,
    safe_nanmean,
)
from embedex.distributions.domains.markov_random_fields.aggregate import (
    BASELINES,
    CELL_DISPLAY,
    FLOOR,
    MODEL_CELLS,
    base_cell,
)
from embedex.distributions.domains.markov_random_fields.config import (
    MRFDistributionConfig,
    VisualizationConfig,
)
from embedex.distributions.domains.markov_random_fields.mrf import MRF
from embedex.distributions.domains.markov_random_fields.pmfs import empirical_pmf
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.metrics import js_divergence
from embedex.plotting import save_figure

SERIES = [*MODEL_CELLS, FLOOR]


def _random_mrf(distribution: MRFDistributionConfig, key: jax.Array) -> MRF:
    return MRF.random(
        num_spins=distribution.num_spins,
        num_states=distribution.num_states,
        coupling_scale=distribution.coupling_scale,
        bias_scale=distribution.bias_scale,
        edge_prob=distribution.edge_prob,
        edge_prob_min=distribution.edge_prob_min,
        edge_prob_max=distribution.edge_prob_max,
        graph_type=distribution.graph_type,
        key=key,
    )


def reconstruct(
    model: FlowEmbedding,
    mrf: MRF,
    num_embedding_samples: int,
    num_generated_samples: int,
    key: jax.Array,
) -> np.ndarray:
    sample_key, gen_key = jax.random.split(key, 2)

    samples = mrf.sample(num_embedding_samples, sample_key)
    embedding, _ = model.embed(samples)
    generated = model.generate(embedding, num_generated_samples, key=gen_key)

    return np.clip(np.round(np.array(generated)), 0, mrf.num_states - 1).astype(int)


def _agreement(samples: np.ndarray) -> np.ndarray:
    return (samples[:, :, None] == samples[:, None, :]).mean(axis=0)


def _edge_descriptors(coupling: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = coupling.shape[-1]
    strength = np.linalg.norm(coupling, axis=(2, 3))
    diag = np.einsum('ijaa->ij', coupling)
    off = coupling.sum(axis=(2, 3)) - diag
    signed = diag / q - off / max(q * (q - 1), 1)
    return strength, signed


def draw_coupling_graph(ax: plt.Axes, coupling: np.ndarray) -> None:
    n = coupling.shape[0]
    angles = 2.0 * np.pi * np.arange(n) / n
    xs, ys = np.cos(angles), np.sin(angles)

    strength, signed = _edge_descriptors(coupling)
    smax = max(float(strength.max()), 1e-8)
    cmax = max(float(np.abs(signed).max()), 1e-8)
    threshold = 1e-3 * smax
    cmap = plt.cm.RdBu  # blue = favours matching, red = favours mismatching
    norm = mpl.colors.Normalize(vmin=-cmax, vmax=cmax)

    for i in range(n):
        for j in range(i + 1, n):
            if strength[i, j] <= threshold:
                continue
            ax.plot(
                [xs[i], xs[j]],
                [ys[i], ys[j]],
                color=cmap(norm(signed[i, j])),
                linewidth=0.5 + 4.0 * strength[i, j] / smax,
                alpha=0.8,
                zorder=1,
            )

    ax.scatter(xs, ys, s=220, c='white', edgecolors='black', linewidths=1.5, zorder=2)
    for i in range(n):
        ax.text(xs[i], ys[i], str(i), ha='center', va='center', fontsize=7, zorder=3)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Coupling graph (J)')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.8)


def _agreement_norm(num_states: int) -> mpl.colors.Normalize:
    return mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=1.0 / num_states, vmax=1.0)


def _triptych(
    true_mat: np.ndarray,
    recon_mat: np.ndarray,
    *,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    yticks: range | None = None,
    aspect: str | None = None,
    cmap: str = 'Blues',
    norm: mpl.colors.Normalize | None = None,
) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    kw = {} if aspect is None else {'aspect': aspect}
    scale_kw = {'norm': norm} if norm is not None else {'vmin': 0, 'vmax': 1}

    im0 = axes[0].imshow(true_mat, cmap=cmap, **scale_kw, **kw)
    axes[0].set_title('True')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(recon_mat, cmap=cmap, **scale_kw, **kw)
    axes[1].set_title('Reconstructed')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    diff = true_mat - recon_mat
    dmax = max(abs(diff.min()), abs(diff.max()), 0.01)
    im2 = axes[2].imshow(diff, cmap='RdBu', vmin=-dmax, vmax=dmax, **kw)
    axes[2].set_title('Difference (true - recon)')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    for ax in axes:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if yticks is not None:
            ax.set_yticks(yticks)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def create_agreement_figure(
    true_agree: np.ndarray, recon_agree: np.ndarray, num_spins: int, num_states: int
) -> Figure:
    return _triptych(
        true_agree,
        recon_agree,
        title=f'Agreement marginals P(xi = xj)  (N={num_spins}, q={num_states})',
        xlabel='Node',
        ylabel='Node',
        cmap='RdBu',
        norm=_agreement_norm(num_states),
    )


def create_single_marginals_figure(
    true_single: np.ndarray, recon_single: np.ndarray, num_spins: int, num_states: int
) -> Figure:
    return _triptych(
        true_single.T,
        recon_single.T,
        title=f'Single-node marginals P(xi = c)  (N={num_spins}, q={num_states})',
        xlabel='Node',
        ylabel='Class',
        yticks=range(num_states),
        aspect='auto',
    )


def create_coupling_graph_figure(coupling: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_coupling_graph(ax, coupling)
    fig.tight_layout()
    return fig


def create_samples_figure(gen: np.ndarray, num_states: int, num_show: int) -> Figure:
    show = gen[:num_show]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        show,
        cmap=plt.get_cmap('viridis', num_states),
        aspect='auto',
        interpolation='nearest',
        vmin=-0.5,
        vmax=num_states - 0.5,
    )
    ax.set_title(f'Generated samples ({len(show)})')
    ax.set_xlabel('Node')
    ax.set_ylabel('Sample')
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, ticks=range(num_states), shrink=0.8)
    cbar.set_label('Class')
    fig.tight_layout()
    return fig


def create_reconstruction_grid(
    model: FlowEmbedding,
    distribution: MRFDistributionConfig,
    config: VisualizationConfig,
    key: jax.Array,
) -> Figure:
    num_plots = config.grid_num_distributions
    ncols = 4
    nrows = (num_plots + ncols // 2 - 1) // (ncols // 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    keys = jax.random.split(key, num_plots)

    for idx in range(num_plots):
        mrf_key, recon_key = jax.random.split(keys[idx], 2)
        mrf = _random_mrf(distribution, mrf_key)
        true_agree = np.array(mrf.agreement_marginals())

        gen = reconstruct(
            model, mrf, config.num_embedding_samples, config.num_generated_samples, recon_key
        )
        recon_agree = _agreement(gen)
        jsd = float(js_divergence(mrf.pmf(), empirical_pmf(gen, mrf.num_spins, mrf.num_states)))

        row = idx // (ncols // 2)
        col_base = (idx % (ncols // 2)) * 2

        agree_norm = _agreement_norm(mrf.num_states)
        axes[row, col_base].imshow(true_agree, cmap='RdBu', norm=agree_norm)
        axes[row, col_base].set_title('True', fontsize=9)
        axes[row, col_base + 1].imshow(recon_agree, cmap='RdBu', norm=agree_norm)
        axes[row, col_base + 1].set_title(f'Recon (JSD={jsd:.3f})', fontsize=9)

        for ax in (axes[row, col_base], axes[row, col_base + 1]):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('MRF Agreement Structure: True vs Reconstructed', fontsize=14)
    fig.tight_layout()

    return fig


def generate_paper_plots(
    model: FlowEmbedding,
    save_path: str,
    config: VisualizationConfig,
    distribution: MRFDistributionConfig,
    key: jax.Array,
) -> None:
    single_key, grid_key = jax.random.split(key, 2)

    print('Generating belief reconstruction figures...')
    mrf_key, recon_key = jax.random.split(single_key, 2)
    mrf = _random_mrf(distribution, mrf_key)
    q = mrf.num_states

    gen = reconstruct(
        model, mrf, config.num_embedding_samples, config.num_generated_samples, recon_key
    )

    true_agree = np.array(mrf.agreement_marginals())
    recon_agree = _agreement(gen)
    true_single = np.array(mrf.single_marginals())
    recon_single = np.stack([(gen == c).mean(axis=0) for c in range(q)], axis=1)
    coupling = np.array(mrf.J)

    figures = {
        'mrf_pairwise_marginals.pdf': create_agreement_figure(
            true_agree, recon_agree, mrf.num_spins, q
        ),
        'mrf_class_marginals.pdf': create_single_marginals_figure(
            true_single, recon_single, mrf.num_spins, q
        ),
        'mrf_coupling_graph.pdf': create_coupling_graph_figure(coupling),
        'mrf_samples.pdf': create_samples_figure(gen, q, config.sample_strip_num),
    }
    for name, fig in figures.items():
        save_figure(fig, os.path.join(save_path, name))
        plt.close(fig)

    print('Generating reconstruction grid...')
    fig = create_reconstruction_grid(model, distribution, config, grid_key)
    save_figure(fig, os.path.join(save_path, 'mrf_reconstruction_grid.pdf'))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results/markov_random_fields/results.jsonl')
    parser.add_argument(
        '--figure-dir', type=str, default='figures/distributions/markov_random_fields/eval'
    )
    return parser.parse_args()


def coupling_rows(rows: list[dict]) -> dict[tuple[str, float], list[float]]:
    """Per-seed mean JSD keyed by (series, coupling), over the coupling-dial families."""
    series_rows = defaultdict(list)
    for row in rows:
        dist = row['distribution']
        if dist['num_states'] != 2 or row['cell'] in BASELINES:
            continue
        series = FLOOR if row['cell'] == FLOOR else base_cell(row['cell'])
        if series is None:
            continue
        key = (series, float(dist['coupling_scale']))
        series_rows[key].append(safe_nanmean(row['metrics']['jsd']['per_dist']))
    return series_rows


def create_coupling_figure(series_rows: dict[tuple[str, float], list[float]]) -> Figure:
    couplings = sorted({coupling for _, coupling in series_rows})

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for series in SERIES:
        xs = [c for c in couplings if (series, c) in series_rows]
        if not xs:
            continue
        means, ses = zip(*(mean_se(series_rows[(series, c)]) for c in xs), strict=True)
        style = {'linestyle': '--', 'color': 'gray'} if series == FLOOR else {'marker': 'o'}
        ax.errorbar(xs, means, yerr=ses, label=CELL_DISPLAY.get(series, series), **style)

    ax.set_xscale('log', base=2)
    ax.set_xticks(couplings, [f'{c:g}' for c in couplings])
    ax.set_xlabel('coupling strength')
    ax.set_ylabel('JSD (nats)')
    ax.legend()
    return fig


def main() -> None:
    args = parse_args()
    apply_plot_style()

    fig = create_coupling_figure(coupling_rows(load_rows(Path(args.results))))
    path = str(Path(args.figure_dir) / 'jsd_vs_coupling.pdf')
    save_figure(fig, path)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
