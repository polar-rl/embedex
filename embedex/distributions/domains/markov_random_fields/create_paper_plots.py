import os

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from embedex.distributions.domains.markov_random_fields.config import (
    MRFDistributionConfig,
    VisualizationConfig,
)
from embedex.distributions.domains.markov_random_fields.evaluate import empirical_pmf
from embedex.distributions.domains.markov_random_fields.mrf import MRF
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.metrics import js_divergence
from embedex.distributions.utils import save_figure


def reconstruct(
    model: FlowEmbedding,
    mrf: MRF,
    num_embedding_samples: int,
    num_generated_samples: int,
    key: jax.Array,
) -> np.ndarray:
    """Embed samples from `mrf` and generate binary samples from the model."""

    sample_key, gen_key = jax.random.split(key, 2)

    samples = mrf.sample(num_embedding_samples, sample_key)
    embedding, _ = model.embed(samples, jnp.ones(samples.shape[0]))
    generated = model.generate(embedding, num_generated_samples, key=gen_key)

    return np.clip(np.round(np.array(generated)), 0, 1)


def draw_coupling_graph(ax: plt.Axes, coupling: np.ndarray) -> None:
    """Draw the field as a node-link graph.

    Nodes sit on a circle; each edge is colored and weighted by its coupling
    J_ij (red positive, blue negative, width by magnitude). Zero couplings are
    omitted, so this reads off the graph topology directly once it varies
    across the family.
    """

    n = coupling.shape[0]
    angles = 2.0 * np.pi * np.arange(n) / n
    xs, ys = np.cos(angles), np.sin(angles)

    jmax = max(float(np.abs(coupling).max()), 1e-8)
    threshold = 1e-3 * jmax
    cmap = plt.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=-jmax, vmax=jmax)

    for i in range(n):
        for j in range(i + 1, n):
            w = float(coupling[i, j])
            if abs(w) <= threshold:
                continue
            ax.plot(
                [xs[i], xs[j]],
                [ys[i], ys[j]],
                color=cmap(norm(w)),
                linewidth=0.5 + 4.0 * abs(w) / jmax,
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


def create_reconstruction_plot(
    model: FlowEmbedding,
    mrf: MRF,
    config: VisualizationConfig,
    key: jax.Array,
) -> Figure:
    """The Ising-style belief-reconstruction panel for a single field.

    True vs reconstructed pairwise marginals and their difference, the single-
    node marginals, the coupling matrix J (the field's graph), and a strip of
    generated samples.
    """

    n = mrf.num_spins
    true_pair = np.array(mrf.pairwise_marginals())
    true_single = np.array(mrf.single_marginals())
    coupling = np.array(mrf.J)

    gen = reconstruct(
        model, mrf, config.num_embedding_samples, config.num_generated_samples, key
    )
    recon_pair = (gen.T @ gen) / len(gen)
    recon_single = gen.mean(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    vmax = max(true_pair.max(), recon_pair.max(), 0.01)
    im0 = axes[0, 0].imshow(true_pair, cmap='Blues', vmin=0, vmax=vmax)
    axes[0, 0].set_title('True P(xi=1, xj=1)')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(recon_pair, cmap='Blues', vmin=0, vmax=vmax)
    axes[0, 1].set_title('Reconstructed')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    diff = true_pair - recon_pair
    dmax = max(abs(diff.min()), abs(diff.max()), 0.01)
    im2 = axes[0, 2].imshow(diff, cmap='RdBu', vmin=-dmax, vmax=dmax)
    axes[0, 2].set_title('Difference')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    x = np.arange(n)
    axes[1, 0].bar(x - 0.15, true_single, width=0.3, label='True', alpha=0.8)
    axes[1, 0].bar(x + 0.15, recon_single, width=0.3, label='Recon', alpha=0.8)
    axes[1, 0].set_title('P(xi=1)')
    axes[1, 0].set_xlabel('Node')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_ylim(0, 1)

    draw_coupling_graph(axes[1, 1], coupling)

    num_show = min(config.sample_strip_num, len(gen))
    axes[1, 2].imshow(gen[:num_show], cmap='Greys', aspect='auto', interpolation='nearest')
    axes[1, 2].set_title(f'Generated samples ({num_show})')
    axes[1, 2].set_xlabel('Node')
    axes[1, 2].set_ylabel('Sample')

    fig.suptitle(f'MRF Belief Reconstruction (N={n})', fontsize=14)
    fig.tight_layout()

    return fig


def create_reconstruction_grid(
    model: FlowEmbedding,
    distribution: MRFDistributionConfig,
    config: VisualizationConfig,
    key: jax.Array,
) -> Figure:
    """Grid of true vs reconstructed pairwise marginals across several fields."""

    num_plots = config.grid_num_distributions
    ncols = 4
    nrows = (num_plots + ncols // 2 - 1) // (ncols // 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    keys = jax.random.split(key, num_plots)

    for idx in range(num_plots):
        mrf_key, recon_key = jax.random.split(keys[idx], 2)
        mrf = MRF.random(
            distribution.num_spins,
            distribution.coupling_scale,
            distribution.bias_scale,
            mrf_key,
        )
        true_pair = np.array(mrf.pairwise_marginals())

        gen = reconstruct(
            model, mrf, config.num_embedding_samples, config.num_generated_samples, recon_key
        )
        recon_pair = (gen.T @ gen) / len(gen)
        jsd = float(js_divergence(mrf.pmf(), empirical_pmf(gen, mrf.num_spins)))

        row = idx // (ncols // 2)
        col_base = (idx % (ncols // 2)) * 2

        vmax = max(true_pair.max(), recon_pair.max(), 0.01)
        axes[row, col_base].imshow(true_pair, cmap='Blues', vmin=0, vmax=vmax)
        axes[row, col_base].set_title('True', fontsize=9)
        axes[row, col_base + 1].imshow(recon_pair, cmap='Blues', vmin=0, vmax=vmax)
        axes[row, col_base + 1].set_title(f'Recon (JSD={jsd:.3f})', fontsize=9)

        for ax in (axes[row, col_base], axes[row, col_base + 1]):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('MRF Pairwise Marginals: True vs Reconstructed', fontsize=14)
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

    print('Generating belief reconstruction plot...')
    mrf_key, recon_key = jax.random.split(single_key, 2)
    mrf = MRF.random(
        distribution.num_spins,
        distribution.coupling_scale,
        distribution.bias_scale,
        mrf_key,
    )
    fig = create_reconstruction_plot(model, mrf, config, recon_key)
    save_figure(fig, os.path.join(save_path, 'mrf_reconstruction.pdf'))
    plt.close(fig)

    print('Generating reconstruction grid...')
    fig = create_reconstruction_grid(model, distribution, config, grid_key)
    save_figure(fig, os.path.join(save_path, 'mrf_reconstruction_grid.pdf'))
    plt.close(fig)
