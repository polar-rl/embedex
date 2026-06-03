import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from embedex.distributions.domains.donuts.config import DonutDistributionConfig, ExperimentConfig
from embedex.distributions.domains.donuts.donut import random_donuts, sample_batch
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.utils import save_figure


def run_embedding_visualization(
    model: FlowEmbedding,
    config: ExperimentConfig,
    distribution: DonutDistributionConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    key, donuts_key, sample_key = jax.random.split(key, 3)
    donuts = random_donuts(
        config.embedding_visualization_num_distributions,
        distribution.mean_std,
        distribution.radius_range,
        distribution.width_range,
        donuts_key,
    )

    samples = sample_batch(donuts, config.embedding_visualization_num_samples, sample_key)
    embeddings = []
    num = donuts.mean.shape[0]
    for i in range(num):
        z, _ = model.embed(samples[i], jnp.ones(samples[i].shape[0]))
        embeddings.append(z)

    embeddings_np = np.array(jnp.stack(embeddings))

    # Guard against collapsed embeddings (e.g. variational posterior collapse)
    if np.std(embeddings_np) < 1e-8:
        print('  Warning: embeddings have near-zero variance — skipping visualization')
        return None

    mean_x = np.array(donuts.mean[:, 0])
    mean_y = np.array(donuts.mean[:, 1])
    radii = np.array(donuts.radius)
    widths = np.array(donuts.width)

    properties = [
        ('mean_x', 'Mean X', mean_x, 'coolwarm'),
        ('mean_y', 'Mean Y', mean_y, 'coolwarm'),
        ('radius', 'Radius', radii, 'coolwarm'),
        ('width', 'Width', widths, 'coolwarm'),
    ]

    print('  Computing t-SNE...')
    tsne = TSNE(
        n_components=2,
        perplexity=min(config.embedding_visualization_perplexity, num - 1),
        random_state=config.embedding_visualization_random_state,
    )
    coords_tsne = tsne.fit_transform(embeddings_np)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for i, (_, name, values, cmap) in enumerate(properties):
        sc = axes[i].scatter(
            coords_tsne[:, 0],
            coords_tsne[:, 1],
            c=values,
            cmap=cmap,
            s=15,
            alpha=0.7,
        )
        axes[i].set_title(name)
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
        plt.colorbar(sc, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle('t-SNE of Donut Embeddings', fontsize=14)
    save_figure(fig, os.path.join(save_path, 'embedding_tsne.pdf'))

    print('  Computing PCA...')
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(embeddings_np)
    var_explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for i, (_, name, values, cmap) in enumerate(properties):
        sc = axes[i].scatter(
            coords_pca[:, 0],
            coords_pca[:, 1],
            c=values,
            cmap=cmap,
            s=15,
            alpha=0.7,
        )
        axes[i].set_title(name)
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel(f'PC1 ({var_explained[0]:.1%})')
        axes[i].set_ylabel(f'PC2 ({var_explained[1]:.1%})')
        plt.colorbar(sc, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle('PCA of Donut Embeddings', fontsize=14)
    save_figure(fig, os.path.join(save_path, 'embedding_pca.pdf'))

    num_components = min(4, embeddings_np.shape[1])
    pca_full = PCA(n_components=num_components)
    coords_full = pca_full.fit_transform(embeddings_np)
    var_full = pca_full.explained_variance_ratio_

    print(f'  PCA variance explained: {[f"{v:.1%}" for v in var_full]}')
    print(f'  Total: {sum(var_full):.1%}')

    num_pairs = num_components * (num_components - 1) // 2
    for slug, prop_name, prop_values, cmap in properties:
        fig, axes = plt.subplots(1, num_pairs, figsize=(5 * num_pairs, 4.5))
        if num_pairs == 1:
            axes = [axes]

        pair_idx = 0
        for pc_i in range(num_components):
            for pc_j in range(pc_i + 1, num_components):
                sc = axes[pair_idx].scatter(
                    coords_full[:, pc_i],
                    coords_full[:, pc_j],
                    c=prop_values,
                    cmap=cmap,
                    s=15,
                    alpha=0.7,
                )
                axes[pair_idx].set_xlabel(f'PC{pc_i + 1} ({var_full[pc_i]:.1%})')
                axes[pair_idx].set_ylabel(f'PC{pc_j + 1} ({var_full[pc_j]:.1%})')
                axes[pair_idx].set_aspect('equal')
                axes[pair_idx].grid(True, alpha=0.3)
                plt.colorbar(sc, ax=axes[pair_idx], fraction=0.046, pad=0.04)
                pair_idx += 1

        fig.suptitle(f'PCA Pairs — colored by {prop_name}', fontsize=14)
        save_figure(fig, os.path.join(save_path, f'embedding_pca_pairs_{slug}.pdf'))
