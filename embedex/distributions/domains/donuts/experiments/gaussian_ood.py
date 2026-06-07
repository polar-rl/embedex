import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.figure import Figure

from embedex.distributions.domains.donuts.config import ExperimentConfig
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.utils import plot_model_density, save_figure


def create_ood_comparison_plot(
    model: FlowEmbedding,
    distributions: list[dict],
    num_samples: int,
    title: str,
    key: jax.Array,
) -> tuple[Figure, plt.Axes]:
    num_dists = len(distributions)
    fig, axes = plt.subplots(2, num_dists, figsize=(4 * num_dists, 8))

    for i, dist in enumerate(distributions):
        key, gen_key = jax.random.split(key, 2)
        samples = dist['samples']

        # Embed and generate
        embedding, _ = model.embed(samples, jnp.ones(samples.shape[0]))
        generated = model.generate(embedding, num_samples, key=gen_key)

        # Plot original samples (top row)
        seaborn.kdeplot(
            {'x': np.array(samples[:, 0]), 'y': np.array(samples[:, 1])},
            x='x',
            y='y',
            fill=True,
            alpha=0.3,
            ax=axes[0, i],
        )
        axes[0, i].scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3, color='blue')
        axes[0, i].set_title(dist['name'])
        axes[0, i].set_xlim(-3, 3)
        axes[0, i].set_ylim(-3, 3)
        axes[0, i].grid(True)

        # Plot generated samples (bottom row)
        plot_model_density(model, embedding, axes[1, i], (-3, 3), (-3, 3), cmap='Oranges')
        axes[1, i].scatter(generated[:, 0], generated[:, 1], s=5, alpha=0.3, color='orange')
        axes[1, i].set_title('Generated')
        axes[1, i].set_xlim(-3, 3)
        axes[1, i].set_ylim(-3, 3)
        axes[1, i].grid(True)

    axes[0, 0].set_ylabel('Input')
    axes[1, 0].set_ylabel('Model Output')

    fig.suptitle(title, fontsize=21)

    return fig, axes


def run_gaussian_ood(
    model: FlowEmbedding,
    config: ExperimentConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    # --- Part 1: Single Gaussians ---
    gaussians_spec = [
        {'mean': jnp.array([0.0, 0.0]), 'cov': jnp.array([[0.3, 0.0], [0.0, 0.3]])},
        {'mean': jnp.array([1.5, 0.0]), 'cov': jnp.array([[0.1, 0.0], [0.0, 0.5]])},
        {'mean': jnp.array([-1.0, 1.0]), 'cov': jnp.array([[0.4, 0.2], [0.2, 0.4]])},
        {'mean': jnp.array([0.0, -1.5]), 'cov': jnp.array([[0.6, -0.3], [-0.3, 0.2]])},
    ]

    gaussians = []
    for i, g in enumerate(gaussians_spec):
        key, sample_key = jax.random.split(key, 2)
        samples = jax.random.multivariate_normal(
            sample_key, g['mean'], g['cov'], (config.gaussian_ood_num_samples,)
        )
        gaussians.append({'samples': samples, 'name': f'Gaussian {i + 1}'})

    key, single_key = jax.random.split(key, 2)
    fig, _ = create_ood_comparison_plot(
        model,
        gaussians,
        config.gaussian_ood_num_samples,
        'Out-of-Distribution: Gaussians',
        single_key,
    )

    save_figure(fig, os.path.join(save_path, 'gaussian_ood_single.pdf'))

    mixture_params = [
        [
            {'mean': jnp.array([-1.2, 0.8]), 'cov': jnp.array([[0.15, 0.0], [0.0, 0.15]])},
            {'mean': jnp.array([1.0, 0.5]), 'cov': jnp.array([[0.15, 0.0], [0.0, 0.15]])},
        ],
        [
            {'mean': jnp.array([0.8, -1.3]), 'cov': jnp.array([[0.15, 0.0], [0.0, 0.15]])},
            {'mean': jnp.array([0.5, 1.2]), 'cov': jnp.array([[0.15, 0.0], [0.0, 0.15]])},
        ],
        [
            {'mean': jnp.array([-1.8, -0.5]), 'cov': jnp.array([[0.12, 0.0], [0.0, 0.12]])},
            {'mean': jnp.array([-0.3, -1.5]), 'cov': jnp.array([[0.12, 0.0], [0.0, 0.12]])},
        ],
        [
            {'mean': jnp.array([0.2, 1.0]), 'cov': jnp.array([[0.1, 0.0], [0.0, 0.1]])},
            {'mean': jnp.array([1.5, 0.3]), 'cov': jnp.array([[0.25, 0.0], [0.0, 0.25]])},
        ],
    ]

    mixture_names = ['Horizontal', 'Vertical', 'Diagonal', 'Asymmetric']
    mixtures = []
    for name, mix in zip(mixture_names, mixture_params, strict=True):
        key, mix0_key, mix1_key, shuffle_key = jax.random.split(key, 4)
        num_per = config.gaussian_ood_num_samples // 2
        samples1 = jax.random.multivariate_normal(
            mix0_key, mix[0]['mean'], mix[0]['cov'], (num_per,)
        )
        samples2 = jax.random.multivariate_normal(
            mix1_key, mix[1]['mean'], mix[1]['cov'], (num_per,)
        )
        samples = jax.random.permutation(shuffle_key, jnp.concatenate([samples1, samples2]))
        mixtures.append({'samples': samples, 'name': name})

    key, mixture_key = jax.random.split(key, 2)
    fig, _ = create_ood_comparison_plot(
        model,
        mixtures,
        config.gaussian_ood_num_samples,
        'Out-of-Distribution: Gaussian Mixtures',
        mixture_key,
    )

    save_figure(fig, os.path.join(save_path, 'gaussian_ood_mixtures.pdf'))
