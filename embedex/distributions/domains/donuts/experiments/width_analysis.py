import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from embedex.distributions.domains.donuts.config import DonutDistributionConfig, ExperimentConfig
from embedex.distributions.domains.donuts.donut import Donut, random_donuts, sample_batch
from embedex.distributions.embedding import FlowEmbedding
from embedex.plotting import save_figure


def create_width_analysis_plot(
    generate_fn: Callable[[Donut, int, jax.Array], jax.Array],
    num_distributions: int,
    num_samples: int,
    distribution: DonutDistributionConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    """Scatter plot of true vs generated radial std (proxy for width).

    Args:
        generate_fn: Callable (donut, num_samples, key) -> generated samples array.
    """

    donut_key, sample_key, gen_key = jax.random.split(key, 3)
    donuts = random_donuts(
        num_distributions,
        distribution.mean_std,
        distribution.radius_range,
        distribution.width_range,
        donut_key,
    )

    gen_keys = jax.random.split(gen_key, num_distributions)

    true_samples = sample_batch(donuts, num_samples, sample_key)
    gen_samples = jax.vmap(generate_fn, in_axes=(0, None, 0))(
        donuts,
        num_samples,
        gen_keys,
    )

    def radial_std(samples):
        mean = jnp.mean(samples, axis=0)
        radii = jnp.linalg.norm(samples - mean, axis=1)
        return jnp.std(radii)

    true_stds = np.array(jax.vmap(radial_std)(true_samples))
    gen_stds = np.array(jax.vmap(radial_std)(gen_samples))

    fig, ax = plt.subplots()
    lim = max(true_stds.max(), gen_stds.max()) * 1.1
    ax.scatter(true_stds, gen_stds, s=15, alpha=0.7)
    ax.plot([0, lim], [0, lim], 'r--', label='ideal')
    ax.set_xlabel('True width')
    ax.set_ylabel('Generated width')
    ax.set_title('Width reconstruction')
    ax.legend()
    ax.grid(True)

    save_figure(fig, os.path.join(save_path, 'width_analysis.pdf'))


def run_width_analysis(
    model: FlowEmbedding,
    config: ExperimentConfig,
    distribution: DonutDistributionConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    def generate_fn(donut, num_samples, key):
        sample_key, gen_key = jax.random.split(key, 2)
        xs = donut.sample(num_samples, key=sample_key)
        embedding, _ = model.embed(xs)
        return model.generate(embedding, num_samples, key=gen_key)

    create_width_analysis_plot(
        generate_fn,
        config.width_analysis_num_distributions,
        config.width_analysis_num_samples,
        distribution,
        save_path,
        key,
    )
