import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from embedex.distributions.domains.config import DonutsEvaluationConfig
from embedex.distributions.domains.donuts.donut import Donut, sample_batch
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.utils import plot_model_density
from embedex.plotting import save_figure


def run_linear_functionals(
    model: FlowEmbedding,
    config: DonutsEvaluationConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    """Test linear functionals on embedding space via vector arithmetic.

    Creates three distributions at (-1, -1), (-1, 1), and (1, -1), computes their
    embeddings z0, z1, z2, then generates samples from various pairwise
    additions and subtractions to test whether embedding arithmetic produces
    meaningful distributions.
    """

    donuts = [
        Donut(mean=jnp.array([-1.0, -1.0]), radius=0.7, width=0.25),
        Donut(mean=jnp.array([-1.0, 1.0]), radius=0.7, width=0.25),
        Donut(mean=jnp.array([1.0, -1.0]), radius=0.7, width=0.25),
    ]

    key, sample_key = jax.random.split(key, 2)
    start = sample_batch(donuts, config.linear_functionals_num_samples, sample_key)

    zs_raw = jax.vmap(model.embed, in_axes=(0, None))(start, jnp.ones(start.shape[1]))
    z0, z1, z2 = zs_raw[0][0], zs_raw[0][1], zs_raw[0][2]

    # Original 3 embeddings + 5 pairwise operations
    zs = jnp.stack(
        [
            z0,  # base: (-1, -1)
            z1,  # base: (-1, 1)
            z2,  # base: (1, -1)
            z1 - z0 + z2,  # analogy: should give (1, 1)
            z0 + z1,  # addition
            z1 - z0,  # difference
            z2 - z0,  # difference
            z0 + z2,  # addition
        ]
    )

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    titles = [
        r'$z_0$ (-1, -1)',
        r'$z_1$ (-1, 1)',
        r'$z_2$ (1, -1)',
        r'$z_1 - z_0 + z_2$',
        r'$z_0 + z_1$',
        r'$z_1 - z_0$',
        r'$z_2 - z_0$',
        r'$z_0 + z_2$',
    ]

    for i in range(8):
        plot_model_density(model, zs[i], axes[i], (-4.5, 4.5), (-4.5, 4.5))
        axes[i].set_title(titles[i])
        axes[i].set_xlim(-4.5, 4.5)
        axes[i].set_ylim(-4.5, 4.5)
        axes[i].grid(True)

    save_figure(fig, os.path.join(save_path, 'linear_functionals.pdf'))
