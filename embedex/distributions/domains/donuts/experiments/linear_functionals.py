import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from embedex.distributions.domains.donuts.config import ExperimentConfig
from embedex.distributions.domains.donuts.donut import Donut, sample_batch
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.utils import save_figure


def run_linear_functionals(
    model: FlowEmbedding,
    config: ExperimentConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    """Test linear functionals on embedding space via vector arithmetic.

    Creates three distributions at (-1,-1), (-1,1), and (1,-1), computes their
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

    weights = jnp.ones(start.shape[1])
    zs_raw = jax.vmap(model.embed, in_axes=(0, None))(start, weights)
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

    key, gen_key = jax.random.split(key, 2)
    generate_keys = jax.random.split(gen_key, zs.shape[0])
    xs = jax.vmap(model.generate, in_axes=(0, None))(
        zs, config.linear_functionals_num_samples, key=generate_keys
    )

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    titles = [
        r'$z_0$ (-1,-1)',
        r'$z_1$ (-1,1)',
        r'$z_2$ (1,-1)',
        r'$z_1 - z_0 + z_2$',
        r'$z_0 + z_1$',
        r'$z_1 - z_0$',
        r'$z_2 - z_0$',
        r'$z_0 + z_2$',
    ]

    for i in range(8):
        seaborn.kdeplot(
            {'x': np.array(xs[i, :, 0]), 'y': np.array(xs[i, :, 1])},
            x='x',
            y='y',
            fill=True,
            alpha=0.3,
            ax=axes[i],
        )
        axes[i].set_title(titles[i])
        axes[i].set_xlim(-4.5, 4.5)
        axes[i].set_ylim(-4.5, 4.5)
        axes[i].grid(True)

    fig.tight_layout()
    save_figure(fig, os.path.join(save_path, 'linear_functionals.pdf'))
    plt.close(fig)
