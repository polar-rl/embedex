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


def run_interpolation(
    model: FlowEmbedding,
    config: ExperimentConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    """Run embedding space interpolation experiment.

    Generates two donut distributions, computes their embeddings, and visualizes
    samples from linearly interpolated points in the embedding space.
    """

    donuts = [
        Donut(mean=jnp.array([0.0, -1.0]), radius=0.7, width=0.25),
        Donut(mean=jnp.array([0.0, 1.0]), radius=0.7, width=0.25),
    ]

    key, sample_key = jax.random.split(key, 2)
    start = sample_batch(donuts, config.interpolation_num_samples, sample_key)

    num_steps = config.interpolation_num_steps
    weights = jnp.ones(start.shape[1])
    zs = jax.vmap(model.embed, in_axes=(0, None))(start, weights)
    zs = jnp.concatenate(
        (
            zs[0][0:1],
            jnp.concatenate(
                [
                    (num_steps - i) / (num_steps + 1) * zs[0][0:1]
                    + (i + 1) / (num_steps + 1) * zs[0][1:2]
                    for i in range(num_steps)
                ]
            ),
            zs[0][1:2],
        )
    )

    key, gen_key = jax.random.split(key, 2)
    generate_keys = jax.random.split(gen_key, zs.shape[0])
    xs = jax.vmap(model.generate, in_axes=(0, None))(
        zs, config.interpolation_num_samples, key=generate_keys
    )

    fig = plt.figure()
    fig.set_figwidth(16)

    total_plots = 2 + num_steps
    for i in range(total_plots):
        ax = plt.subplot(1, total_plots, i + 1)

        seaborn.kdeplot(
            {'x': np.array(xs[i, :, 0]), 'y': np.array(xs[i, :, 1])},
            x='x',
            y='y',
            fill=True,
            alpha=0.3,
        )

        ax.set_ylim(-2.5, 2.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.grid(True)

    save_figure(fig, os.path.join(save_path, 'interpolation.pdf'))
    plt.close(fig)
