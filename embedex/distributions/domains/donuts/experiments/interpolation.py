import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from embedex.distributions.domains.donuts.config import ExperimentConfig
from embedex.distributions.domains.donuts.donut import Donut, sample_batch
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.utils import plot_model_density, save_figure


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
    zs = jax.vmap(model.embed, in_axes=(0, None))(start, jnp.ones(start.shape[1]))
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

    fig = plt.figure()
    fig.set_figwidth(16)

    total_plots = 2 + num_steps
    for i in range(total_plots):
        ax = plt.subplot(1, total_plots, i + 1)

        plot_model_density(model, zs[i], ax, (-1.5, 1.5), (-2.5, 2.5))
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.grid(True)

    save_figure(fig, os.path.join(save_path, 'interpolation.pdf'))
