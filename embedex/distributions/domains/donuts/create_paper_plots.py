import os

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray

from embedex.distributions.domains.donuts.config import VisualizationConfig
from embedex.distributions.domains.donuts.donut import PAPER_DONUTS, Donut, sample_batch
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.utils import save_figure


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
    ax.grid(True)

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
    ax.grid(True)

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
