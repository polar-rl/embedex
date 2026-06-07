import os

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from embedex.distributions.embedding import FlowEmbedding


def plot_loss_curve(
    losses: list[float],
    save_path: str,
    smoothing_window: int = 16,
    log_interval: int = 1,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    smoothed = np.convolve(losses, np.ones(smoothing_window), 'valid') / smoothing_window
    xs = np.arange(len(smoothed)) * log_interval

    ax.plot(xs, smoothed)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True)

    save_figure(fig, os.path.join(save_path, 'loss_curve.pdf'))


def plot_norm_curve(
    norms: list[float],
    save_path: str,
    smoothing_window: int = 16,
    log_interval: int = 1,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    smoothed = np.convolve(norms, np.ones(smoothing_window), 'valid') / smoothing_window
    xs = np.arange(len(smoothed)) * log_interval

    ax.plot(xs, smoothed)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Embedding Norm')
    ax.grid(True)

    save_figure(fig, os.path.join(save_path, 'embedding_norm_curve.pdf'))


def plot_model_density(
    model: FlowEmbedding,
    embedding: jax.Array,
    ax: plt.Axes,
    xlims: tuple[float, float],
    ylims: tuple[float, float],
    grid_size: int = 100,
    num_contour_levels: int = 10,
    fill: bool = True,
    cmap: str | Colormap = 'Blues',
) -> None:
    x = np.linspace(*xlims, grid_size)
    y = np.linspace(*ylims, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    log_probs, _, _ = model.log_prob(grid, embedding)
    density = np.exp(np.array(log_probs)).reshape(grid_size, grid_size)

    contour_fn = ax.contourf if fill else ax.contour
    contour_fn(xx, yy, density, levels=num_contour_levels, cmap=cmap)


def save_figure(fig: Figure, path: str, tight: bool = True, **kwargs) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', **kwargs)
    plt.close(fig)
