import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


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
    plt.close(fig)


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
    plt.close(fig)


def save_figure(fig: Figure, path: str, **kwargs) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig.savefig(path, bbox_inches='tight', **kwargs)
