import math
import os

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from embedex.distributions.domains.config import DonutsEvaluationConfig, DonutsVisualizationConfig
from embedex.distributions.domains.donuts.donut import Donut, sample_batch
from embedex.distributions.domains.donuts.plots import (
    FIGURE_WIDTH,
    create_gradient_cmap,
    density_on_grid,
    style_panel,
    view_box,
)
from embedex.distributions.embedding import FlowEmbedding
from embedex.plotting import apply_latex_style, save_figure

# Interpolation endpoints: differ in centre, radius and width, so a straight line in
# embedding space has to move all three at once. Both stay inside the training ranges.
INTERPOLATION_DONUTS = [
    Donut(mean=jnp.array([-1.2, -0.3]), radius=jnp.float32(0.8), width=jnp.float32(0.65)),
    Donut(mean=jnp.array([1.0, 0.4]), radius=jnp.float32(1.8), width=jnp.float32(0.12)),
]


@apply_latex_style
def build_interpolation_figure(
    model: FlowEmbedding,
    config: DonutsVisualizationConfig,
    num_steps: int,
    num_samples: int,
    key: jax.Array,
) -> Figure:
    num_cols = 4

    samples = np.asarray(sample_batch(INTERPOLATION_DONUTS, num_samples, key))
    endpoints = [model.embed(jnp.asarray(s))[0] for s in samples]

    xlims, ylims = view_box(samples.reshape(-1, 2))
    gx = np.linspace(*xlims, config.grid_size)
    gy = np.linspace(*ylims, config.grid_size)
    xx, yy = np.meshgrid(gx, gy)

    c0, c1 = (np.array(mpl.colors.to_rgb(c)) for c in ('C0', 'C2'))

    alphas = np.linspace(0.0, 1.0, num_steps + 2)
    num_rows = math.ceil(len(alphas) / num_cols)
    panel_width = FIGURE_WIDTH / num_cols
    aspect = (ylims[1] - ylims[0]) / (xlims[1] - xlims[0])

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(FIGURE_WIDTH, num_rows * panel_width * aspect + 0.9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for i, alpha in enumerate(alphas):
        ax = axes[i]
        color = tuple((1.0 - alpha) * c0 + alpha * c1)
        z = (1.0 - alpha) * endpoints[0] + alpha * endpoints[1]

        density = density_on_grid(model, z, xx, yy, config.density_threshold)
        ax.contourf(
            xx,
            yy,
            density,
            levels=max(config.num_contour_levels, 20),
            cmap=create_gradient_cmap(color, max_alpha=0.8),
            norm='log',
        )

        style_panel(
            ax,
            xlims,
            ylims,
            rf'$\alpha = {alpha:.2f}$',
            xlabel=(i >= len(alphas) - num_cols),
            ylabel=(i % num_cols == 0),
            label_size=9.0,
            tick_size=7.5,
        )

    for ax in axes[len(alphas) :]:
        ax.set_visible(False)

    return fig


def run_interpolation(
    model: FlowEmbedding,
    config: DonutsEvaluationConfig,
    visualization: DonutsVisualizationConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    """Embed two donuts, then decode a straight line between their embeddings."""

    fig = build_interpolation_figure(
        model,
        visualization,
        config.interpolation_num_steps,
        config.interpolation_num_samples,
        key,
    )
    save_figure(fig, os.path.join(save_path, 'interpolation.pdf'))
