import itertools
import os

import jax
import jax.numpy as jnp
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

# Parallelogram test: each property has a low and a high setting, applied on top of
# PARALLELOGRAM_BASE. A donut is named by which setting each of two properties takes.
PARALLELOGRAM_BASE = {'mean_x': 0.0, 'mean_y': 0.0, 'radius': 1.0, 'width': 0.25}
PARALLELOGRAM_PROPERTIES = {
    'Pos X': {'low': {'mean_offset_x': -1.2}, 'high': {'mean_offset_x': 1.2}},
    'Pos Y': {'low': {'mean_offset_y': -1.2}, 'high': {'mean_offset_y': 1.2}},
    'Radius': {'low': {'radius': 0.7}, 'high': {'radius': 1.8}},
    'Width': {'low': {'width': 0.1}, 'high': {'width': 0.5}},
}


def parallelogram_donut(overrides: dict[str, float]) -> Donut:
    """A PARALLELOGRAM_BASE donut with one or two property settings applied."""

    return Donut(
        mean=jnp.array(
            [
                PARALLELOGRAM_BASE['mean_x'] + overrides.get('mean_offset_x', 0.0),
                PARALLELOGRAM_BASE['mean_y'] + overrides.get('mean_offset_y', 0.0),
            ]
        ),
        radius=jnp.float32(overrides.get('radius', PARALLELOGRAM_BASE['radius'])),
        width=jnp.float32(overrides.get('width', PARALLELOGRAM_BASE['width'])),
    )


LOW, HIGH = r'\downarrow', r'\uparrow'


def _parallelogram_titles(prop_names: list[str], letters: list[str], formula: str) -> list[str]:
    if len(prop_names) == 2:
        p1, p2 = prop_names
        settings = [
            rf'{p1}${LOW}$ {p2}${LOW}$',
            rf'{p1}${HIGH}$ {p2}${LOW}$',
            rf'{p1}${LOW}$ {p2}${HIGH}$',
            rf'{p1}${HIGH}$ {p2}${HIGH}$',
        ]
    else:
        settings = [rf'all ${LOW}$', *(rf'{p}${HIGH}$' for p in prop_names), rf'all ${HIGH}$']

    titles = [rf'${x}$: {s}' for x, s in zip(letters, settings, strict=True)]
    titles.append(rf'$\hat{{{letters[-1]}}} = {formula}$')

    return titles


@apply_latex_style
def build_parallelogram_figure(
    model: FlowEmbedding,
    config: DonutsVisualizationConfig,
    prop_names: list[str],
    num_samples: int,
    key: jax.Array,
) -> tuple[Figure, dict]:
    k = len(prop_names)
    props = [PARALLELOGRAM_PROPERTIES[name] for name in prop_names]

    all_low = {key_: val for p in props for key_, val in p['low'].items()}
    all_high = {key_: val for p in props for key_, val in p['high'].items()}
    singles = [{**all_low, **p['high']} for p in props]

    samples = np.asarray(
        sample_batch(
            [parallelogram_donut(o) for o in [all_low, *singles, all_high]], num_samples, key
        )
    )

    embeddings = [model.embed(jnp.asarray(s))[0] for s in samples]
    z_base, z_singles, z_actual = embeddings[0], embeddings[1:-1], embeddings[-1]
    z_pred = sum(z_singles) - (k - 1) * z_base

    metrics = {
        'l2_true': float(jnp.linalg.norm(z_actual)),
        'l2_error': float(jnp.linalg.norm(z_pred - z_actual)),
        'cos_sim': float(
            jnp.dot(z_pred, z_actual) / (jnp.linalg.norm(z_pred) * jnp.linalg.norm(z_actual) + 1e-8)
        ),
    }

    letters = [chr(ord('A') + i) for i in range(k + 2)]
    base_term = f'{k - 1}{letters[0]}' if k > 2 else letters[0]
    formula = ' + '.join(letters[1:-1]) + f' - {base_term}'
    titles = _parallelogram_titles(prop_names, letters, formula)
    colors = ['C0'] * (k + 2) + ['C1']  # blue: really sampled; orange: the algebraic prediction

    xlims, ylims = view_box(samples.reshape(-1, 2))
    gx = np.linspace(*xlims, config.grid_size)
    gy = np.linspace(*ylims, config.grid_size)
    xx, yy = np.meshgrid(gx, gy)

    panels = [*embeddings, z_pred]
    panel_width = FIGURE_WIDTH / len(panels)
    aspect = (ylims[1] - ylims[0]) / (xlims[1] - xlims[0])
    fig, axes = plt.subplots(
        1, len(panels), figsize=(FIGURE_WIDTH, panel_width * aspect + 1.1), sharey=True
    )

    title_size, label_size, tick_size = (6.6, 7.5, 6.0) if k == 2 else (6.0, 7.0, 5.5)
    for i, (ax, z, title, color) in enumerate(zip(axes, panels, titles, colors, strict=True)):
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
            title,
            ylabel=(i == 0),
            label_size=label_size,
            tick_size=tick_size,
        )
        ax.title.set_fontsize(title_size)

    return fig, metrics


def run_parallelogram(
    model: FlowEmbedding,
    config: DonutsEvaluationConfig,
    visualization: DonutsVisualizationConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    """Parallelogram test over every property pair and every triple.

    Predicts the all-high corner as `sum(singles) - (k - 1) * base`
    and reports how far that lands from the real embedding.
    """

    subsets = [
        *itertools.combinations(PARALLELOGRAM_PROPERTIES, 2),
        *itertools.combinations(PARALLELOGRAM_PROPERTIES, 3),
    ]
    print(f'  Testing {len(subsets)} property subsets...')

    for names in subsets:
        key, subset_key = jax.random.split(key, 2)
        fig, metrics = build_parallelogram_figure(
            model,
            visualization,
            list(names),
            config.parallelogram_num_samples,
            subset_key,
        )
        rel_error = metrics['l2_error'] / metrics['l2_true']
        print(
            f'  {" x ".join(names)}:  L2 error = {metrics["l2_error"]:.4f}  '
            f'(rel_error = {rel_error * 100:.1f}%)  cos_sim = {metrics["cos_sim"]:.4f}'
        )

        prefix = 'parallelogram' if len(names) == 2 else 'parallelogram_triple'
        slug = '_'.join(names).replace(' ', '_').lower()
        save_figure(fig, os.path.join(save_path, f'{prefix}_{slug}.pdf'))
