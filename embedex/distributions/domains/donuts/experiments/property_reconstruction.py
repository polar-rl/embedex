import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from embedex.distributions.domains.donuts.config import DonutDistributionConfig, ExperimentConfig
from embedex.distributions.domains.donuts.donut import Donut, donut_moments
from embedex.distributions.embedding import FlowEmbedding
from embedex.plotting import save_figure

VIEW = (-4.5, 4.5)


def _measure(samples: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Recover (center, radius, width) from generated samples (cf. evaluate_all)."""
    center, radius, width = donut_moments(jnp.asarray(samples))
    return np.asarray(center), float(radius), float(width)


def draw_recon_panel(
    ax,
    model: FlowEmbedding,
    build,
    value: float,
    true_of,
    prop: str,
    n_embed: int,
    n_gen: int,
    key: jax.Array,
) -> float:
    sample_key, gen_key = jax.random.split(key, 2)
    donut = build(value)
    true_samples = np.asarray(donut.sample(n_embed, sample_key))

    z, _ = model.embed(jnp.asarray(true_samples), jnp.ones(n_embed))
    generated = np.asarray(model.generate(z, n_gen, key=gen_key))

    center, radius, width = _measure(generated)
    err = abs(true_of(center, radius, width) - float(value))

    ax.scatter(true_samples[:, 0], true_samples[:, 1], s=4, color='0.7', marker='.')
    ax.scatter(generated[:, 0], generated[:, 1], s=4, color='C0', alpha=0.5, marker='.')
    ax.set_xlim(*VIEW)
    ax.set_ylim(*VIEW)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{prop}={value:.2f}\nerr={err:.3f}')
    return err


def _sweep(distribution: DonutDistributionConfig, num_steps: int) -> dict:
    r_lo, r_hi = distribution.radius_range
    w_lo, w_hi = distribution.width_range
    return {
        'radius': (
            np.linspace(r_lo, r_hi, num_steps),
            lambda v: Donut(mean=jnp.array([0.0, 0.0]), radius=float(v), width=0.25),
            lambda c, r, w: r,
        ),
        'width': (
            np.linspace(w_lo, w_hi, num_steps),
            lambda v: Donut(mean=jnp.array([0.0, 0.0]), radius=1.2, width=float(v)),
            lambda c, r, w: w,
        ),
        'pos_x': (
            np.linspace(-1.5, 1.5, num_steps),
            lambda v: Donut(mean=jnp.array([float(v), 0.0]), radius=1.0, width=0.25),
            lambda c, r, w: c[0],
        ),
        'pos_y': (
            np.linspace(-1.5, 1.5, num_steps),
            lambda v: Donut(mean=jnp.array([0.0, float(v)]), radius=1.0, width=0.25),
            lambda c, r, w: c[1],
        ),
    }


def run_property_reconstruction(
    model: FlowEmbedding,
    config: ExperimentConfig,
    distribution: DonutDistributionConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    n_embed = config.property_reconstruction_num_samples
    n_gen = config.property_reconstruction_num_generated
    steps = config.property_reconstruction_num_steps
    sweeps = _sweep(distribution, steps)

    fig, axes = plt.subplots(
        len(sweeps), steps, figsize=(2.4 * steps, 2.4 * len(sweeps)), squeeze=False
    )

    for row, (prop, (values, build, true_of)) in enumerate(sweeps.items()):
        for col, value in enumerate(values):
            key, panel_key = jax.random.split(key, 2)
            ax = axes[row][col]
            draw_recon_panel(ax, model, build, value, true_of, prop, n_embed, n_gen, panel_key)
            if col == 0:
                ax.set_ylabel(prop, fontsize=11)

    fig.suptitle('Property reconstruction (grey: true, blue: generated; err = |decoded - true|)')
    fig.tight_layout()
    save_figure(fig, os.path.join(save_path, 'property_reconstruction.pdf'))
