import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from embedex.distributions.domains.config import DonutsEvaluationConfig
from embedex.distributions.domains.donuts.donut import Donut, sample_batch
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.utils import plot_model_density
from embedex.plotting import save_figure

VIEW = (-4.5, 4.5)

PROPERTY_LABELS = {
    'radius': 'Radius',
    'width': 'Width',
    'pos_x': 'Position X',
    'pos_y': 'Position Y',
}


def _radius_pair(key: jax.Array) -> tuple[Donut, Donut]:
    mean = jax.random.normal(key, (2,)) * 0.5
    return Donut(mean=mean, radius=0.8, width=0.25), Donut(mean=mean, radius=1.6, width=0.25)


def _width_pair(key: jax.Array) -> tuple[Donut, Donut]:
    mean = jax.random.normal(key, (2,)) * 0.5
    return Donut(mean=mean, radius=1.2, width=0.1), Donut(mean=mean, radius=1.2, width=0.5)


def _pos_x_pair(key: jax.Array) -> tuple[Donut, Donut]:
    y = jax.random.normal(key, ()) * 0.5
    return (
        Donut(mean=jnp.array([-1.0, y]), radius=1.0, width=0.25),
        Donut(mean=jnp.array([1.0, y]), radius=1.0, width=0.25),
    )


def _pos_y_pair(key: jax.Array) -> tuple[Donut, Donut]:
    x = jax.random.normal(key, ()) * 0.5
    return (
        Donut(mean=jnp.array([x, -1.0]), radius=1.0, width=0.25),
        Donut(mean=jnp.array([x, 1.0]), radius=1.0, width=0.25),
    )


PAIR_BUILDERS = {
    'radius': _radius_pair,
    'width': _width_pair,
    'pos_x': _pos_x_pair,
    'pos_y': _pos_y_pair,
}


def _compute_direction(
    model: FlowEmbedding,
    pair_builder,
    num_pairs: int,
    num_samples: int,
    key: jax.Array,
) -> jax.Array:
    directions = []
    for _ in range(num_pairs):
        key, pair_key, low_key, high_key = jax.random.split(key, 4)
        low_donut, high_donut = pair_builder(pair_key)

        low_samples = sample_batch([low_donut], num_samples, low_key)[0]
        high_samples = sample_batch([high_donut], num_samples, high_key)[0]

        z_low, _ = model.embed(low_samples)
        z_high, _ = model.embed(high_samples)

        directions.append(z_high - z_low)

    return jnp.mean(jnp.stack(directions), axis=0)


def compute_property_directions(
    model: FlowEmbedding,
    config: DonutsEvaluationConfig,
    key: jax.Array,
    props: list[str] | None = None,
) -> dict[str, jax.Array]:
    """Average embedding-difference direction for each requested property."""
    props = props or list(PAIR_BUILDERS)
    directions = {}
    for prop in props:
        key, dir_key = jax.random.split(key, 2)
        directions[prop] = _compute_direction(
            model,
            PAIR_BUILDERS[prop],
            config.property_directions_num_pairs,
            config.property_directions_num_samples,
            dir_key,
        )
    return directions


def base_embedding(
    model: FlowEmbedding, config: DonutsEvaluationConfig, key: jax.Array
) -> jax.Array:
    """Embedding of the canonical test donut that directions are added to."""
    base_donut = Donut(mean=jnp.array([0.0, 0.0]), radius=1.0, width=0.25)
    samples = sample_batch([base_donut], config.property_directions_num_samples, key)[0]
    z, _ = model.embed(samples)
    return z


def draw_direction_panel(
    ax,
    model: FlowEmbedding,
    z_base: jax.Array,
    direction: jax.Array,
    alpha: float,
    num_samples: int,
    key: jax.Array,
) -> None:
    """Decode the test embedding perturbed by ``alpha`` along ``direction``."""
    generated = np.asarray(model.generate(z_base + alpha * direction, num_samples, key=key))
    ax.scatter(generated[:, 0], generated[:, 1], s=5, alpha=0.5)
    ax.set_xlim(*VIEW)
    ax.set_ylim(*VIEW)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def run_property_directions(
    model: FlowEmbedding,
    config: DonutsEvaluationConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    key, dir_key, base_key = jax.random.split(key, 3)
    directions = compute_property_directions(model, config, dir_key)
    z_base = base_embedding(model, config, base_key)

    steps = config.property_directions_num_alpha_steps
    alphas = jnp.linspace(-1.5, 1.5, steps)

    for slug, direction in directions.items():
        fig, axes = plt.subplots(2, steps, figsize=(3 * steps, 6))

        for i, alpha in enumerate(alphas):
            key, gen_key = jax.random.split(key, 2)
            draw_direction_panel(
                axes[0, i],
                model,
                z_base,
                direction,
                float(alpha),
                config.property_directions_num_samples,
                gen_key,
            )
            axes[0, i].set_title(rf'$\alpha = {float(alpha):.1f}$')

            plot_model_density(model, z_base + alpha * direction, axes[1, i], VIEW, VIEW)
            axes[1, i].set_xlim(*VIEW)
            axes[1, i].set_ylim(*VIEW)
            axes[1, i].set_aspect('equal')
            axes[1, i].grid(True)

        axes[0, 0].set_ylabel('Samples')
        axes[1, 0].set_ylabel('Density')

        fig.suptitle(f'Property Direction: {PROPERTY_LABELS[slug]}', fontsize=21)
        save_figure(fig, os.path.join(save_path, f'property_direction_{slug}.pdf'))
