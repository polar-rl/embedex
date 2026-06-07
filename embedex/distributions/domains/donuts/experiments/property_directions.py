import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from embedex.distributions.domains.donuts.config import ExperimentConfig
from embedex.distributions.domains.donuts.donut import Donut, sample_batch
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.utils import plot_model_density, save_figure


def _compute_direction(
    model: FlowEmbedding,
    donut_pairs: list[tuple[Donut, Donut]],
    num_samples: int,
    key: jax.Array,
) -> jax.Array:
    directions = []

    for low_donut, high_donut in donut_pairs:
        key, low_key, high_key = jax.random.split(key, 3)

        low_samples = sample_batch([low_donut], num_samples, low_key)[0]
        high_samples = sample_batch([high_donut], num_samples, high_key)[0]

        z_low, _ = model.embed(low_samples, jnp.ones(low_samples.shape[0]))
        z_high, _ = model.embed(high_samples, jnp.ones(high_samples.shape[0]))

        directions.append(z_high - z_low)

    return jnp.mean(jnp.stack(directions), axis=0)


def run_property_directions(
    model: FlowEmbedding,
    config: ExperimentConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    radius_pairs = []
    for _ in range(config.property_directions_num_pairs):
        key, mean_key = jax.random.split(key, 2)
        mean = jax.random.normal(mean_key, (2,)) * 0.5
        width = 0.25
        low_donut = Donut(mean=mean, radius=0.8, width=width)
        high_donut = Donut(mean=mean, radius=1.6, width=width)
        radius_pairs.append((low_donut, high_donut))

    width_pairs = []
    for _ in range(config.property_directions_num_pairs):
        key, mean_key = jax.random.split(key, 2)
        mean = jax.random.normal(mean_key, (2,)) * 0.5
        radius = 1.2
        low_donut = Donut(mean=mean, radius=radius, width=0.1)
        high_donut = Donut(mean=mean, radius=radius, width=0.5)
        width_pairs.append((low_donut, high_donut))

    pos_x_pairs = []
    for _ in range(config.property_directions_num_pairs):
        key, y_key = jax.random.split(key, 2)
        y = jax.random.normal(y_key, ()) * 0.5
        radius, width = 1.0, 0.25
        low_donut = Donut(mean=jnp.array([-1.0, y]), radius=radius, width=width)
        high_donut = Donut(mean=jnp.array([1.0, y]), radius=radius, width=width)
        pos_x_pairs.append((low_donut, high_donut))

    pos_y_pairs = []
    for _ in range(config.property_directions_num_pairs):
        key, x_key = jax.random.split(key, 2)
        x = jax.random.normal(x_key, ()) * 0.5
        radius, width = 1.0, 0.25
        low_donut = Donut(mean=jnp.array([x, -1.0]), radius=radius, width=width)
        high_donut = Donut(mean=jnp.array([x, 1.0]), radius=radius, width=width)
        pos_y_pairs.append((low_donut, high_donut))

    key, radius_key, width_key, pos_x_key, pos_y_key = jax.random.split(key, 5)
    d_radius = _compute_direction(
        model, radius_pairs, config.property_directions_num_samples, radius_key
    )
    d_width = _compute_direction(
        model, width_pairs, config.property_directions_num_samples, width_key
    )
    d_pos_x = _compute_direction(
        model, pos_x_pairs, config.property_directions_num_samples, pos_x_key
    )
    d_pos_y = _compute_direction(
        model, pos_y_pairs, config.property_directions_num_samples, pos_y_key
    )

    directions = [
        ('radius', 'Radius', d_radius),
        ('width', 'Width', d_width),
        ('pos_x', 'Position X', d_pos_x),
        ('pos_y', 'Position Y', d_pos_y),
    ]

    base_donut = Donut(mean=jnp.array([0.0, 0.0]), radius=1.0, width=0.25)
    key, base_key = jax.random.split(key, 2)
    base_samples = sample_batch([base_donut], config.property_directions_num_samples, base_key)[0]
    z_base, _ = model.embed(base_samples, jnp.ones(base_samples.shape[0]))

    alphas = jnp.linspace(-1.5, 1.5, config.property_directions_num_alpha_steps)

    for slug, dir_name, direction in directions:
        fig, axes = plt.subplots(
            2,
            config.property_directions_num_alpha_steps,
            figsize=(3 * config.property_directions_num_alpha_steps, 6),
        )

        for i, alpha in enumerate(alphas):
            key, gen_key = jax.random.split(key, 2)

            z_perturbed = z_base + alpha * direction
            generated = model.generate(
                z_perturbed, config.property_directions_num_samples, key=gen_key
            )

            axes[0, i].scatter(generated[:, 0], generated[:, 1], s=5, alpha=0.5)
            axes[0, i].set_title(rf'$\alpha = {alpha:.1f}$')
            axes[0, i].set_xlim(-4.5, 4.5)
            axes[0, i].set_ylim(-4.5, 4.5)
            axes[0, i].set_aspect('equal')
            axes[0, i].grid(True)

            plot_model_density(model, z_perturbed, axes[1, i], (-4.5, 4.5), (-4.5, 4.5))
            axes[1, i].set_xlim(-4.5, 4.5)
            axes[1, i].set_ylim(-4.5, 4.5)
            axes[1, i].set_aspect('equal')
            axes[1, i].grid(True)

        axes[0, 0].set_ylabel('Samples')
        axes[1, 0].set_ylabel('Density')

        fig.suptitle(f'Property Direction: {dir_name}', fontsize=21)
        save_figure(fig, os.path.join(save_path, f'property_direction_{slug}.pdf'))
