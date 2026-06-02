import itertools
import os

import jax
import jax.numpy as jnp
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from embedex.distributions.domains.donuts.config import ExperimentConfig
from embedex.distributions.domains.donuts.donut import Donut, sample_batch
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.utils import save_figure

BASE = {'mean_x': 0.0, 'mean_y': 0.0, 'radius': 1.0, 'width': 0.25}
PROPERTIES = {
    'Pos X': {'low': {'mean_offset_x': -1.2}, 'high': {'mean_offset_x': 1.2}},
    'Pos Y': {'low': {'mean_offset_y': -1.2}, 'high': {'mean_offset_y': 1.2}},
    'Radius': {'low': {'radius': 0.7}, 'high': {'radius': 1.8}},
    'Width': {'low': {'width': 0.1}, 'high': {'width': 0.5}},
}


def _make_donut(overrides: dict[str, float]) -> Donut:
    mean_x = BASE['mean_x'] + overrides.get('mean_offset_x', 0.0)
    mean_y = BASE['mean_y'] + overrides.get('mean_offset_y', 0.0)
    radius = overrides.get('radius', BASE['radius'])
    width = overrides.get('width', BASE['width'])

    return Donut(mean=jnp.array([mean_x, mean_y]), radius=radius, width=width)


def _embed_donut(
    model: FlowEmbedding,
    donut: Donut,
    num_samples: int,
    key: jax.Array,
) -> jax.Array:
    samples = sample_batch([donut], num_samples, key)[0]
    return model.embed(samples, jnp.ones(samples.shape[0]))[0]


def run_compositional(
    model: FlowEmbedding,
    config: ExperimentConfig,
    save_path: str,
    key: jax.Array,
) -> None:
    prop_names = list(PROPERTIES.keys())
    pairs = list(itertools.combinations(prop_names, 2))

    print(f'  Testing {len(pairs)} property pairs...')

    for p1_name, p2_name in pairs:
        p1 = PROPERTIES[p1_name]
        p2 = PROPERTIES[p2_name]

        donut_a = _make_donut({**p1['low'], **p2['low']})
        donut_b = _make_donut({**p1['high'], **p2['low']})
        donut_c = _make_donut({**p1['low'], **p2['high']})
        donut_d = _make_donut({**p1['high'], **p2['high']})

        key, key_a, key_b, key_c, key_d = jax.random.split(key, 5)
        z_a = _embed_donut(model, donut_a, config.compositional_num_samples, key_a)
        z_b = _embed_donut(model, donut_b, config.compositional_num_samples, key_b)
        z_c = _embed_donut(model, donut_c, config.compositional_num_samples, key_c)
        z_d = _embed_donut(model, donut_d, config.compositional_num_samples, key_d)

        z_d_hat = z_b + z_c - z_a

        l2_true = jnp.linalg.norm(z_d)
        l2_error = jnp.linalg.norm(z_d_hat - z_d)
        cos_sim = jnp.dot(z_d_hat, z_d) / (jnp.linalg.norm(z_d_hat) * jnp.linalg.norm(z_d) + 1e-8)
        print(
            f'  {p1_name} × {p2_name}:  L2 error = {l2_error:.4f}  '
            f'(‖z_D‖ = {l2_true:.4f})  cos sim = {cos_sim:.4f}'
        )

        key, gen_true_key, gen_pred_key = jax.random.split(key, 3)
        samples_d_true = model.generate(z_d, config.compositional_num_samples, key=gen_true_key)
        samples_d_pred = model.generate(z_d_hat, config.compositional_num_samples, key=gen_pred_key)

        key, gen_a_key, gen_b_key, gen_c_key = jax.random.split(key, 4)
        samples_a = model.generate(z_a, config.compositional_num_samples, key=gen_a_key)
        samples_b = model.generate(z_b, config.compositional_num_samples, key=gen_b_key)
        samples_c = model.generate(z_c, config.compositional_num_samples, key=gen_c_key)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        all_samples = [
            (samples_a, f'A: {p1_name}↓ {p2_name}↓'),
            (samples_b, f'B: {p1_name}↑ {p2_name}↓'),
            (samples_c, f'C: {p1_name}↓ {p2_name}↑'),
        ]

        for i, (samp, title) in enumerate(all_samples):
            seaborn.kdeplot(
                {'x': np.array(samp[:, 0]), 'y': np.array(samp[:, 1])},
                x='x',
                y='y',
                fill=True,
                alpha=0.5,
                ax=axes[0, i],
            )
            axes[0, i].set_title(title)
            axes[0, i].set_xlim(-4.5, 4.5)
            axes[0, i].set_ylim(-4.5, 4.5)
            axes[0, i].set_aspect('equal')
            axes[0, i].grid(True)

        seaborn.kdeplot(
            {'x': np.array(samples_d_true[:, 0]), 'y': np.array(samples_d_true[:, 1])},
            x='x',
            y='y',
            fill=True,
            alpha=0.5,
            ax=axes[1, 0],
        )
        axes[1, 0].set_title(f'D (actual): {p1_name}↑ {p2_name}↑')
        axes[1, 0].set_xlim(-4.5, 4.5)
        axes[1, 0].set_ylim(-4.5, 4.5)
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True)

        seaborn.kdeplot(
            {'x': np.array(samples_d_pred[:, 0]), 'y': np.array(samples_d_pred[:, 1])},
            x='x',
            y='y',
            fill=True,
            alpha=0.5,
            ax=axes[1, 1],
        )
        axes[1, 1].set_title('D̂ = B + C − A (predicted)')
        axes[1, 1].set_xlim(-4.5, 4.5)
        axes[1, 1].set_ylim(-4.5, 4.5)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].grid(True)

        seaborn.kdeplot(
            {'x': np.array(samples_d_true[:, 0]), 'y': np.array(samples_d_true[:, 1])},
            x='x',
            y='y',
            fill=False,
            alpha=0.7,
            ax=axes[1, 2],
            color='blue',
        )
        seaborn.kdeplot(
            {'x': np.array(samples_d_pred[:, 0]), 'y': np.array(samples_d_pred[:, 1])},
            x='x',
            y='y',
            fill=False,
            alpha=0.7,
            ax=axes[1, 2],
            color='red',
        )
        axes[1, 2].set_title(f'Overlay (L2={l2_error:.3f}, cos={cos_sim:.3f})')
        axes[1, 2].set_xlim(-4.5, 4.5)
        axes[1, 2].set_ylim(-4.5, 4.5)
        axes[1, 2].set_aspect('equal')
        axes[1, 2].grid(True)
        axes[1, 2].legend(
            handles=[
                mlines.Line2D([], [], color='blue', label='Actual'),
                mlines.Line2D([], [], color='red', label='Predicted'),
            ]
        )

        fig.suptitle(f'Parallelogram Test: {p1_name} × {p2_name}', fontsize=14)
        fig.tight_layout()

        slug = f'{p1_name}_{p2_name}'.replace(' ', '_').lower()
        save_figure(fig, os.path.join(save_path, f'compositional_{slug}.pdf'))
        plt.close(fig)
