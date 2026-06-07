import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from embedex.distributions.builder import load_ckpt
from embedex.distributions.domains.donuts.config import DonutDistributionConfig, ExperimentConfig
from embedex.distributions.domains.donuts.donut import Donut, sample_batch
from embedex.distributions.domains.donuts.experiments.property_reconstruction import (
    _sweep,
    draw_recon_panel,
)
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.utils import plot_model_density, save_figure

RECON_PROPS = ['radius', 'width']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--set-transformer', required=True, help='base set_transformer_fm ckpt (A2/A4)')
    p.add_argument(
        '--wide-pointwise',
        required=True,
        help='wide single-round pointwise ckpt (the failure case in A2 and A4)',
    )
    p.add_argument('--out-dir', default='figures/distributions/donuts/compare')
    p.add_argument('--steps', type=int, default=4, help='sweep steps per property in A2')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def load(path: str, key: jax.Array) -> FlowEmbedding:
    return load_ckpt(path, dequantize=False, key=key)


def property_reconstruction_compare(
    models: dict[str, FlowEmbedding],
    distribution: DonutDistributionConfig,
    config: ExperimentConfig,
    steps: int,
    out_path: str,
    key: jax.Array,
) -> None:
    n_embed = config.property_reconstruction_num_samples
    n_gen = config.property_reconstruction_num_generated
    sweeps = _sweep(distribution, steps)

    fig = plt.figure(figsize=(2.3 * steps * len(models), 2.3 * len(RECON_PROPS) + 0.9))
    subfigs = fig.subfigures(1, len(models), wspace=0.04)
    subfigs = [subfigs] if len(models) == 1 else list(subfigs)

    for sub, (name, model) in zip(subfigs, models.items(), strict=True):
        sub.suptitle(name, fontweight='bold')
        axes = sub.subplots(len(RECON_PROPS), steps, squeeze=False)
        for row, prop in enumerate(RECON_PROPS):
            values, build, true_of = sweeps[prop]
            for col, value in enumerate(values):
                key, panel_key = jax.random.split(key, 2)
                ax = axes[row][col]
                draw_recon_panel(ax, model, build, value, true_of, prop, n_embed, n_gen, panel_key)
                if col == 0:
                    ax.set_ylabel(prop)

    save_figure(fig, out_path)
    print(f'Saved {out_path}')


def _interpolation_path(model: FlowEmbedding, start: jax.Array, num_steps: int) -> jax.Array:
    zs = jax.vmap(model.embed, in_axes=(0, None))(start, jnp.ones(start.shape[1]))[0]
    a, b = zs[0:1], zs[1:2]
    mids = jnp.concatenate(
        [
            (num_steps - i) / (num_steps + 1) * a + (i + 1) / (num_steps + 1) * b
            for i in range(num_steps)
        ]
    )
    return jnp.concatenate((a, mids, b))


def interpolation_compare(
    rows: list[tuple[str, FlowEmbedding]],
    config: ExperimentConfig,
    out_path: str,
    key: jax.Array,
) -> None:
    donuts = [
        Donut(mean=jnp.array([0.0, -1.0]), radius=0.7, width=0.25),
        Donut(mean=jnp.array([0.0, 1.0]), radius=0.7, width=0.25),
    ]
    key, sample_key = jax.random.split(key, 2)
    start = sample_batch(donuts, config.interpolation_num_samples, sample_key)

    num_steps = config.interpolation_num_steps
    total = 2 + num_steps
    xlim, ylim = (-1.5, 1.5), (-2.5, 2.5)
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig, axes = plt.subplots(
        len(rows), total, figsize=(2.3 * total, 2.3 * aspect * len(rows)), squeeze=False
    )
    for r, (name, model) in enumerate(rows):
        zs = _interpolation_path(model, start, num_steps)
        for c in range(total):
            ax = axes[r][c]
            plot_model_density(model, zs[c], ax, xlim, ylim)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect('equal')
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name)
        if r == 0:
            axes[r][0].set_title('start')
            axes[r][-1].set_title('end')
    fig.suptitle('Embedding interpolation between two donuts (same endpoints per row)')
    fig.tight_layout()
    save_figure(fig, out_path)
    print(f'Saved {out_path}')


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    distribution = DonutDistributionConfig()
    key = jax.random.key(args.seed)

    k_st, k_wide = jax.random.split(key, 2)
    set_transformer = load(args.set_transformer, k_st)
    wide_pointwise = load(args.wide_pointwise, k_wide)

    os.makedirs(args.out_dir, exist_ok=True)
    st_label, pw_label = 'Set Transformer (3 rounds)', 'Pointwise (1 round)'

    key, a2_key = jax.random.split(key, 2)
    property_reconstruction_compare(
        {pw_label: wide_pointwise, st_label: set_transformer},
        distribution,
        config,
        args.steps,
        os.path.join(args.out_dir, 'property_reconstruction.pdf'),
        a2_key,
    )

    key, a4_key = jax.random.split(key, 2)
    interpolation_compare(
        [(st_label, set_transformer), (pw_label, wide_pointwise)],
        config,
        os.path.join(args.out_dir, 'interpolation.pdf'),
        a4_key,
    )


if __name__ == '__main__':
    main()
