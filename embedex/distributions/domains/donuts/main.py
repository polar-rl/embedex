import os

import jax

from embedex.checkpointing import load_ckpt, save_ckpt
from embedex.distributions.builder import build_from_config, create_model
from embedex.distributions.domains.donuts.config import parse_args
from embedex.distributions.domains.donuts.donut import random_donuts, sample_batch
from embedex.distributions.domains.donuts.experiments import (
    run_embedding_visualization,
    run_gaussian_ood,
    run_interpolation,
    run_linear_functionals,
    run_parallelogram,
    run_property_directions,
    run_width_analysis,
)
from embedex.distributions.domains.donuts.plots import generate_paper_plots
from embedex.distributions.training import train
from embedex.distributions.utils import plot_loss_curve, plot_norm_curve
from embedex.logger import count_params


def main() -> None:
    cfg = parse_args()
    key = jax.random.key(cfg.seed)

    if cfg.model_ckpt is not None:
        cfg.run_name = os.path.basename(os.path.dirname(cfg.model_ckpt))

    figure_save_dir = os.path.join(cfg.figure_dir, cfg.run_name)

    if cfg.model_ckpt is None:
        key, model_key = jax.random.split(key, 2)
        model = create_model(cfg.model, dequantize=False, key=model_key)
        print(f'Model Parameters: {count_params(model):,}')

        def generate_batch(batch_size, num_samples, key):
            donut_key, sample_key = jax.random.split(key, 2)
            donuts = random_donuts(
                batch_size,
                cfg.distribution.mean_std,
                cfg.distribution.radius_range,
                cfg.distribution.width_range,
                donut_key,
            )
            return sample_batch(donuts, num_samples, sample_key)

        key, train_key = jax.random.split(key, 2)
        model, losses, norms = train(model, cfg.training, generate_batch, key=train_key)

        plot_loss_curve(losses, figure_save_dir, log_interval=cfg.training.log_interval)
        plot_norm_curve(norms, figure_save_dir, log_interval=cfg.training.log_interval)

        save_ckpt(model, cfg, cfg.training.num_iterations)
    else:
        key, model_key = jax.random.split(key, 2)
        model = load_ckpt(
            cfg.model_ckpt, lambda c: build_from_config(c, dequantize=False, key=model_key)
        )

    if cfg.model_ckpt is None:
        return

    key, plots_key = jax.random.split(key, 2)
    generate_paper_plots(model, figure_save_dir, cfg.visualization, plots_key)

    if cfg.experiments.width_analysis:
        print('\nRunning width analysis experiment...')
        key, width_key = jax.random.split(key, 2)
        run_width_analysis(model, cfg.experiments, cfg.distribution, figure_save_dir, width_key)

    if cfg.experiments.interpolation:
        print('Running interpolation experiment...')
        key, interp_key = jax.random.split(key, 2)
        run_interpolation(model, cfg.experiments, cfg.visualization, figure_save_dir, interp_key)

    if cfg.experiments.linear_functionals:
        print('Running linear functionals experiment...')
        key, linear_key = jax.random.split(key, 2)
        run_linear_functionals(model, cfg.experiments, figure_save_dir, linear_key)

    if cfg.experiments.gaussian_ood:
        print('Running Gaussian OOD experiment...')
        key, ood_key = jax.random.split(key, 2)
        run_gaussian_ood(model, cfg.experiments, figure_save_dir, ood_key)

    if cfg.experiments.property_directions:
        print('Running property directions experiment...')
        key, prop_key = jax.random.split(key, 2)
        run_property_directions(model, cfg.experiments, figure_save_dir, prop_key)

    if cfg.experiments.embedding_visualization:
        print('Running embedding visualization experiment...')
        key, viz_key = jax.random.split(key, 2)
        run_embedding_visualization(
            model, cfg.experiments, cfg.distribution, figure_save_dir, viz_key
        )

    if cfg.experiments.parallelogram:
        print('\nRunning parallelogram experiment...')
        key, comp_key = jax.random.split(key, 2)
        run_parallelogram(model, cfg.experiments, cfg.visualization, figure_save_dir, comp_key)


if __name__ == '__main__':
    main()
