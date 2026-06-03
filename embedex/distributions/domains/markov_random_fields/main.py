import os

import jax

from embedex.distributions.builder import count_parameters, create_model, load_ckpt, save_ckpt
from embedex.distributions.domains.markov_random_fields.config import parse_args
from embedex.distributions.domains.markov_random_fields.create_paper_plots import generate_paper_plots
from embedex.distributions.domains.markov_random_fields.evaluate import evaluate
from embedex.distributions.domains.markov_random_fields.mrf import random_mrfs, sample_batch
from embedex.distributions.training import train
from embedex.distributions.utils import plot_loss_curve, plot_norm_curve


def main() -> None:
    cfg = parse_args()
    key = jax.random.key(cfg.seed)

    if cfg.model_ckpt is not None:
        cfg.run_name = os.path.basename(os.path.dirname(cfg.model_ckpt))

    figure_save_dir = os.path.join(cfg.figure_dir, cfg.run_name)

    cfg.model.input_dim = cfg.distribution.num_spins
    cfg.model.output_dim = cfg.distribution.num_spins
    if cfg.model.generator.flow_type == 'dfm':
        cfg.model.generator.dfm_num_classes = cfg.distribution.num_states

    dequantize = cfg.model.generator.flow_type == 'nf'

    if cfg.model_ckpt is None:
        key, model_key = jax.random.split(key, 2)
        model = create_model(cfg.model, dequantize=dequantize, key=model_key)
        print(f'Model Parameters: {count_parameters(model):,}')

        def generate_batch(batch_size, num_samples, key):
            mrf_key, sample_key = jax.random.split(key, 2)
            mrfs = random_mrfs(
                batch_size,
                cfg.distribution.num_spins,
                cfg.distribution.num_states,
                cfg.distribution.coupling_scale,
                cfg.distribution.bias_scale,
                cfg.distribution.edge_prob,
                mrf_key,
                graph_type=cfg.distribution.graph_type,
                edge_prob_min=cfg.distribution.edge_prob_min,
                edge_prob_max=cfg.distribution.edge_prob_max,
            )
            return sample_batch(mrfs, num_samples, sample_key)

        key, train_key = jax.random.split(key, 2)
        model, losses, norms = train(model, cfg.training, generate_batch, key=train_key)

        plot_loss_curve(losses, figure_save_dir, log_interval=cfg.training.log_interval)
        plot_norm_curve(norms, figure_save_dir, log_interval=cfg.training.log_interval)

        save_ckpt(model, cfg, cfg.training.num_iterations)
    else:
        key, model_key = jax.random.split(key, 2)
        model = load_ckpt(cfg.model_ckpt, dequantize=dequantize, key=model_key)

    key, plots_key = jax.random.split(key, 2)
    generate_paper_plots(model, figure_save_dir, cfg.visualization, cfg.distribution, plots_key)

    print('Running model evaluation...')
    key, eval_key = jax.random.split(key, 2)
    evaluate(model, cfg.evaluation, cfg.distribution, eval_key)


if __name__ == '__main__':
    main()
