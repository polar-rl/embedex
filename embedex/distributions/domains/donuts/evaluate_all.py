import argparse
import json
import re
import zlib
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from embedex.distributions.builder import count_parameters, load_ckpt
from embedex.distributions.domains.donuts.baselines import KDEBaseline, OracleDonutFit
from embedex.distributions.domains.donuts.config import DonutDistributionConfig
from embedex.distributions.domains.donuts.donut import random_donuts, sample_batch
from embedex.distributions.metrics import sliced_wasserstein_distance

N_EVAL_GRID = [16, 32, 64, 128, 256, 1024]
NUM_SWD_PROJECTIONS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=str, default='models/distributions/donuts')
    parser.add_argument('--output', type=str, default='results/donuts/results.jsonl')
    parser.add_argument('--eval-seed', type=int, default=0)
    parser.add_argument('--num-distributions', type=int, default=256)
    parser.add_argument('--n-eval', type=int, nargs='*', default=N_EVAL_GRID)
    parser.add_argument('--num-logprob-samples', type=int, default=512)
    parser.add_argument('--num-generated-samples', type=int, default=1024)
    parser.add_argument('--dist-batch-size', type=int, default=64)
    parser.add_argument('--runs', nargs='*', default=None, help='Only evaluate these run names')
    parser.add_argument('--skip-baselines', action='store_true')
    return parser.parse_args()


def discover_runs(model_dir: Path) -> list[tuple[str, Path]]:
    runs = []
    for run_dir in sorted(model_dir.iterdir()):
        if run_dir.is_dir():
            ckpts = sorted(run_dir.glob('*.eqx'))
            if ckpts:
                runs.append((run_dir.name, ckpts[-1]))
    return runs


def split_run_name(name: str) -> tuple[str, int | None]:
    match = re.fullmatch(r'(.+)_s(\d+)', name)
    if match:
        return match.group(1), int(match.group(2))
    return name, None


def eval_key(eval_seed: int, name: str, n_eval: int) -> jax.Array:
    key = jax.random.key(eval_seed)
    key = jax.random.fold_in(key, zlib.crc32(name.encode()))
    return jax.random.fold_in(key, n_eval)


def param_errors(
    generated: jax.Array,
    true_mean: jax.Array,
    true_radius: jax.Array,
    true_width: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    center = jnp.mean(generated, axis=0)
    center_error = jnp.linalg.norm(center - true_mean)

    dists = jnp.linalg.norm(generated - center, axis=-1)
    radius = jnp.mean(dists)
    radius_error = jnp.abs(radius - true_radius)

    width = jnp.sqrt(12.0) * jnp.std(dists) / (radius + 1e-8)
    width_error = jnp.abs(width - true_width)

    return center_error, radius_error, width_error


@eqx.filter_jit
def eval_model_chunk(
    model,
    embed_data: jax.Array,
    holdout: jax.Array,
    reference: jax.Array,
    true_means: jax.Array,
    true_radii: jax.Array,
    true_widths: jax.Array,
    num_generated: int,
    key: jax.Array,
):
    keys = jax.random.split(key, embed_data.shape[0])

    def per_dist(emb, hold, ref, t_mean, t_radius, t_width, k):
        gen_key, swd_key = jax.random.split(k, 2)
        z, _ = model.embed(emb, jnp.ones(emb.shape[0]))
        generated = model.generate(z, num_generated, key=gen_key)
        # Exact log-density path; the stochastic trace estimator is biased in
        # probability space and inflates FM densities (see donuts/evaluate.py).
        log_probs, _, _ = model.log_prob(hold, z, key=None)
        nll = -jnp.mean(log_probs)
        swd = sliced_wasserstein_distance(ref, generated, NUM_SWD_PROJECTIONS, swd_key)
        center_error, radius_error, width_error = param_errors(generated, t_mean, t_radius, t_width)
        # Fraction of non-finite generated samples: the NF inverse-sampling
        # instability shows up here, and it's the root cause of NaN swd/errors.
        nan_rate = 1.0 - jnp.mean(jnp.all(jnp.isfinite(generated), axis=-1))
        return nll, swd, center_error, radius_error, width_error, nan_rate

    return jax.vmap(per_dist)(
        embed_data, holdout, reference, true_means, true_radii, true_widths, keys
    )


@eqx.filter_jit
def eval_baseline_chunk(
    fit_fn,
    embed_data: jax.Array,
    holdout: jax.Array,
    reference: jax.Array,
    true_means: jax.Array,
    true_radii: jax.Array,
    true_widths: jax.Array,
    num_generated: int,
    key: jax.Array,
):
    keys = jax.random.split(key, embed_data.shape[0])

    def per_dist(emb, hold, ref, t_mean, t_radius, t_width, k):
        gen_key, swd_key = jax.random.split(k, 2)
        baseline = fit_fn(emb)
        generated = baseline.sample(num_generated, gen_key)
        nll = -jnp.mean(baseline.log_prob(hold))
        swd = sliced_wasserstein_distance(ref, generated, NUM_SWD_PROJECTIONS, swd_key)
        center_error, radius_error, width_error = param_errors(generated, t_mean, t_radius, t_width)
        nan_rate = 1.0 - jnp.mean(jnp.all(jnp.isfinite(generated), axis=-1))
        return nll, swd, center_error, radius_error, width_error, nan_rate

    return jax.vmap(per_dist)(
        embed_data, holdout, reference, true_means, true_radii, true_widths, keys
    )


METRIC_NAMES = ['nll', 'swd', 'center_error', 'radius_error', 'width_error', 'nan_rate']


def evaluate_subject(evaluate_chunk, donuts, data, n_eval, args, key) -> dict[str, np.ndarray]:
    max_n = max(args.n_eval)
    num_dists = data.shape[0]
    chunks = []

    for start in range(0, num_dists, args.dist_batch_size):
        end = min(start + args.dist_batch_size, num_dists)
        chunk_key = jax.random.fold_in(key, start)
        chunks.append(
            evaluate_chunk(
                data[start:end, :n_eval],
                data[start:end, max_n : max_n + args.num_logprob_samples],
                data[start:end, max_n + args.num_logprob_samples :],
                donuts.mean[start:end],
                donuts.radius[start:end],
                donuts.width[start:end],
                chunk_key,
            )
        )

    return {
        name: np.concatenate([np.asarray(chunk[i]) for chunk in chunks])
        for i, name in enumerate(METRIC_NAMES)
    }


def result_row(name: str, seed, n_eval: int, num_params, values: dict, args) -> dict:
    cell, _ = split_run_name(name)
    return {
        'name': name,
        'cell': cell,
        'seed': seed,
        'n_eval': n_eval,
        'num_params': num_params,
        'eval_seed': args.eval_seed,
        'num_distributions': args.num_distributions,
        'metrics': {
            metric: {'mean': float(np.mean(vals)), 'per_dist': vals.tolist()}
            for metric, vals in values.items()
        },
    }


def main() -> None:
    args = parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        with open(output) as f:
            for line in f:
                row = json.loads(line)
                done.add((row['name'], row['n_eval']))

    max_n = max(args.n_eval)
    total = max_n + args.num_logprob_samples + args.num_generated_samples
    dist_config = DonutDistributionConfig()

    donut_key, sample_key = jax.random.split(jax.random.key(args.eval_seed), 2)
    donuts = random_donuts(
        args.num_distributions,
        dist_config.mean_std,
        dist_config.radius_range,
        dist_config.width_range,
        donut_key,
    )
    data = sample_batch(donuts, total, sample_key)

    subjects = []
    if not args.skip_baselines:
        for baseline_name, fit_fn in [('kde', KDEBaseline.fit), ('oracle', OracleDonutFit.fit)]:

            def chunk_fn(emb, hold, ref, means, radii, widths, key, fit_fn=fit_fn):
                return eval_baseline_chunk(
                    fit_fn,
                    emb,
                    hold,
                    ref,
                    means,
                    radii,
                    widths,
                    args.num_generated_samples,
                    key,
                )

            subjects.append((baseline_name, None, None, chunk_fn))

    for run_name, ckpt_path in discover_runs(Path(args.model_dir)):
        if args.runs is not None and run_name not in args.runs:
            continue
        model = load_ckpt(str(ckpt_path), dequantize=False, key=jax.random.key(0))
        _, seed = split_run_name(run_name)

        def chunk_fn(emb, hold, ref, means, radii, widths, key, model=model):
            return eval_model_chunk(
                model,
                emb,
                hold,
                ref,
                means,
                radii,
                widths,
                args.num_generated_samples,
                key,
            )

        subjects.append((run_name, seed, count_parameters(model), chunk_fn))

    with open(output, 'a') as f:
        for name, seed, num_params, chunk_fn in subjects:
            for n_eval in args.n_eval:
                if (name, n_eval) in done:
                    print(f'{name} @ n={n_eval}: already in results, skipping')
                    continue

                key = eval_key(args.eval_seed, name, n_eval)
                values = evaluate_subject(chunk_fn, donuts, data, n_eval, args, key)
                row = result_row(name, seed, n_eval, num_params, values, args)

                f.write(json.dumps(row) + '\n')
                f.flush()
                summary = '  '.join(f'{m}={np.mean(v):.4f}' for m, v in values.items())
                print(f'{name} @ n={n_eval}: {summary}')


if __name__ == '__main__':
    main()
