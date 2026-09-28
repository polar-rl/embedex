import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from embedex.checkpointing import load_ckpt
from embedex.distributions.builder import build_from_config
from embedex.distributions.domains.donuts.config import DonutDistributionConfig
from embedex.distributions.domains.donuts.donut import random_donuts, sample_batch
from embedex.distributions.domains.donuts.evaluate_all import discover_runs, split_run_name
from embedex.logger import count_params

TARGETS = ['center_x', 'center_y', 'radius', 'width']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=str, default='models/distributions/donuts')
    parser.add_argument('--output', type=str, default='results/donuts/probe.jsonl')
    parser.add_argument('--eval-seed', type=int, default=0)
    parser.add_argument('--num-distributions', type=int, default=256)
    parser.add_argument(
        '--n-eval',
        type=int,
        nargs='*',
        default=[128],
        help='Reference sample counts at which to embed each distribution before probing.',
    )
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--dist-batch-size', type=int, default=64)
    parser.add_argument('--probe-seed', type=int, default=0, help='K-fold shuffle seed')
    parser.add_argument('--runs', nargs='*', default=None, help='Only probe these run names')
    return parser.parse_args()


@eqx.filter_jit
def embed_chunk(model, embed_data: jax.Array) -> jax.Array:
    def per_dist(emb):
        z, _ = model.embed(emb)
        return z

    return jax.vmap(per_dist)(embed_data)


def compute_embeddings(model, data: jax.Array, n_eval: int, dist_batch_size: int) -> np.ndarray:
    num_dists = data.shape[0]
    chunks = []
    for start in range(0, num_dists, dist_batch_size):
        end = min(start + dist_batch_size, num_dists)
        chunks.append(np.asarray(embed_chunk(model, data[start:end, :n_eval])))
    return np.concatenate(chunks)


def linear_probe_r2(z: np.ndarray, y: np.ndarray, num_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(z))
    folds = np.array_split(idx, num_folds)
    per_fold = []
    for k in range(num_folds):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(num_folds) if j != k])
        z_tr, z_te = z[train], z[test]
        y_tr, y_te = y[train], y[test]

        mu, sd = z_tr.mean(0), z_tr.std(0) + 1e-8
        z_tr = (z_tr - mu) / sd
        z_te = (z_te - mu) / sd
        a_tr = np.concatenate([z_tr, np.ones((len(z_tr), 1))], axis=1)
        a_te = np.concatenate([z_te, np.ones((len(z_te), 1))], axis=1)

        w, *_ = np.linalg.lstsq(a_tr, y_tr, rcond=None)
        pred = a_te @ w
        ss_res = ((y_te - pred) ** 2).sum(0)
        ss_tot = ((y_te - y_te.mean(0)) ** 2).sum(0) + 1e-12
        per_fold.append(1.0 - ss_res / ss_tot)
    return np.stack(per_fold)


def result_row(name, seed, n_eval, num_params, per_fold, args) -> dict:
    cell, _ = split_run_name(name)
    return {
        'name': name,
        'cell': cell,
        'seed': seed,
        'n_eval': n_eval,
        'num_params': num_params,
        'eval_seed': args.eval_seed,
        'num_distributions': args.num_distributions,
        'num_folds': args.num_folds,
        'metrics': {
            f'{target}_r2': {
                'mean': float(per_fold[:, i].mean()),
                'per_fold': per_fold[:, i].tolist(),
            }
            for i, target in enumerate(TARGETS)
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
    dist_config = DonutDistributionConfig()
    donut_key, sample_key = jax.random.split(jax.random.key(args.eval_seed), 2)
    donuts = random_donuts(
        args.num_distributions,
        dist_config.mean_std,
        dist_config.radius_range,
        dist_config.width_range,
        donut_key,
    )
    data = sample_batch(donuts, max_n, sample_key)

    targets = np.stack(
        [
            np.asarray(donuts.mean[:, 0]),
            np.asarray(donuts.mean[:, 1]),
            np.asarray(donuts.radius),
            np.asarray(donuts.width),
        ],
        axis=1,
    )

    with open(output, 'a') as f:
        for run_name, ckpt_path in discover_runs(Path(args.model_dir)):
            if args.runs is not None and run_name not in args.runs:
                continue
            _, seed = split_run_name(run_name)
            model = None
            for n_eval in args.n_eval:
                if (run_name, n_eval) in done:
                    print(f'{run_name} @ n={n_eval}: already in probe results, skipping')
                    continue
                if model is None:
                    model = load_ckpt(
                        str(ckpt_path),
                        lambda c: build_from_config(c, dequantize=False, key=jax.random.key(0)),
                    )
                    num_params = count_params(model)
                z = compute_embeddings(model, data, n_eval, args.dist_batch_size)
                per_fold = linear_probe_r2(z, targets, args.num_folds, args.probe_seed)
                row = result_row(run_name, seed, n_eval, num_params, per_fold, args)
                f.write(json.dumps(row) + '\n')
                f.flush()
                summary = '  '.join(
                    f'{t}_r2={per_fold[:, i].mean():.3f}' for i, t in enumerate(TARGETS)
                )
                print(f'{run_name} @ n={n_eval}: {summary}')


if __name__ == '__main__':
    main()
