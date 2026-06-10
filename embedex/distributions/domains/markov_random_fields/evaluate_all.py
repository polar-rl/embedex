import argparse
import json
import zlib
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from embedex.checkpointing import load_ckpt, read_ckpt_config
from embedex.distributions.builder import build_from_config
from embedex.distributions.domains.experiment_utils import discover_runs, split_run_name
from embedex.distributions.domains.markov_random_fields.mrf import (
    all_states,
    random_mrfs,
    sample_batch,
)
from embedex.distributions.domains.markov_random_fields.pmfs import (
    empirical_pmf,
    mean_field_pmf,
    pmf_metrics,
    state_indices,
)
from embedex.logger import count_params

BASELINES = {
    'empirical': lambda samples, states, num_states: empirical_pmf(
        samples, states.shape[1], num_states
    ),
    'mean_field': mean_field_pmf,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model-dir', type=str, default='models/distributions/markov_random_fields'
    )
    parser.add_argument('--output', type=str, default='results/markov_random_fields/results.jsonl')
    parser.add_argument('--eval-seed', type=int, default=0)
    parser.add_argument('--num-distributions', type=int, default=256)
    parser.add_argument('--num-embed-samples', type=int, default=128)
    parser.add_argument('--num-generated-samples', type=int, default=65536)
    parser.add_argument('--dist-chunk-size', type=int, default=16)
    parser.add_argument('--gen-chunk-size', type=int, default=8192)
    parser.add_argument('--runs', nargs='*', default=None, help='Only evaluate these run names')
    parser.add_argument('--skip-baselines', action='store_true')
    return parser.parse_args()


def family_id(dist: dict) -> str:
    graph = {
        'full': 'full',
        'random_edge_fixed_prob': f're{dist["edge_prob"]:g}',
        'random_edge_random_prob': f'rr{dist["edge_prob_min"]:g}-{dist["edge_prob_max"]:g}',
    }[dist['graph_type']]
    return (
        f'N{dist["num_spins"]}_q{dist["num_states"]}'
        f'_c{dist["coupling_scale"]:g}_b{dist["bias_scale"]:g}_{graph}'
    )


def family_key(eval_seed: int, fam_id: str) -> jax.Array:
    return jax.random.fold_in(jax.random.key(eval_seed), zlib.crc32(fam_id.encode()))


def build_eval_set(dist: dict, args: argparse.Namespace, fam_id: str):
    """Shared held-out fields for one family: exact PMFs, embed samples, floor PMFs."""
    mrf_key, sample_key, floor_key = jax.random.split(family_key(args.eval_seed, fam_id), 3)

    mrfs = random_mrfs(
        args.num_distributions,
        dist['num_spins'],
        dist['num_states'],
        dist['coupling_scale'],
        dist['bias_scale'],
        dist['edge_prob'],
        mrf_key,
        graph_type=dist['graph_type'],
        edge_prob_min=dist['edge_prob_min'],
        edge_prob_max=dist['edge_prob_max'],
    )
    log_pmfs = jax.vmap(lambda m: m.log_pmf())(mrfs)
    embed_data = np.asarray(sample_batch(mrfs, args.num_embed_samples, sample_key))

    def floor_counts(log_pmf: jax.Array, key: jax.Array) -> jax.Array:
        draws = jax.random.categorical(key, log_pmf, shape=(args.num_generated_samples,))
        return jnp.bincount(draws, length=log_pmf.shape[0])

    floor_keys = jax.random.split(floor_key, args.num_distributions)
    floor_pmfs = np.asarray(jax.vmap(floor_counts)(log_pmfs, floor_keys), dtype=float)
    floor_pmfs /= args.num_generated_samples

    states = np.asarray(all_states(dist['num_spins'], dist['num_states']))
    return states, np.asarray(jnp.exp(log_pmfs)), embed_data, floor_pmfs


@eqx.filter_jit
def _embed_chunk(model, data: jax.Array) -> jax.Array:
    return jax.vmap(lambda xs: model.embed(xs, jnp.ones(xs.shape[0]))[0])(data)


@eqx.filter_jit
def _generate_chunk(model, zs: jax.Array, num_samples: int, key: jax.Array) -> jax.Array:
    keys = jax.random.split(key, zs.shape[0])
    return jax.vmap(lambda z, k: model.generate(z, num_samples, key=k))(zs, keys)


def model_pmfs(
    model,
    embed_data: np.ndarray,
    num_spins: int,
    num_states: int,
    args: argparse.Namespace,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram PMFs of generated samples per field, plus non-finite sample rates."""
    num_dists = embed_data.shape[0]
    support_size = num_states**num_spins
    pmfs = np.zeros((num_dists, support_size))
    nan_rates = np.zeros(num_dists)

    for start in range(0, num_dists, args.dist_chunk_size):
        end = min(start + args.dist_chunk_size, num_dists)
        zs = _embed_chunk(model, jnp.asarray(embed_data[start:end]))
        counts = np.zeros((end - start, support_size))

        for gen_start in range(0, args.num_generated_samples, args.gen_chunk_size):
            num = min(args.gen_chunk_size, args.num_generated_samples - gen_start)
            chunk_key = jax.random.fold_in(jax.random.fold_in(key, start), gen_start)
            generated = np.asarray(_generate_chunk(model, zs, num, chunk_key))

            flat = generated.reshape(-1, num_spins)
            finite = np.all(np.isfinite(flat), axis=-1)
            owner = np.repeat(np.arange(end - start), num)[finite]
            indices = state_indices(flat[finite], num_spins, num_states)
            counts += np.bincount(
                owner * support_size + indices, minlength=(end - start) * support_size
            ).reshape(end - start, support_size)
            nan_rates[start:end] += (~finite).reshape(end - start, num).sum(axis=1)

        pmfs[start:end] = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)

    return pmfs, nan_rates / args.num_generated_samples


def metric_arrays(
    true_pmfs: np.ndarray, subject_pmfs: np.ndarray, states: np.ndarray, num_states: int
) -> dict[str, np.ndarray]:
    per_dist = [
        pmf_metrics(true_pmf, subject_pmf, states, num_states)
        for true_pmf, subject_pmf in zip(true_pmfs, subject_pmfs, strict=True)
    ]
    return {name: np.array([m[name] for m in per_dist]) for name in per_dist[0]}


def result_row(
    name: str,
    seed: int | None,
    fam_id: str,
    dist: dict,
    num_params: int | None,
    values: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict:
    cell, _ = split_run_name(name)
    return {
        'name': name,
        'cell': cell,
        'seed': seed,
        'family': fam_id,
        'distribution': dist,
        'num_params': num_params,
        'eval_seed': args.eval_seed,
        'num_distributions': args.num_distributions,
        'num_embed_samples': args.num_embed_samples,
        'num_generated_samples': args.num_generated_samples,
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
                done.add((row['name'], row['family']))

    families: dict[str, dict] = {}
    for run_name, ckpt_path in discover_runs(Path(args.model_dir)):
        if args.runs is not None and run_name not in args.runs:
            continue
        cfg = read_ckpt_config(str(ckpt_path))
        fam_id = family_id(cfg['distribution'])
        family = families.setdefault(fam_id, {'dist': cfg['distribution'], 'runs': []})
        family['runs'].append((run_name, ckpt_path, cfg))

    with open(output, 'a') as f:

        def emit(name: str, seed, fam_id: str, dist: dict, num_params, values) -> None:
            f.write(json.dumps(result_row(name, seed, fam_id, dist, num_params, values, args)))
            f.write('\n')
            f.flush()
            summary = '  '.join(f'{m}={np.mean(v):.4f}' for m, v in values.items())
            print(f'{name} [{fam_id}]: {summary}')

        for fam_id, family in families.items():
            pending_runs = [r for r in family['runs'] if (r[0], fam_id) not in done]
            pending_floor = not args.skip_baselines and ('floor', fam_id) not in done
            pending_baselines = [
                b for b in BASELINES if not args.skip_baselines and (b, fam_id) not in done
            ]
            if not pending_runs and not pending_floor and not pending_baselines:
                print(f'[{fam_id}] all subjects already in results, skipping')
                continue

            dist = family['dist']
            num_states = dist['num_states']
            states, true_pmfs, embed_data, floor_pmfs = build_eval_set(dist, args, fam_id)

            if pending_floor:
                values = metric_arrays(true_pmfs, floor_pmfs, states, num_states)
                emit('floor', None, fam_id, dist, None, values)

            for baseline in pending_baselines:
                fit = BASELINES[baseline]
                subject_pmfs = np.stack(
                    [fit(samples, states, num_states) for samples in embed_data]
                )
                values = metric_arrays(true_pmfs, subject_pmfs, states, num_states)
                emit(baseline, None, fam_id, dist, None, values)

            for run_name, ckpt_path, cfg in pending_runs:
                dequantize = cfg['model']['generator']['flow_type'] == 'nf'
                model = load_ckpt(
                    str(ckpt_path),
                    lambda c, deq=dequantize: build_from_config(
                        c, dequantize=deq, key=jax.random.key(0)
                    ),
                )
                _, seed = split_run_name(run_name)
                key = jax.random.fold_in(
                    family_key(args.eval_seed, fam_id), zlib.crc32(run_name.encode())
                )
                pmfs, nan_rates = model_pmfs(
                    model, embed_data, dist['num_spins'], num_states, args, key
                )
                values = metric_arrays(true_pmfs, pmfs, states, num_states)
                values['nan_rate'] = nan_rates
                emit(run_name, seed, fam_id, dist, count_params(model), values)


if __name__ == '__main__':
    main()
