"""Shared plumbing for the per-domain experiment pipelines (sweep, evaluate, aggregate)."""

import argparse
import json
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

import jax
import matplotlib as mpl
import numpy as np
import yaml

BOOTSTRAP_RESAMPLES = 10_000

PLOT_RC = {
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 14,
}


def apply_plot_style() -> None:
    mpl.rcParams.update(PLOT_RC)


def run_sweep(
    domain_dir: Path,
    cells: list[str],
    default_model_dir: str,
    default_num_iterations: int = 30000,
) -> None:
    """Train cells x seeds by shelling out to the domain's main.py.

    Resumable: runs whose final checkpoint exists are skipped. `--suffix` names
    variant runs `<cell>_<suffix>_s<seed>` so they do not collide with the base grid.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', nargs='*', default=cells, choices=cells)
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--suffix',
        type=str,
        default=None,
        help='Run-name suffix for variants, e.g. --suffix r1 names runs <cell>_r1_s<seed>. '
        'Use alongside --set so variant checkpoints do not collide with the base grid.',
    )
    parser.add_argument(
        '--set',
        dest='overrides',
        nargs='*',
        default=[],
        help='Overrides in the form key.path=value, forwarded to every run',
    )
    args = parser.parse_args()

    def final_ckpt_path(cell: str, run_name: str) -> Path:
        with open(domain_dir / 'configs' / f'{cell}.yaml') as f:
            config = yaml.safe_load(f)

        num_iterations = config.get('training', {}).get('num_iterations', default_num_iterations)
        model_dir = config.get('model_dir', default_model_dir)

        for override in args.overrides:
            key, _, value = override.partition('=')
            if key == 'training.num_iterations':
                num_iterations = int(value)
            elif key == 'model_dir':
                model_dir = value

        filename = f'{num_iterations:0{len(str(num_iterations))}}.eqx'
        return Path(model_dir) / run_name / filename

    runs = [(cell, seed) for cell in args.cells for seed in range(args.num_seeds)]
    completed, skipped, failed = [], [], []

    for i, (cell, seed) in enumerate(runs, 1):
        variant = f'{cell}_{args.suffix}' if args.suffix else cell
        run_name = f'{variant}_s{seed}'

        if final_ckpt_path(cell, run_name).exists():
            print(f'[{i}/{len(runs)}] {run_name}: checkpoint exists, skipping')
            skipped.append(run_name)
            continue

        command = [
            sys.executable,
            str(domain_dir / 'main.py'),
            '--config',
            str(domain_dir / 'configs' / f'{cell}.yaml'),
            '--set',
            f'run_name={run_name}',
            f'seed={seed}',
            f'training.seed={seed}',
            *args.overrides,
        ]

        print(f'[{i}/{len(runs)}] {run_name}: {" ".join(command)}')
        if args.dry_run:
            continue

        start = time.monotonic()
        result = subprocess.run(command)
        elapsed = time.monotonic() - start

        if result.returncode == 0:
            print(f'[{i}/{len(runs)}] {run_name}: done in {elapsed / 60:.1f} min')
            completed.append(run_name)
        else:
            print(f'[{i}/{len(runs)}] {run_name}: FAILED (exit {result.returncode})')
            failed.append(run_name)

    print(f'\nCompleted: {len(completed)}  Skipped: {len(skipped)}  Failed: {len(failed)}')
    if failed:
        print('Failed runs:', ', '.join(failed))
        sys.exit(1)


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


def eval_key(eval_seed: int, name: str, fold: int = 0) -> jax.Array:
    key = jax.random.key(eval_seed)
    key = jax.random.fold_in(key, zlib.crc32(name.encode()))
    return jax.random.fold_in(key, fold)


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def mean_se(values) -> tuple[float, float]:
    """Mean and standard error over seeds (ddof=1; SE=0 for a single value)."""
    arr = np.array(values, dtype=float)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), se


def safe_nanmean(values) -> float:
    """Mean over finite entries; NaN only if every entry is non-finite."""
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    return float(arr[mask].mean()) if mask.any() else float('nan')


def cell_mean_se(rows: list[dict], metric: str) -> tuple[float, float, int]:
    per_seed = np.array([safe_nanmean(r['metrics'][metric]['per_dist']) for r in rows])
    finite = per_seed[np.isfinite(per_seed)]
    if len(finite) == 0:
        return float('nan'), 0.0, 0
    se = float(finite.std(ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0
    return float(finite.mean()), se, len(finite)


def seed_averaged_per_dist(rows: list[dict], metric: str) -> np.ndarray:
    stacked = np.array([r['metrics'][metric]['per_dist'] for r in rows], dtype=float)
    mask = np.isfinite(stacked)
    counts = mask.sum(axis=0)
    summed = np.where(mask, stacked, 0.0).sum(axis=0)
    return np.where(counts > 0, summed / np.maximum(counts, 1), np.nan)


def paired_bootstrap(
    diffs: np.ndarray,
    rng: np.random.Generator,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float]:
    idx = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
    resampled = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(resampled, [2.5, 97.5])
    return diffs.mean(), lo, hi


def latex_table(
    colspec: str,
    header: str,
    body: list[str],
    caption: str,
    label: str,
    generated_by: str,
) -> str:
    lines = [
        rf'% Auto-generated by {generated_by}. Do not hand-edit; regenerate.',
        r'\begin{table}[t]',
        r'  \centering',
        r'  \footnotesize',
        rf'  \begin{{tabular}}{{{colspec}}}',
        r'    \toprule',
        '    ' + header,
        r'    \midrule',
        *(f'    {line}' for line in body),
        r'    \bottomrule',
        r'  \end{tabular}',
        rf'  \caption{{{caption}}}',
        rf'  \label{{{label}}}',
        r'\end{table}',
    ]
    return '\n'.join(lines)
