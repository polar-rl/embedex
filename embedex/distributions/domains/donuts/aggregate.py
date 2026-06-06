import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ['nll', 'swd', 'center_error', 'radius_error', 'width_error']
BOOTSTRAP_RESAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=str, default='results/donuts/results.jsonl')
    parser.add_argument('--metric', nargs='*', default=['nll', 'swd'], choices=METRICS)
    parser.add_argument('--n-eval', type=int, nargs='*', default=None)
    parser.add_argument('--compare', action='store_true', help='Paired bootstrap, all cell pairs')
    parser.add_argument('--latex', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='Bootstrap resampling seed')
    parser.add_argument(
        '--nan-report',
        type=str,
        default=None,
        help='Where to write the NaN diagnostics (default: nan_report.txt beside --results)',
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def group_by_cell(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row['cell'], row['n_eval'])].append(row)
    return groups


def summary_table(groups: dict, metrics: list[str], n_evals: list[int], latex: bool) -> None:
    cells = sorted({cell for cell, _ in groups})

    for n_eval in n_evals:
        print(f'\n=== n_eval = {n_eval} ===')
        header = ['cell', 'seeds'] + metrics
        lines = []
        for cell in cells:
            rows = groups.get((cell, n_eval))
            if not rows:
                continue
            entries = [cell, str(len(rows))]
            for metric in metrics:
                # Per-seed mean over that seed's finite distributions, then mean
                # ± SE across seeds. nan-aware so one diverged seed's NaNs don't
                # blank the whole cell (the NaN counts live in the nan report).
                per_seed = np.array([_safe_nanmean(r['metrics'][metric]['per_dist']) for r in rows])
                finite = per_seed[np.isfinite(per_seed)]
                if len(finite) == 0:
                    entries.append('nan')
                elif len(finite) > 1:
                    se = finite.std(ddof=1) / np.sqrt(len(finite))
                    entries.append(f'{finite.mean():.4f} ± {se:.4f}')
                else:
                    entries.append(f'{finite.mean():.4f}')
            lines.append(entries)

        if latex:
            print(' & '.join(header) + r' \\')
            for entries in lines:
                print(' & '.join(entries).replace('±', r'$\pm$') + r' \\')
        else:
            widths = [max(len(h), *(len(e[i]) for e in lines)) for i, h in enumerate(header)]
            print('  '.join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
            for entries in lines:
                print('  '.join(e.ljust(w) for e, w in zip(entries, widths, strict=True)))


def _safe_nanmean(values) -> float:
    """Mean over finite entries; NaN only if every entry is non-finite."""
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    return float(arr[mask].mean()) if mask.any() else float('nan')


def seed_averaged_per_dist(rows: list[dict], metric: str) -> np.ndarray:
    """Per-distribution mean across seeds, ignoring non-finite seeds.

    A distribution stays NaN only if it is non-finite for every seed; otherwise
    the finite seeds carry it. (Done without np.nanmean to avoid all-NaN-slice
    warnings.)
    """
    stacked = np.array([r['metrics'][metric]['per_dist'] for r in rows], dtype=float)
    mask = np.isfinite(stacked)
    counts = mask.sum(axis=0)
    summed = np.where(mask, stacked, 0.0).sum(axis=0)
    return np.where(counts > 0, summed / np.maximum(counts, 1), np.nan)


def paired_bootstrap(diffs: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    idx = rng.integers(0, len(diffs), size=(BOOTSTRAP_RESAMPLES, len(diffs)))
    resampled = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(resampled, [2.5, 97.5])
    return diffs.mean(), lo, hi


def comparison_table(groups: dict, metrics: list[str], n_evals: list[int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    cells = sorted({cell for cell, _ in groups})

    for n_eval in n_evals:
        print(f'\n=== paired comparisons, n_eval = {n_eval} ===')
        for metric in metrics:
            print(f'\n[{metric}] (negative diff favors A)')
            for i, cell_a in enumerate(cells):
                for cell_b in cells[i + 1 :]:
                    rows_a = groups.get((cell_a, n_eval))
                    rows_b = groups.get((cell_b, n_eval))
                    if not rows_a or not rows_b:
                        continue
                    diffs = seed_averaged_per_dist(rows_a, metric) - seed_averaged_per_dist(
                        rows_b, metric
                    )
                    valid = np.isfinite(diffs)
                    dropped = int((~valid).sum())
                    diffs = diffs[valid]
                    if len(diffs) == 0:
                        print(f'    {cell_a} vs {cell_b}: all distributions NaN, skipped')
                        continue
                    mean, lo, hi = paired_bootstrap(diffs, rng)
                    significant = '*' if (lo > 0) == (hi > 0) else ' '
                    note = f'  (dropped {dropped} NaN dist)' if dropped else ''
                    print(
                        f'  {significant} {cell_a} vs {cell_b}: '
                        f'{mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]{note}'
                    )


def write_nan_report(rows: list[dict], path: Path) -> int:
    """Write NaN diagnostics to a standalone file. Returns total NaN count.

    Two sections: (1) per-distribution non-finite metric values, per run/metric,
    listing only entries with at least one NaN; (2) the generated-sample
    nan_rate (fraction of non-finite samples) where it is positive. This is the
    paper-trail for the nan-aware aggregation — it records exactly which seeds
    and distributions were carried by their finite counterparts.
    """
    by_ne = defaultdict(list)
    for r in rows:
        by_ne[r['n_eval']].append(r)

    count_lines, rate_lines = [], []
    total_nan = 0
    for n_eval in sorted(by_ne):
        ne_counts, ne_rates = [], []
        for r in sorted(by_ne[n_eval], key=lambda r: r['name']):
            for metric, m in r['metrics'].items():
                if metric == 'nan_rate':
                    continue
                vals = np.asarray(m['per_dist'], dtype=float)
                n_nan = int((~np.isfinite(vals)).sum())
                if n_nan:
                    total_nan += n_nan
                    ne_counts.append(f'  {r["name"]:<28} {metric:<13} {n_nan}/{len(vals)}')
            rate = r['metrics'].get('nan_rate')
            if rate is not None and _safe_nanmean(rate['per_dist']) > 0:
                ne_rates.append(f'  {r["name"]:<28} {_safe_nanmean(rate["per_dist"]):.5f}')
        if ne_counts:
            count_lines.append(f'n_eval={n_eval}')
            count_lines.extend(ne_counts)
        if ne_rates:
            rate_lines.append(f'n_eval={n_eval}')
            rate_lines.extend(ne_rates)

    out = [
        '# NaN report (generated by aggregate.py)',
        '# Aggregation is nan-aware: finite seeds carry a distribution; a metric',
        '# is NaN only where every seed is non-finite. Counts below are what was',
        '# dropped/carried, not silently ignored.',
        '',
        '## Non-finite per-distribution metric values (count / total distributions)',
    ]
    out += count_lines or ['  (none)']
    out += [
        '',
        '## Generated-sample nan_rate (mean fraction of non-finite samples, where > 0)',
    ]
    out += rate_lines or ['  (none — or nan_rate not recorded; re-run evaluate_all to populate)']
    out.append('')

    path.write_text('\n'.join(out))
    return total_nan


def main() -> None:
    args = parse_args()

    results_path = Path(args.results)
    rows = load_rows(results_path)
    groups = group_by_cell(rows)

    n_evals = args.n_eval or sorted({n for _, n in groups})

    summary_table(groups, args.metric, n_evals, args.latex)
    if args.compare:
        comparison_table(groups, args.metric, n_evals, args.seed)

    nan_path = (
        Path(args.nan_report) if args.nan_report else results_path.with_name('nan_report.txt')
    )
    total_nan = write_nan_report(rows, nan_path)
    print(f'\nNaN report ({total_nan} non-finite metric values) written to {nan_path}')


if __name__ == '__main__':
    main()
