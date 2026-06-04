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
                means = np.array([r['metrics'][metric]['mean'] for r in rows])
                if len(means) > 1:
                    se = means.std(ddof=1) / np.sqrt(len(means))
                    entries.append(f'{means.mean():.4f} ± {se:.4f}')
                else:
                    entries.append(f'{means.mean():.4f}')
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


def seed_averaged_per_dist(rows: list[dict], metric: str) -> np.ndarray:
    return np.array([r['metrics'][metric]['per_dist'] for r in rows]).mean(axis=0)


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
                    mean, lo, hi = paired_bootstrap(diffs, rng)
                    significant = '*' if (lo > 0) == (hi > 0) else ' '
                    print(
                        f'  {significant} {cell_a} vs {cell_b}: {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]'
                    )


def main() -> None:
    args = parse_args()

    rows = load_rows(Path(args.results))
    groups = group_by_cell(rows)

    n_evals = args.n_eval or sorted({n for _, n in groups})

    summary_table(groups, args.metric, n_evals, args.latex)
    if args.compare:
        comparison_table(groups, args.metric, n_evals, args.seed)


if __name__ == '__main__':
    main()
