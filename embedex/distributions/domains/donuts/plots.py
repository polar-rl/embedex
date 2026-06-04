import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = ['nll', 'swd', 'center_error', 'radius_error', 'width_error']
BASELINES = ['kde', 'oracle']

METRIC_LABELS = {
    'nll': 'Held-out NLL',
    'swd': 'Sliced Wasserstein-2',
    'center_error': 'Center error',
    'radius_error': 'Radius error',
    'width_error': 'Width error',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=str, default='results/donuts/results.jsonl')
    parser.add_argument('--figure-dir', type=str, default='figures/distributions/donuts/eval')
    parser.add_argument('--metric', nargs='*', default=METRICS, choices=METRICS)
    parser.add_argument('--format', type=str, default='pdf', choices=['pdf', 'png'])
    return parser.parse_args()


def load_curves(path: Path, metric: str) -> dict[str, dict[int, np.ndarray]]:
    curves: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            curves[row['cell']][row['n_eval']].append(row['metrics'][metric]['mean'])

    return {
        cell: {n: np.array(means) for n, means in by_n.items()} for cell, by_n in curves.items()
    }


def plot_metric(curves: dict, metric: str, figure_dir: Path, fmt: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for cell in sorted(curves):
        by_n = curves[cell]
        n_vals = np.array(sorted(by_n))
        means = np.array([by_n[n].mean() for n in n_vals])
        style = {'linestyle': '--', 'alpha': 0.8} if cell in BASELINES else {'linestyle': '-'}

        (line,) = ax.plot(n_vals, means, marker='o', markersize=3, label=cell, **style)

        ses = np.array(
            [
                by_n[n].std(ddof=1) / np.sqrt(len(by_n[n])) if len(by_n[n]) > 1 else 0.0
                for n in n_vals
            ]
        )
        if ses.any():
            ax.fill_between(
                n_vals, means - ses, means + ses, color=line.get_color(), alpha=0.15, lw=0
            )

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Embedding sample count (n_eval)')
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    figure_dir.mkdir(parents=True, exist_ok=True)
    out = figure_dir / f'{metric}_vs_n_eval.{fmt}'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()

    results = Path(args.results)
    figure_dir = Path(args.figure_dir)

    for metric in args.metric:
        curves = load_curves(results, metric)
        if not curves:
            print(f'{metric}: no rows found, skipping')
            continue
        out = plot_metric(curves, metric, figure_dir, args.format)
        print(f'{metric}: wrote {out}')


if __name__ == '__main__':
    main()
