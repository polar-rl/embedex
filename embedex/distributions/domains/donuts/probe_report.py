import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from embedex.distributions.utils import save_figure

TARGETS = ['center_x', 'center_y', 'radius', 'width']
BASE_DIM = 16
BASE_ROUNDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=str, default='results/donuts/probe.jsonl')
    parser.add_argument('--mode', choices=['size', 'rounds'], required=True)
    parser.add_argument(
        '--n-eval', type=int, default=None, help='Which reference n_eval to report (default: max)'
    )
    parser.add_argument('--out', type=str, default=None, help='Figure path (default: derived)')
    return parser.parse_args()


def load_rows(path: Path, n_eval: int | None) -> tuple[list[dict], int]:
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    n_evals = sorted({r['n_eval'] for r in rows})
    chosen = n_eval if n_eval is not None else max(n_evals)
    return [r for r in rows if r['n_eval'] == chosen], chosen


def parse_cell(cell: str) -> dict:
    for suffix, kind in ((r'_d(\d+)$', 'dim'), (r'_r(\d+)$', 'rounds'), (r'_w(\d+)$', 'width')):
        m = re.search(suffix, cell)
        if m:
            block = cell[: m.start()]
            val = int(m.group(1))
            if kind == 'dim':
                return {
                    'block': block,
                    'kind': 'dim',
                    'dim': val,
                    'rounds': BASE_ROUNDS,
                    'mechanism': 'depth',
                }
            if kind == 'rounds':
                return {
                    'block': block,
                    'kind': 'rounds',
                    'dim': BASE_DIM,
                    'rounds': val,
                    'mechanism': 'depth',
                }
            return {
                'block': block,
                'kind': 'width',
                'dim': BASE_DIM,
                'rounds': 1,
                'mechanism': 'width',
            }
    return {
        'block': cell,
        'kind': 'base',
        'dim': BASE_DIM,
        'rounds': BASE_ROUNDS,
        'mechanism': 'depth',
    }


MODE_KINDS = {'size': {'base', 'dim'}, 'rounds': {'base', 'rounds', 'width'}}


def aggregate(rows: list[dict], mode: str) -> dict:
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r['cell']].append(r)
    out = {}
    for cell, cell_rows in by_cell.items():
        meta = parse_cell(cell)
        if meta['kind'] not in MODE_KINDS[mode]:
            continue
        meta['num_params'] = int(np.mean([r['num_params'] for r in cell_rows]))
        r2 = {}
        for t in TARGETS:
            means = np.array([r['metrics'][f'{t}_r2']['mean'] for r in cell_rows])
            se = means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else 0.0
            r2[t] = (float(means.mean()), float(se))
        meta['seeds'] = len(cell_rows)
        meta['r2'] = r2
        out[cell] = meta
    return out


def print_table(agg: dict, mode: str, n_eval: int) -> None:
    key = 'dim' if mode == 'size' else 'num_params'
    print(f'\n=== linear-probe R^2 (n_eval={n_eval}) ===')
    header = ['cell', 'block', key, 'mech', 'seeds', *TARGETS]
    lines = []
    for cell, m in sorted(agg.items(), key=lambda kv: (kv[1]['block'], kv[1][key])):
        lines.append(
            [
                cell,
                m['block'],
                str(m[key]),
                m['mechanism'],
                str(m['seeds']),
                *(f'{m["r2"][t][0]:.3f}±{m["r2"][t][1]:.3f}' for t in TARGETS),
            ]
        )
    widths = [max(len(h), *(len(line[i]) for line in lines)) for i, h in enumerate(header)]
    print('  '.join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    for line in lines:
        print('  '.join(c.ljust(w) for c, w in zip(line, widths, strict=True)))


def plot(agg: dict, mode: str, n_eval: int, out: str) -> None:
    import matplotlib.pyplot as plt

    x_key = 'dim' if mode == 'size' else 'num_params'
    x_label = 'embedding_dim' if mode == 'size' else 'num_params'

    series = defaultdict(list)
    for m in agg.values():
        skey = m['block'] if mode == 'size' else (m['block'], m['mechanism'])
        series[skey].append(m)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, target in zip(axes.flatten(), TARGETS, strict=True):
        for skey, points in sorted(series.items(), key=lambda kv: str(kv[0])):
            points = sorted(points, key=lambda m: m[x_key])
            xs = [m[x_key] for m in points]
            ys = [m['r2'][target][0] for m in points]
            es = [m['r2'][target][1] for m in points]
            label = skey if isinstance(skey, str) else f'{skey[0]} ({skey[1]})'
            ax.errorbar(xs, ys, yerr=es, marker='o', capsize=3, label=label)
        ax.set_title(f'{target}  R$^2$')
        ax.set_xlabel(x_label)
        ax.set_ylabel('R$^2$')
        if mode == 'rounds':
            ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
    fig.suptitle(f'Structural decodability vs {x_label} (n_eval={n_eval})')
    fig.tight_layout()
    save_figure(fig, out)
    print(f'\nSaved figure to {out}')


def main() -> None:
    args = parse_args()
    rows, n_eval = load_rows(Path(args.probe), args.n_eval)
    if not rows:
        raise SystemExit(f'No probe rows for n_eval={args.n_eval} in {args.probe}')
    agg = aggregate(rows, args.mode)
    if not agg:
        raise SystemExit(
            f'No {args.mode}-ablation cells in {args.probe} (kinds {MODE_KINDS[args.mode]})'
        )
    print_table(agg, args.mode, n_eval)
    out = args.out or f'figures/distributions/donuts/probe_{args.mode}_n{n_eval}.pdf'
    plot(agg, args.mode, n_eval, out)


if __name__ == '__main__':
    main()
