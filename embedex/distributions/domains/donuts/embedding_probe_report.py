import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from embedex.distributions.domains.donuts.aggregate import CELL_DISPLAY
from embedex.distributions.domains.experiment_utils import mean_se
from embedex.plotting import save_figure

TARGETS = ['center_x', 'center_y', 'radius', 'width']
BASE_DIM = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=str, default='results/donuts/probe.jsonl')
    parser.add_argument('--mode', choices=['size', 'rounds'], required=True)
    parser.add_argument(
        '--n-eval', type=int, default=None, help='Which reference n_eval to report (default: max)'
    )
    parser.add_argument('--out', type=str, default=None, help='Figure path (default: derived)')
    parser.add_argument('--latex', action='store_true', help='Also print a paper-ready LaTeX table')
    parser.add_argument(
        '--latex-dir',
        type=str,
        default=None,
        help='Write the LaTeX table here as probe_<mode>.tex (implies --latex)',
    )
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
                return {'block': block, 'kind': 'dim', 'dim': val, 'mechanism': 'depth'}
            if kind == 'rounds':
                return {'block': block, 'kind': 'rounds', 'dim': BASE_DIM, 'mechanism': 'depth'}
            return {'block': block, 'kind': 'width', 'dim': BASE_DIM, 'mechanism': 'width'}
    return {'block': cell, 'kind': 'base', 'dim': BASE_DIM, 'mechanism': 'depth'}


MODE_KINDS = {'size': {'base', 'dim'}, 'rounds': {'base', 'rounds', 'width'}}


def series_key(meta: dict, mode: str):
    return meta['block'] if mode == 'size' else (meta['block'], meta['mechanism'])


def series_label(skey) -> str:
    return skey if isinstance(skey, str) else f'{skey[0]} ({skey[1]})'


def x_of(meta: dict, row: dict, mode: str) -> float:
    return meta['dim'] if mode == 'size' else row['num_params']


def collect_series(rows: list[dict], mode: str) -> dict:
    series: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        meta = parse_cell(r['cell'])
        if meta['kind'] not in MODE_KINDS[mode]:
            continue
        sk = series_key(meta, mode)
        x = x_of(meta, r, mode)
        for t in TARGETS:
            series[sk][x][t].append(r['metrics'][f'{t}_r2']['mean'])

    full = {sk: by_x for sk, by_x in series.items() if len(by_x) >= 2}
    dropped = [sk for sk, by_x in series.items() if len(by_x) < 2]
    if dropped:
        print(
            'Dropped single-point series: '
            + ', '.join(series_label(s) for s in sorted(dropped, key=str))
        )
    return full


def print_table(series: dict, mode: str, n_eval: int) -> None:
    x_label = 'dim' if mode == 'size' else 'num_params'
    print(f'\n=== variance explained % (n_eval={n_eval}) ===')
    header = ['series', x_label, 'seeds', *TARGETS]
    lines = []
    for sk in sorted(series, key=str):
        by_x = series[sk]
        for x in sorted(by_x):
            tvals = by_x[x]
            n = len(tvals[TARGETS[0]])
            lines.append(
                [
                    series_label(sk),
                    str(x),
                    str(n),
                    *(
                        f'{100 * mean_se(tvals[t])[0]:.1f}±{100 * mean_se(tvals[t])[1]:.1f}'
                        for t in TARGETS
                    ),
                ]
            )
    widths = [max(len(h), *(len(line[i]) for line in lines)) for i, h in enumerate(header)]
    print('  '.join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    for line in lines:
        print('  '.join(c.ljust(w) for c, w in zip(line, widths, strict=True)))


TARGET_LABEL = {
    'center_x': r'Center$_x$',
    'center_y': r'Center$_y$',
    'radius': 'Radius',
    'width': 'Width',
}


def latex_label(skey) -> str:
    if isinstance(skey, str):
        return CELL_DISPLAY.get(skey, skey.replace('_', r'\_'))
    block, mech = skey
    name = CELL_DISPLAY.get(block, block.replace('_', r'\_'))
    return f'{name} ({mech})'


def latex_table(series: dict, mode: str, n_eval: int) -> str:
    """Paper-ready booktabs table: rows = (series, x), cols = per-target VE (%)."""
    x_head = 'Dim' if mode == 'size' else 'Params'
    colspec = 'll' + 'r' * len(TARGETS)
    head = f'Series & {x_head} & ' + ' & '.join(TARGET_LABEL[t] for t in TARGETS) + r' \\'

    body = []
    for i, sk in enumerate(sorted(series, key=str)):
        if i > 0:
            body.append(r'\addlinespace')
        by_x = series[sk]
        for x in sorted(by_x):
            cells = []
            for t in TARGETS:
                m, se = mean_se(by_x[x][t])
                cells.append(f'${100 * m:.1f} \\pm {100 * se:.1f}$')
            body.append(f'{latex_label(sk)} & {int(x)} & ' + ' & '.join(cells) + r' \\')

    what = 'embedding dimension' if mode == 'size' else 'parameter count (depth vs.\\ width)'
    lines = [
        r'% Auto-generated by embedding_probe_report.py --latex. Do not hand-edit; regenerate.',
        r'\begin{table}[t]',
        r'  \centering',
        r'  \footnotesize',
        r'  \setlength{\tabcolsep}{4pt}',
        rf'  \begin{{tabular}}{{{colspec}}}',
        r'    \toprule',
        '    ' + head,
        r'    \midrule',
        *(f'    {line}' for line in body),
        r'    \bottomrule',
        r'  \end{tabular}',
        rf'  \caption{{Linear-probe variance explained (\%, i.e.\ $R^2 \times 100$) for decoding '
        rf'structural properties from the embedding ($n_{{\text{{eval}}}}={n_eval}$), versus '
        rf'{what}. Mean $\pm$ SE over seeds; higher is better.}}',
        rf'  \label{{tab:donut-probe-{mode}}}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def plot(series: dict, mode: str, n_eval: int, out: str) -> None:
    if mode == 'rounds':
        _plot_rounds(series, n_eval, out)
    else:
        _plot_size(series, n_eval, out)


# Depth = add rounds (solid, circles); width = widen one block (dashed, triangles).
MECH_STYLE = {
    'depth': {'linestyle': '-', 'marker': 'o', 'color': 'C0'},
    'width': {'linestyle': '--', 'marker': '^', 'color': 'C1'},
}
ROUND_LABELS = ['r1', 'r2', 'r3']


def _plot_rounds(series: dict, n_eval: int, out: str) -> None:
    targets = ['radius', 'width']
    blocks = sorted({sk[0] for sk in series})

    fig, axes = plt.subplots(
        len(targets), len(blocks), figsize=(5.0 * len(blocks), 4.2 * len(targets)), sharey=True
    )
    axes = np.atleast_2d(axes)
    for i, target in enumerate(targets):
        for j, block in enumerate(blocks):
            ax = axes[i][j]
            for mech in ('depth', 'width'):
                by_x = series.get((block, mech))
                if not by_x:
                    continue
                xs = sorted(by_x)
                xk = [x / 1000.0 for x in xs]
                ys = [100 * mean_se(by_x[x][target])[0] for x in xs]
                es = [100 * mean_se(by_x[x][target])[1] for x in xs]
                ax.errorbar(xk, ys, yerr=es, capsize=3, label=mech, **MECH_STYLE[mech])
                if mech == 'depth' and len(xs) == len(ROUND_LABELS):
                    for lbl, x, y in zip(ROUND_LABELS, xk, ys, strict=True):
                        ax.annotate(
                            lbl, (x, y), textcoords='offset points', xytext=(4, 5), fontsize=10
                        )
            ax.axhline(0, color='0.6', lw=0.8, zorder=0)
            ax.set_ylim(-25, 105)
            ax.set_title(f'{CELL_DISPLAY.get(block, block)} · {target}')
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(True, alpha=0.3)
            if i == len(targets) - 1:
                ax.set_xlabel('parameters (thousands)')
            if j == 0:
                ax.set_ylabel('variance explained (%)')

    axes[0][0].legend(loc='center right', framealpha=0.9, title='mechanism')
    fig.suptitle(f'Parameter efficiency: depth vs. width (n_eval={n_eval})', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, out)
    print(f'\nSaved figure to {out}')


def _plot_size(series: dict, n_eval: int, out: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    xticks = sorted({x for by_x in series.values() for x in by_x})
    for ax, target in zip(axes.flatten(), TARGETS, strict=True):
        for sk in sorted(series, key=str):
            by_x = series[sk]
            xs = sorted(by_x)
            ys = [100 * mean_se(by_x[x][target])[0] for x in xs]
            es = [100 * mean_se(by_x[x][target])[1] for x in xs]
            ax.errorbar(xs, ys, yerr=es, marker='o', capsize=3, label=series_label(sk))
        ax.axhline(0, color='0.6', lw=0.8, zorder=0)
        ax.set_title(TARGET_LABEL[target])
        ax.set_xlabel('embedding\\_dim')
        ax.set_ylabel('variance explained (%)')
        ax.set_ylim(-25, 105)
        ax.set_xticks(xticks)
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f'Structural decodability vs embedding\\_dim (n\\_eval={n_eval})')
    fig.tight_layout()
    save_figure(fig, out)
    print(f'\nSaved figure to {out}')


def main() -> None:
    args = parse_args()
    rows, n_eval = load_rows(Path(args.probe), args.n_eval)
    if not rows:
        raise SystemExit(f'No probe rows for n_eval={args.n_eval} in {args.probe}')
    series = collect_series(rows, args.mode)
    if not series:
        raise SystemExit(f'No multi-point {args.mode}-ablation series in {args.probe}')
    print_table(series, args.mode, n_eval)
    out = args.out or f'figures/distributions/donuts/probe_{args.mode}_n{n_eval}.pdf'
    plot(series, args.mode, n_eval, out)

    if args.latex or args.latex_dir:
        table = latex_table(series, args.mode, n_eval)
        if args.latex_dir:
            path = Path(args.latex_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / f'probe_{args.mode}.tex').write_text(table + '\n')
            print(f'Wrote {path / f"probe_{args.mode}.tex"}')
        else:
            print('\n' + table)


if __name__ == '__main__':
    main()
