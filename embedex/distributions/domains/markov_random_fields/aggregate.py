import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from embedex.distributions.domains.config import MRFDataConfig
from embedex.distributions.domains.experiment_utils import (
    cell_mean_se,
    latex_table,
    load_rows,
    paired_bootstrap,
    seed_averaged_per_dist,
)

METRICS = ['jsd', 'class_mae', 'agree_mae', 'mass_outside', 'mode_coverage']
MODEL_CELLS = ['st_nf_dequant', 'st_dfm_attn', 'st_dfm']
BASELINES = ['empirical', 'mean_field']
FLOOR = 'floor'

CELL_DISPLAY = {
    'st_nf_dequant': 'NF (dequantized)',
    'st_dfm': 'DFM',
    'st_dfm_attn': 'DFM (attention)',
    'empirical': 'Empirical PMF',
    'mean_field': 'Mean-field',
    'floor': 'Estimator floor',
}

# Table columns: (metric, header, better, decimals). Support metrics get their own table.
SUMMARY_COLS = [
    ('jsd', 'JSD', 'min', 3),
    ('class_mae', 'Class MAE', 'min', 3),
    ('agree_mae', 'Agreement MAE', 'min', 3),
]
SUPPORT_COLS = [
    ('jsd', 'JSD', 'min', 3),
    ('mass_outside', 'Mass outside (\\%)', 'min', 1),
    ('mode_coverage', 'Mode coverage (\\%)', 'max', 1),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=str, default='results/markov_random_fields/results.jsonl')
    parser.add_argument('--metric', nargs='*', default=METRICS, choices=METRICS)
    parser.add_argument('--compare', action='store_true', help='Paired bootstrap, all cell pairs')
    parser.add_argument(
        '--latex-dir',
        type=str,
        default=None,
        help='Write LaTeX tables here as mrf_{summary,support,q4}.tex',
    )
    parser.add_argument('--seed', type=int, default=0, help='Bootstrap resampling seed')
    return parser.parse_args()


def group_by_family_cell(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row['family'], row['cell'])].append(row)
    return groups


def family_dists(rows: list[dict]) -> dict[str, dict]:
    return {row['family']: row['distribution'] for row in rows}


def is_default_family(dist: dict) -> bool:
    return dist == asdict(MRFDataConfig())


def base_cell(cell: str) -> str | None:
    """Map a variant cell like st_dfm_c4 to its base model cell."""
    matches = [c for c in MODEL_CELLS if cell == c or cell.startswith(f'{c}_')]
    return max(matches, key=len) if matches else None


def display(cell: str) -> str:
    cell = base_cell(cell) or cell
    return CELL_DISPLAY.get(cell, cell.replace('_', r'\_'))


def order_cells(cells: list[str]) -> tuple[list[str], list[str]]:
    models = sorted(c for c in cells if c not in BASELINES and c != FLOOR)
    baselines = [b for b in BASELINES if b in cells]
    return models, baselines


def summary_print(groups: dict, metrics: list[str]) -> None:
    families = sorted({family for family, _ in groups})
    for family in families:
        print(f'\n=== {family} ===')
        cells = sorted({cell for fam, cell in groups if fam == family})
        header = ['cell', 'seeds'] + metrics
        lines = []
        for cell in cells:
            rows = groups[(family, cell)]
            entries = [cell, str(len(rows))]
            for metric in metrics:
                if metric not in rows[0]['metrics']:
                    entries.append('--')
                    continue
                mean, se, n = cell_mean_se(rows, metric)
                entries.append(f'{mean:.4f} ± {se:.4f}' if n > 1 else f'{mean:.4f}')
            lines.append(entries)

        widths = [max(len(h), *(len(e[i]) for e in lines)) for i, h in enumerate(header)]
        print('  '.join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
        for entries in lines:
            print('  '.join(e.ljust(w) for e, w in zip(entries, widths, strict=True)))


def comparison_print(groups: dict, metrics: list[str], seed: int) -> None:
    rng = np.random.default_rng(seed)
    families = sorted({family for family, _ in groups})
    for family in families:
        cells = sorted(
            {cell for fam, cell in groups if fam == family and cell not in (*BASELINES, FLOOR)}
        )
        print(f'\n=== paired comparisons, {family} ===')
        for metric in metrics:
            print(f'\n[{metric}] (negative diff favors A)')
            for i, cell_a in enumerate(cells):
                for cell_b in cells[i + 1 :]:
                    diffs = seed_averaged_per_dist(
                        groups[(family, cell_a)], metric
                    ) - seed_averaged_per_dist(groups[(family, cell_b)], metric)
                    diffs = diffs[np.isfinite(diffs)]
                    if len(diffs) == 0:
                        continue
                    mean, lo, hi = paired_bootstrap(diffs, rng)
                    significant = '*' if (lo > 0) == (hi > 0) else ' '
                    print(
                        f'  {significant} {cell_a} vs {cell_b}: {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]'
                    )


def family_table(
    groups: dict,
    family: str,
    columns: list[tuple[str, str, str, int]],
    caption: str,
    label: str,
    percent: tuple[str, ...] = (),
) -> str:
    cells = [cell for fam, cell in groups if fam == family]
    models, baselines = order_cells(cells)

    vals: dict[str, dict[str, tuple[float, float] | None]] = {}
    for cell in models + baselines + ([FLOOR] if FLOOR in cells else []):
        row = {}
        for metric, _, _, _ in columns:
            rows = groups[(family, cell)]
            if metric not in rows[0]['metrics']:
                row[metric] = None
                continue
            mean, se, n = cell_mean_se(rows, metric)
            scale = 100.0 if metric in percent else 1.0
            row[metric] = (mean * scale, se * scale) if n > 0 else None
        vals[cell] = row

    best = {}
    for metric, _, better, _ in columns:
        present = [(c, vals[c][metric][0]) for c in models + baselines if vals[c][metric]]
        if present:
            best[metric] = (min if better == 'min' else max)(present, key=lambda kv: kv[1])[0]

    def fmt(cell: str, metric: str, decimals: int) -> str:
        value = vals[cell][metric]
        if value is None:
            return '--'
        mean, se = value
        body = f'{mean:.{decimals}f} \\pm {se:.{decimals}f}' if se else f'{mean:.{decimals}f}'
        return f'$\\mathbf{{{body}}}$' if best.get(metric) == cell else f'${body}$'

    def row(cell: str) -> str:
        formatted = ' & '.join(fmt(cell, m, d) for m, _, _, d in columns)
        return f'{display(cell)} & {formatted} \\\\'

    body = [row(c) for c in models]
    if baselines:
        body.append(r'\midrule')
        body.extend(row(c) for c in baselines)
    if FLOOR in cells:
        body.append(r'\midrule')
        body.append(row(FLOOR))

    return latex_table(
        colspec='l' + 'r' * len(columns),
        header='Model & ' + ' & '.join(h for _, h, _, _ in columns) + r' \\',
        body=body,
        caption=caption,
        label=label,
        generated_by='aggregate.py',
    )


def write_tables(groups: dict, rows: list[dict], latex_dir: Path) -> None:
    latex_dir.mkdir(parents=True, exist_ok=True)
    dists = family_dists(rows)

    default = [f for f, d in dists.items() if is_default_family(d)]
    strong = [
        f
        for f, d in dists.items()
        if d['num_states'] == 2 and d['coupling_scale'] == 4.0 and not is_default_family(d)
    ]
    q4 = [f for f, d in dists.items() if d['num_states'] == 4]

    tables = []
    if default:
        tables.append(
            (
                'mrf_summary.tex',
                family_table(
                    groups,
                    default[0],
                    SUMMARY_COLS,
                    caption=r'MRF reconstruction at the default family '
                    r'(mean $\pm$ SE over seeds, lower is better). The estimator floor is the '
                    r'sample-based JSD of the exact PMF against itself.',
                    label='tab:mrf-summary',
                ),
            )
        )
    if strong:
        tables.append(
            (
                'mrf_support.tex',
                family_table(
                    groups,
                    strong[0],
                    SUPPORT_COLS,
                    caption=r'Support metrics at strong coupling ($4\times$): mass placed '
                    r'outside the effective support and coverage of the true top-8 modes '
                    r'(mean $\pm$ SE over seeds).',
                    label='tab:mrf-support',
                    percent=('mass_outside', 'mode_coverage'),
                ),
            )
        )
    if q4:
        tables.append(
            (
                'mrf_q4.tex',
                family_table(
                    groups,
                    q4[0],
                    SUMMARY_COLS,
                    caption=r'MRF reconstruction with $q = 4$ classes per node at matched '
                    r'support size ($N = 6$, mean $\pm$ SE over seeds, lower is better).',
                    label='tab:mrf-q4',
                ),
            )
        )

    for filename, table in tables:
        out = latex_dir / filename
        out.write_text(table + '\n')
        print(f'Wrote {out}')


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.results))
    groups = group_by_family_cell(rows)

    summary_print(groups, args.metric)
    if args.compare:
        comparison_print(groups, args.metric, args.seed)
    if args.latex_dir:
        write_tables(groups, rows, Path(args.latex_dir))


if __name__ == '__main__':
    main()
