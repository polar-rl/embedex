import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

CELLS = [
    'pointwise_nf',
    'pointwise_fm',
    'set_transformer_nf',
    'set_transformer_fm',
]

DOMAIN_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = 'models/distributions/donuts'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', nargs='*', default=CELLS, choices=CELLS)
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--set',
        dest='overrides',
        nargs='*',
        default=[],
        help='Overrides in the form key.path=value, forwarded to every run',
    )
    return parser.parse_args()


def final_ckpt_path(cell: str, run_name: str, overrides: list[str]) -> Path:
    with open(DOMAIN_DIR / 'configs' / f'{cell}.yaml') as f:
        config = yaml.safe_load(f)

    num_iterations = config.get('training', {}).get('num_iterations', 30000)
    model_dir = config.get('model_dir', DEFAULT_MODEL_DIR)

    for override in overrides:
        key, _, value = override.partition('=')
        if key == 'training.num_iterations':
            num_iterations = int(value)
        elif key == 'model_dir':
            model_dir = value

    filename = f'{num_iterations:0{len(str(num_iterations))}}.eqx'
    return Path(model_dir) / run_name / filename


def main() -> None:
    args = parse_args()

    runs = [(cell, seed) for cell in args.cells for seed in range(args.num_seeds)]
    completed, skipped, failed = [], [], []

    for i, (cell, seed) in enumerate(runs, 1):
        run_name = f'{cell}_s{seed}'
        ckpt = final_ckpt_path(cell, run_name, args.overrides)

        if ckpt.exists():
            print(f'[{i}/{len(runs)}] {run_name}: checkpoint exists, skipping')
            skipped.append(run_name)
            continue

        command = [
            sys.executable,
            str(DOMAIN_DIR / 'main.py'),
            '--config',
            str(DOMAIN_DIR / 'configs' / f'{cell}.yaml'),
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


if __name__ == '__main__':
    main()
