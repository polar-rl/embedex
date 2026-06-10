from pathlib import Path

from embedex.distributions.domains.experiment_utils import run_sweep

CELLS = [
    'st_nf_dequant',
    'st_dfm',
    'st_dfm_attn',
]


def main() -> None:
    run_sweep(
        Path(__file__).resolve().parent,
        CELLS,
        default_model_dir='models/distributions/markov_random_fields',
        default_num_iterations=50000,
    )


if __name__ == '__main__':
    main()
