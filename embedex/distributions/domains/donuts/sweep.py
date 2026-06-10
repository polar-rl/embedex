from pathlib import Path

from embedex.distributions.domains.experiment_utils import run_sweep

CELLS = [
    'pointwise_nf',
    'pointwise_fm',
    'set_transformer_nf',
    'set_transformer_fm',
]


def main() -> None:
    run_sweep(
        Path(__file__).resolve().parent,
        CELLS,
        default_model_dir='models/distributions/donuts',
        default_num_iterations=30000,
    )


if __name__ == '__main__':
    main()
