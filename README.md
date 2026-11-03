# embedex

`embedex` is a framework for representing higher-order objects by fixed-size embedding vectors learned over families of such objects. An encoder maps an object, seen only through a finite set of samples, to an embedding. A domain-specific decoder supplies the training signal by modeling the object conditioned on that embedding. The framework currently supports families of distributions, functions and graphs.

This repository accompanies the paper [Universal Sample-Based Embeddings of Higher-Order Objects](https://arxiv.org/abs/2610.19899), which proposes the framework. The commit tagged `2610.19899v1` marks the code used for its experiments; later commits may differ. [Neural Bayesian Filtering](https://arxiv.org/abs/2510.03614) and [Belief Embedding Tree Search](https://arxiv.org/abs/2610.15046) build on `embedex`'s distribution embeddings.

## Installation

`embedex` requires Python 3.13 and is installed with [uv](https://docs.astral.sh/uv/), which also fetches Python if needed. Clone the repository and let `uv` set up the environment:

```bash
git clone https://github.com/polar-rl/embedex.git
cd embedex
uv sync
```

Two extras add optional dependencies, alone or together (`uv sync --extra gpu --extra dev`):

- `gpu`: JAX with CUDA 12 support (Linux only).
- `dev`: the development tools (pytest, ruff and pre-commit).

### Contributing

Install with `--extra dev` and enable the pre-commit hooks by running `uv run pre-commit install`. Run the tests with `uv run pytest`, or `uv run pytest -m "not slow"` to skip the end-to-end training runs.

## Contents

Each of the three objects presented in the paper has its own package, which also contains the families used in the experiments:

- `embedex/distributions/`: embeddings of distributions that condition normalizing flows, flow matching, discrete flow matching or an autoregressive model.
- `embedex/functions/`: embeddings of functions from (x, y) samples, a decoder that evaluates the embedded functions, and operators that map one embedding to another.
- `embedex/graphs/`: embeddings of graphs, built from samples of their edges, that condition a model of the graph's Laplacian representation.

## Running the experiments

The configs of the paper's experiments live in the `configs/` directories next to the code that runs them. Run each one from the repository root, which is where checkpoints (`models/`) and figures (`figures/`) are written:

```bash
uv run python -m embedex.distributions.domains.donuts.main --config embedex/distributions/domains/donuts/configs/donuts_fm.yaml
uv run python -m embedex.functions.families.lotka_volterra --config embedex/functions/configs/lotka_volterra.yaml
uv run python -m embedex.graphs.families.random_trees --config embedex/graphs/configs/random_trees.yaml
```

`--set key.path=value` overrides any config field, for example `--set seed=1`.

## Citation

```bibtex
@misc{haluska2026universalembeddings,
    title={Universal Sample-Based Embeddings of Higher-Order Objects},
    author={Radovan Haluska and Christopher Solinas and Nathan R. Sturtevant and Martin Schmid and Michael Bowling},
    year={2026},
    eprint={2610.19899},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
}
```
