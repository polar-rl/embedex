# Embedex

Embedex provides a framework for learning embeddings of higher-order objects (such as distributions and functions) along with operations on them like sampling and density estimation.

## Installation

`embedex` currently requires Python 3.13 or 3.14. Clone the repository and let `uv` set up the environment:

```bash
git clone git@github.com:polar-rl/embedex.git
cd embedex
uv sync  # optionally add `--extra gpu` for GPU support
```

If you plan to contribute, install with `--extra dev` and enable the pre-commit hooks by running `uv run pre-commit install`.
