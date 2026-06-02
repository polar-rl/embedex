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

## Configuration

Each domain ships with a set of YAML configs under its `configs/` directory. Domain entry points expose two CLI flags:

- `--config <path/to/file.yaml>` loads a YAML config.
- `--set key.path=value [key.path=value ...]` overrides individual fields; multiple `key.path=value` pairs can be passed to a single `--set`.

```bash
python3 embedex/distributions/domains/donuts/main.py \
    --config embedex/distributions/domains/donuts/configs/baseline.yaml \
    --set training.num_iterations=50000 training.learning_rate=5e-4
```

## Checkpointing

Training writes a single `.eqx` checkpoint whose first line is the full run config as JSON and whose remaining bytes are the serialized model weights. Checkpoints land at `<model_dir>/<run_name>/<num_iterations>.eqx`. Each domain sets its own `model_dir` defaults — e.g. donuts uses `models/distributions/donuts/` — and `run_name` is set per config.

To skip retraining and load a previously saved model, point `model_ckpt` at the checkpoint file:

```bash
python3 embedex/distributions/domains/donuts/main.py \
    --set model_ckpt=models/distributions/donuts/baseline/30000.eqx
```

The model architecture is reconstructed from the JSON header inside the checkpoint, so no separate config file is required at load time.

## Embedding Distributions

Each distribution domain trains a model that takes a set of samples drawn from some target distribution, embeds the set into a learned fixed-size representation, and reconstructs the distribution via a conditional generator (normalizing flow, flow matching, or discrete flow matching). The embedding network and generator are trained jointly against a known data-generating process supplied by the domain.

The shared infrastructure for embedding networks, conditional generators, and the training loop lives in `embedex/distributions/`.

### Donuts

Donuts is a continuous 2D distribution reconstruction benchmark. Each target distribution is a **donut** parameterized by its center, radius, and width.

#### Training Configs

All training configs live in `embedex/distributions/domains/donuts/configs/`:

| Config                  | Embedding                 | Flow                      | Description               |
|-------------------------|---------------------------|---------------------------|---------------------------|
| `baseline.yaml`         | Pointwise MLP (1 round)   | NF (8 coupling layers)    | Single-round baseline     |
| `multi_round.yaml`      | Pointwise MLP (3 rounds)  | NF (8 coupling layers)    | Multi-round embedding     |
| `set_transformer.yaml`  | SetTransformer (3 rounds) | NF (8 coupling layers)    | Attention-based embedding |
| `flow_matching_st.yaml` | SetTransformer (3 rounds) | Flow Matching (100 steps) | FM with SetTransformer    |

Use any of them by running:

```bash
python3 embedex/distributions/domains/donuts/main.py \
    --config embedex/distributions/domains/donuts/configs/<config_file.yaml>
```

#### Experiments

Post-training analysis experiments (visualization, interpolation, linear probes, ...) are toggled via boolean flags in the `experiments` section of the config. The default configs enable every experiment listed in the table below. It is possible to toggle individual experiments via overrides, e.g. `--set experiments.interpolation=false`.

| Experiment              | Flag                      | Description                                                                   |
|-------------------------|---------------------------|-------------------------------------------------------------------------------|
| Width analysis          | `width_analysis`          | Scatter plot of true vs. generated radial std (proxy for donut width)         |
| Interpolation           | `interpolation`           | Interpolates embeddings between donut pairs                                   |
| Linear functionals      | `linear_functionals`      | Tests linear probes on embedding space                                        |
| Gaussian OOD            | `gaussian_ood`            | Evaluates on out-of-distribution Gaussian inputs                              |
| Property directions     | `property_directions`     | Finds directions in embedding space corresponding to donut properties         |
| Embedding visualization | `embedding_visualization` | t-SNE and PCA visualizations of the embedding space colored by donut property |
| Compositional           | `compositional`           | Tests compositional disentanglement via parallelogram test in embedding space |
