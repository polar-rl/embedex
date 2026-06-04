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
    --config embedex/distributions/domains/donuts/configs/pointwise_nf.yaml \
    --set training.num_iterations=50000 training.learning_rate=5e-4
```

Data and output paths (`model_dir`, `figure_dir`, results files) are relative to the working directory, so run commands from the repo root.

## Checkpointing

Training writes a single `.eqx` checkpoint whose first line is the full run config as JSON and whose remaining bytes are the serialized model weights. Checkpoints land at `<model_dir>/<run_name>/<num_iterations>.eqx`. Each domain sets its own `model_dir` defaults — e.g. donuts uses `models/distributions/donuts/` — and `run_name` is set per config.

To skip retraining and load a previously saved model, point `model_ckpt` at the checkpoint file:

```bash
python3 embedex/distributions/domains/donuts/main.py \
    --set model_ckpt=models/distributions/donuts/pointwise_nf/100000.eqx
```

The model architecture is reconstructed from the JSON header inside the checkpoint, so no separate config file is required at load time.

## Embedding Distributions

Each distribution domain trains a model that takes a set of samples drawn from some target distribution, embeds the set into a learned fixed-size representation, and reconstructs the distribution via a conditional generator (normalizing flow, flow matching, or discrete flow matching). The embedding network and generator are trained jointly against a known data-generating process supplied by the domain.

The shared infrastructure for embedding networks, conditional generators, and the training loop lives in `embedex/distributions/`.

### Donuts

Donuts is a continuous 2D distribution reconstruction benchmark. Each target distribution is a **donut** parameterized by its center, radius, and width.

#### Training Configs

The configs in `embedex/distributions/domains/donuts/configs/` form a 2x2 architecture grid (embedding block x conditional generator) at `embedding_dim=16` and 3 rounds. Hidden sizes are chosen so the two embedding blocks have matched parameter budgets (30,704 vs. 30,608, within 0.3%):

| Config                    | Embedding                   | Flow                      |
|---------------------------|-----------------------------|---------------------------|
| `pointwise_nf.yaml`       | Pointwise MLP (hidden 64)   | NF (8 coupling layers)    |
| `pointwise_fm.yaml`       | Pointwise MLP (hidden 64)   | Flow Matching (100 steps) |
| `set_transformer_nf.yaml` | SetTransformer (hidden 112) | NF (8 coupling layers)    |
| `set_transformer_fm.yaml` | SetTransformer (hidden 112) | Flow Matching (100 steps) |

Use any of them by running:

```bash
python3 embedex/distributions/domains/donuts/main.py \
    --config embedex/distributions/domains/donuts/configs/<config_file.yaml>
```

#### Architecture Sweep and Evaluation

```bash
# 1. Train every grid cell across seeds (resumable; skips existing checkpoints).
python3 embedex/distributions/domains/donuts/sweep.py --num-seeds 5

# 2. Score all checkpoints and the training-free baselines on one shared eval set.
python3 embedex/distributions/domains/donuts/evaluate_all.py

# 3. Summary tables (mean +/- SE over seeds) and paired bootstrap comparisons.
python3 embedex/distributions/domains/donuts/aggregate.py --compare

# 4. Metric-vs-sample-count figures from results.jsonl.
python3 embedex/distributions/domains/donuts/plots.py
```

`sweep.py` shells out to `main.py` once per (cell, seed) with `run_name=<cell>_s<seed>`; `--dry-run` prints the commands and `--set` forwards overrides to every run.

`evaluate_all.py` builds its eval set from `--eval-seed`, independent of all training seeds and identical for every run, so comparisons are paired per eval distribution. For each run and each embedding-set size in `--n-eval` it records held-out NLL (averaged over `--logprob-reps` stochastic passes), sliced Wasserstein-2 against a disjoint reference set, and method-of-moments parameter-recovery errors, writing per-distribution values to JSONL. It is resumable: existing (run, n_eval) rows are skipped.

Two training-free baselines from `baselines.py` are scored the same way on the same samples: a Gaussian KDE (Scott bandwidth; no family knowledge) and an oracle method-of-moments donut fit (the true family hard-coded; its log-density smooths the annulus edge since the raw fit's hard support makes NLL degenerate).

`aggregate.py --compare` reports all-pairs mean differences with bootstrap 95% CIs over the shared eval distributions; a starred row means the CI excludes zero.

#### Experiments

Post-training analysis experiments (visualization, interpolation, linear probes, ...) are toggled via boolean flags in the `experiments` section of the config.

```bash
python3 embedex/distributions/domains/donuts/main.py \
    --set model_ckpt=models/distributions/donuts/pointwise_nf_s0/100000.eqx \
          experiments.interpolation=true experiments.embedding_visualization=true
```

| Experiment              | Flag                      | Description                                                                   |
|-------------------------|---------------------------|-------------------------------------------------------------------------------|
| Width analysis          | `width_analysis`          | Scatter plot of true vs. generated radial std (proxy for donut width)         |
| Interpolation           | `interpolation`           | Interpolates embeddings between donut pairs                                   |
| Linear functionals      | `linear_functionals`      | Tests linear probes on embedding space                                        |
| Gaussian OOD            | `gaussian_ood`            | Evaluates on out-of-distribution Gaussian inputs                              |
| Property directions     | `property_directions`     | Finds directions in embedding space corresponding to donut properties         |
| Embedding visualization | `embedding_visualization` | t-SNE and PCA visualizations of the embedding space colored by donut property |
| Compositional           | `compositional`           | Tests compositional disentanglement via parallelogram test in embedding space |
