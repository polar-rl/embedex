"""Shared fixtures for the distribution-embedding tests.

Every domain is registered once in `DOMAINS`. Tests that take the `domain` fixture run for each
of them, so a new domain only needs an entry here; structure specific to one domain is tested
next to it in `tests/distributions/domains/`.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import pytest

from embedex.distributions.builder import create_model
from embedex.distributions.config import Config, DataConfig, ModelConfig
from embedex.distributions.domains.config import DonutsConfig, MRFConfig
from embedex.distributions.domains.donuts import donut
from embedex.distributions.domains.donuts import main as donuts_main
from embedex.distributions.domains.markov_random_fields import main as mrf_main
from embedex.distributions.domains.markov_random_fields import mrf
from embedex.distributions.embedding import FlowEmbedding


def sample_donuts(data: DataConfig, batch_size: int, num_samples: int, key: jax.Array):
    donut_key, sample_key = jax.random.split(key)
    donuts = donut.random_donuts(
        batch_size, data.mean_std, data.radius_range, data.width_range, donut_key
    )
    return donut.sample_batch(donuts, num_samples, sample_key)


def sample_mrfs(data: DataConfig, batch_size: int, num_samples: int, key: jax.Array):
    mrf_key, sample_key = jax.random.split(key)
    mrfs = mrf.random_mrfs(
        batch_size,
        data.num_spins,
        data.num_states,
        data.coupling_scale,
        data.bias_scale,
        data.edge_prob,
        mrf_key,
        graph_type=data.graph_type,
        edge_prob_min=data.edge_prob_min,
        edge_prob_max=data.edge_prob_max,
    )
    return mrf.sample_batch(mrfs, num_samples, sample_key)


@dataclass(frozen=True)
class Domain:
    name: str
    config_cls: type[Config]
    sample: Callable[[DataConfig, int, int, jax.Array], jax.Array]  # (data, batch, n, key)
    main: Callable[[Config], None]
    flow_type: str  # the paper's generative model for the domain
    input_dim: int
    figures: tuple[str, ...]  # files `main` writes when it evaluates a trained checkpoint
    num_classes: int | None = None  # categorical domains; None means continuous samples
    # Overrides of the config defaults that keep evaluation and plotting tiny.
    data: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    visualization: dict[str, Any] = field(default_factory=dict)

    def config(self) -> Config:
        cfg = self.config_cls()
        for section in ('data', 'evaluation', 'visualization'):
            for name, value in getattr(self, section).items():
                setattr(getattr(cfg, section), name, value)

        return cfg


# The parallelogram experiment's file names: every pair and triple of the donut's properties.
DONUT_PAIRS = [
    '_'.join(p) for p in itertools.combinations(('pos_x', 'pos_y', 'radius', 'width'), 2)
]
DONUT_TRIPLES = [
    '_'.join(p) for p in itertools.combinations(('pos_x', 'pos_y', 'radius', 'width'), 3)
]

DOMAINS = (
    Domain(
        name='donuts',
        config_cls=DonutsConfig,
        sample=sample_donuts,
        main=donuts_main.main,
        flow_type='fm',
        input_dim=2,
        figures=(
            'donuts_paper.pdf',
            'width_analysis.pdf',
            'interpolation.pdf',
            'linear_functionals.pdf',
            'gaussian_ood_single.pdf',
            'gaussian_ood_mixtures.pdf',
            *(f'property_direction_{p}.pdf' for p in ('radius', 'width', 'pos_x', 'pos_y')),
            'embedding_pca.pdf',
            *(f'embedding_pca_pairs_{p}.pdf' for p in ('mean_x', 'mean_y', 'radius', 'width')),
            *(f'parallelogram_{pair}.pdf' for pair in DONUT_PAIRS),
            *(f'parallelogram_triple_{triple}.pdf' for triple in DONUT_TRIPLES),
        ),
        evaluation={
            'width_analysis_num_distributions': 4,
            'width_analysis_num_samples': 16,
            'interpolation_num_steps': 2,
            'interpolation_num_samples': 16,
            'linear_functionals_num_samples': 16,
            'gaussian_ood_num_samples': 16,
            'property_directions_num_pairs': 2,
            'property_directions_num_samples': 16,
            'property_directions_num_alpha_steps': 2,
            'embedding_visualization_num_distributions': 8,
            'embedding_visualization_num_samples': 16,
            'parallelogram_num_samples': 16,
        },
        visualization={'grid_size': 8},
    ),
    Domain(
        name='markov_random_fields',
        config_cls=MRFConfig,
        sample=sample_mrfs,
        main=mrf_main.main,
        flow_type='ar',
        input_dim=4,
        num_classes=2,
        figures=(
            'mrf_pairwise_marginals.pdf',
            'mrf_class_marginals.pdf',
            'mrf_coupling_graph.pdf',
            'mrf_samples.pdf',
            'mrf_reconstruction_grid.pdf',
        ),
        data={'num_spins': 4, 'num_states': 2},
        visualization={
            'num_embedding_samples': 16,
            'num_generated_samples': 32,
            'grid_num_distributions': 2,
            'sample_strip_num': 8,
        },
    ),
)

# Generative models over continuous samples, and over categorical ones (where the normalizing
# flow dequantizes its input).
CONTINUOUS_FLOWS = ('nf', 'fm')
DISCRETE_FLOWS = ('nf', 'dfm', 'ar')


def tiny_model_config(
    block_type: str = 'set_transformer',
    flow_type: str = 'nf',
    input_dim: int = 2,
    num_classes: int = 2,
) -> ModelConfig:
    """A model small enough to build, train and evaluate in seconds, for any block and flow."""

    cfg = ModelConfig(input_dim=input_dim)

    emb = cfg.embedding
    emb.block_type = block_type
    emb.embedding_dim = 8
    emb.hidden_size = 16
    emb.num_layers = 1
    emb.num_rounds = 2
    emb.num_heads = 2
    emb.num_inducing = 4
    emb.pool_num_heads = 2
    emb.num_classes = num_classes
    emb.rel_d_e = 4
    emb.rel_num_layers = 1
    emb.rel_num_heads = 2

    gen = cfg.generator
    gen.flow_type = flow_type
    gen.nf_hidden_size = 8
    gen.nf_num_layers = 1
    gen.nf_num_coupling_layers = 2
    gen.nf_dequant_hidden_size = 8
    gen.nf_dequant_num_layers = 1
    gen.fm_time_embed_dim = 4
    gen.fm_hidden_size = 16
    gen.fm_num_layers = 1
    gen.fm_num_steps = 4
    gen.dfm_num_classes = num_classes
    gen.dfm_hidden_size = 16
    gen.dfm_num_layers = 1
    gen.dfm_time_embed_dim = 4
    gen.dfm_num_steps = 4
    gen.dfm_attn_token_dim = 8
    gen.dfm_attn_num_heads = 2
    gen.dfm_attn_num_layers = 1
    gen.dfm_attn_ff_multiplier = 2
    gen.ar_num_classes = num_classes
    gen.ar_token_dim = 8
    gen.ar_num_heads = 2
    gen.ar_num_layers = 1
    gen.ar_ff_multiplier = 2

    return cfg


def tiny_config(domain: Domain, out_dir: str = '.') -> Config:
    """A domain's config with a `tiny_model_config` of the paper's architecture (set-transformer
    rounds and the domain's generative model) and a few iterations of training."""

    cfg = domain.config()
    cfg.model = tiny_model_config(
        flow_type=domain.flow_type,
        input_dim=domain.input_dim,
        num_classes=domain.num_classes or 2,
    )

    cfg.training.num_iterations = 4
    cfg.training.batch_size = 4
    cfg.training.num_samples = 16
    cfg.training.log_interval = 1

    cfg.run_name = 'test'
    cfg.model_dir = f'{out_dir}/models'
    cfg.figure_dir = f'{out_dir}/figures'

    return cfg


@pytest.fixture
def make_config(tmp_path) -> Callable[[Domain], Config]:
    """Builds a `tiny_config` whose model and figure directories live in `tmp_path`."""

    return lambda domain: tiny_config(domain, out_dir=str(tmp_path))


@pytest.fixture
def make_model_config() -> Callable[..., ModelConfig]:
    """Builds a `tiny_model_config` from its arguments, plus any other field of the model
    (`likelihood_dim`, ...), its embedding (`relational_mode`, `pool_type`, ...) or its generator
    (`conditioning`, ...)."""

    def _make(
        block_type: str = 'set_transformer',
        flow_type: str = 'nf',
        input_dim: int = 2,
        num_classes: int = 2,
        **fields: Any,
    ) -> ModelConfig:
        cfg = tiny_model_config(block_type, flow_type, input_dim, num_classes)
        for name, value in fields.items():
            section = next(s for s in (cfg, cfg.embedding, cfg.generator) if hasattr(s, name))
            setattr(section, name, value)

        return cfg

    return _make


@pytest.fixture
def make_model(make_model_config) -> Callable[..., FlowEmbedding]:
    """Builds an untrained model from the arguments of `make_model_config`."""

    def _make(dequantize: bool = False, key: jax.Array | None = None, **kwargs) -> FlowEmbedding:
        key = jax.random.key(1) if key is None else key
        return create_model(make_model_config(**kwargs), dequantize, key)

    return _make


@pytest.fixture
def domains() -> dict[str, Domain]:
    """Every domain by name, for the tests of one domain's structure."""

    return {d.name: d for d in DOMAINS}


@pytest.fixture(params=DOMAINS, ids=lambda d: d.name)
def domain(request) -> Domain:
    return request.param


@pytest.fixture(params=['pointwise', 'set_transformer'])
def block_type(request) -> str:
    return request.param


@pytest.fixture(params=CONTINUOUS_FLOWS)
def continuous_flow(request) -> str:
    return request.param


@pytest.fixture(params=DISCRETE_FLOWS)
def discrete_flow(request) -> str:
    return request.param


@pytest.fixture(params=['mean', 'attention'])
def pool_type(request) -> str:
    return request.param


@pytest.fixture
def embedding_network(make_model, block_type: str, pool_type: str) -> FlowEmbedding:
    """An untrained embedding of 2D samples with a normalizing-flow generator, for each kind of
    embedding block and readout."""

    return make_model(block_type=block_type, pool_type=pool_type)
