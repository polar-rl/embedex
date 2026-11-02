"""Shared fixtures for the function-embedding tests.

Every function family is registered once in `FAMILIES`. Tests that take the `family` fixture run
for each of them, so a new family only needs an entry here; structure specific to one family is
tested next to it in `tests/functions/families/`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from embedex.functions.builder import create_embedding_network
from embedex.functions.config import Config, DataConfig
from embedex.functions.embedding import FunctionEmbedding
from embedex.functions.families import (
    antiderivatives,
    damped_oscillator,
    derivatives,
    logistic,
    lotka_volterra,
)
from embedex.functions.families.config import (
    AntiderivativesConfig,
    DampedOscillatorConfig,
    DerivativesConfig,
    LogisticConfig,
    LotkaVolterraConfig,
)
from embedex.functions.generator import Generator


def logistic_solution(generator: Generator, t: jax.Array, params: jax.Array) -> jax.Array:
    """The closed form from `LogisticGenerator`'s docstring, y(t) = K y0 / (y0 + (K - y0) e^-rt)."""

    y0, r, K = params
    return (K * y0 / (y0 + (K - y0) * jnp.exp(-r * t)))[:, jnp.newaxis]


def trajectory(generator: Generator, t: jax.Array, params: jax.Array) -> jax.Array:
    return generator.trajectory(t, params)


@dataclass(frozen=True)
class Family:
    name: str
    config_cls: type[Config]
    make_generator: Callable[[DataConfig], Generator]
    main: Callable[[Config], None]
    figures: tuple[str, ...]  # files `main` writes to the run's figure directory
    output_size: int = 1
    data: dict[str, Any] = field(default_factory=dict)  # overrides of the data config defaults
    # The exact solution y(t) for the parameters an ODE family returns alongside its samples;
    # polynomial families have none, since their operators act on embeddings, not parameters.
    solution: Callable[[Generator, jax.Array, jax.Array], jax.Array] | None = None

    @property
    def ode(self) -> bool:
        return self.solution is not None

    def config(self) -> Config:
        cfg = self.config_cls()
        for name, value in self.data.items():
            setattr(cfg.data, name, value)

        return cfg

    def generator(self) -> Generator:
        """The family's generator. One instance is shared, so its jitted sampler compiles once
        per batch shape rather than once per test."""

        if self.name not in _GENERATORS:
            _GENERATORS[self.name] = self.make_generator(self.config().data)

        return _GENERATORS[self.name]

    def domain(self) -> tuple[float, float]:
        data = self.config().data
        return data.t_range if self.ode else data.x_range


_GENERATORS: dict[str, Generator] = {}


FAMILIES = (
    Family(
        name='derivatives',
        config_cls=DerivativesConfig,
        make_generator=lambda d: derivatives.PolynomialDerivativeGenerator(
            d.x_range, d.coeff_range, d.min_degree, d.max_degree
        ),
        main=derivatives.main,
        figures=(
            'polynomial_embeddings_by_degree.png',
            'polynomial_differential_operator_by_degree.png',
        ),
    ),
    Family(
        name='antiderivatives',
        config_cls=AntiderivativesConfig,
        make_generator=lambda d: antiderivatives.PolynomialAntiderivativeGenerator(
            d.x_range, d.coeff_range, d.min_degree, d.max_degree
        ),
        main=antiderivatives.main,
        figures=(
            'polynomial_embeddings_by_degree.png',
            'polynomial_antidifferential_operator_by_degree.png',
        ),
    ),
    Family(
        name='logistic',
        config_cls=LogisticConfig,
        make_generator=lambda d: logistic.LogisticGenerator(
            d.t_range, d.y0_range, d.r_range, d.K_range
        ),
        main=logistic.main,
        figures=('logistic_embeddings.png', 'logistic_solution_operator.png'),
        solution=logistic_solution,
    ),
    Family(
        name='damped_oscillator',
        config_cls=DampedOscillatorConfig,
        make_generator=lambda d: damped_oscillator.DampedOscillatorGenerator(
            d.t_range,
            d.x0_range,
            d.v0_range,
            d.gamma_range,
            d.omega_range,
            d.coupling_range,
            d.num_masses,
        ),
        main=damped_oscillator.main,
        figures=('damped_oscillator_embedding.png', 'damped_oscillator_solution_operator.png'),
        output_size=2,
        solution=trajectory,
    ),
    Family(
        name='coupled_oscillators',
        config_cls=DampedOscillatorConfig,
        make_generator=lambda d: damped_oscillator.DampedOscillatorGenerator(
            d.t_range,
            d.x0_range,
            d.v0_range,
            d.gamma_range,
            d.omega_range,
            d.coupling_range,
            d.num_masses,
        ),
        main=damped_oscillator.main,
        figures=('damped_oscillator_embedding.png', 'damped_oscillator_solution_operator.png'),
        output_size=6,
        data={'num_masses': 3},
        solution=trajectory,
    ),
    Family(
        name='lotka_volterra',
        config_cls=LotkaVolterraConfig,
        make_generator=lambda d: lotka_volterra.LotkaVolterraGenerator(
            d.t_range,
            d.x0_range,
            d.y0_range,
            d.alpha_range,
            d.beta_range,
            d.gamma_range,
            d.delta_range,
        ),
        main=lotka_volterra.main,
        figures=('lotka_volterra_embedding.png', 'lotka_volterra_solution_operator.png'),
        output_size=2,
        solution=trajectory,
    ),
)

ODE_FAMILIES = tuple(f for f in FAMILIES if f.ode)
POLYNOMIAL_FAMILIES = tuple(f for f in FAMILIES if not f.ode)

STAGE_BLOCKS = ('concat', 'dot_product', 'set_transformer', 'set_transformer_dot_product')


def tiny_config(
    family: Family | None = None,
    stage_block: str = 'set_transformer_dot_product',
    decoder: str = 'attention',
    out_dir: str = '.',
) -> Config:
    """A config small enough that every network builds, trains and evaluates in seconds. Its
    architecture follows the paper's (two set-transformer dot-product stages, an attention
    decoder, a relative L2 loss), only smaller."""

    cfg = family.config() if family is not None else Config()

    cfg.model.output_size = family.output_size if family is not None else 1
    cfg.model.embedding.stage_block = stage_block
    cfg.model.embedding.embedding_size = 8
    cfg.model.embedding.num_stages = 2
    cfg.model.embedding.hidden_size = 16
    cfg.model.embedding.num_hidden_layers = 1
    cfg.model.embedding.num_heads = 2
    cfg.model.embedding.num_inducing = 4
    cfg.model.evaluator.decoder = decoder
    cfg.model.evaluator.qk_size = 4
    cfg.model.evaluator.kv_size = 4
    cfg.model.evaluator.hidden_size = 16
    cfg.model.evaluator.num_hidden_layers = 1
    cfg.model.operator.hidden_size = 16
    cfg.model.operator.num_hidden_layers = 1
    cfg.model.operator.num_blocks = 1

    cfg.training.loss_fn = 'rel_l2'
    cfg.training.num_iterations = 4
    cfg.training.operator_num_iterations = 4
    cfg.training.batch_size = 4
    cfg.training.num_samples = 16

    cfg.evaluation.num_eval_functions = 4
    cfg.evaluation.num_eval_samples = 16
    cfg.evaluation.num_plot_functions = 2

    cfg.run_name = 'test'
    cfg.model_dir = f'{out_dir}/models'
    cfg.figure_dir = f'{out_dir}/figures'
    cfg.debug_mode = True

    return cfg


@pytest.fixture
def make_config(tmp_path) -> Callable[..., Config]:
    """Builds a `tiny_config` whose model and figure directories live in `tmp_path`."""

    def _make(family: Family | None = None, **kwargs) -> Config:
        return tiny_config(family, out_dir=str(tmp_path), **kwargs)

    return _make


@pytest.fixture
def families() -> dict[str, Family]:
    """Every family by name, for the tests of one family's structure."""

    return {f.name: f for f in FAMILIES}


@pytest.fixture(params=FAMILIES, ids=lambda f: f.name)
def family(request) -> Family:
    return request.param


@pytest.fixture(params=ODE_FAMILIES, ids=lambda f: f.name)
def ode_family(request) -> Family:
    return request.param


@pytest.fixture(params=POLYNOMIAL_FAMILIES, ids=lambda f: f.name)
def polynomial_family(request) -> Family:
    return request.param


@pytest.fixture(params=STAGE_BLOCKS)
def stage_block(request) -> str:
    return request.param


@pytest.fixture(params=['concat', 'attention'])
def decoder(request) -> str:
    return request.param


@pytest.fixture
def embedding_network(stage_block: str) -> FunctionEmbedding:
    """An untrained two-stage embedding of scalar functions, for each kind of stage block."""

    return create_embedding_network(tiny_config(stage_block=stage_block), jax.random.key(1))
