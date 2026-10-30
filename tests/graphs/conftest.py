"""Shared fixtures for the graph-embedding tests.

Every graph family is registered once in `FAMILIES`. Tests that take the `family` fixture run
for each of them, so a new family only needs an entry here; structure specific to one family is
tested next to it in `tests/graphs/families/`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax
import pytest

from embedex.graphs import builder
from embedex.graphs.builder import create_embedding_network
from embedex.graphs.config import Config
from embedex.graphs.embedding import GraphEmbedding
from embedex.graphs.families import arms, gridworld, random_trees, scale_free
from embedex.graphs.families.config import (
    ArmsConfig,
    GridWorldConfig,
    RandomTreesConfig,
    ScaleFreeConfig,
)
from embedex.graphs.generator import Generator


@dataclass(frozen=True)
class Family:
    name: str
    config_cls: type[Config]
    data: dict[str, Any]  # small data config, shared by `generator` and `config`
    make_generator: Callable[..., Generator]
    num_vertices: int
    weighted: bool
    create_reconstructor: Callable[[Config, jax.Array], eqx.Module]
    main: Callable[[Config], None]
    figures: tuple[str, ...] = field(default=())  # files `main` writes besides the geometry plots

    def generator(self) -> Generator:
        """The family's generator. One instance is shared, so its jitted sampler compiles once
        per batch shape rather than once per test."""

        if self.name not in _GENERATORS:
            _GENERATORS[self.name] = self.make_generator(**self.data)

        return _GENERATORS[self.name]


_GENERATORS: dict[str, Generator] = {}


BINARY_FIGURES = (
    'ground_truth_reconstruction.pdf',
    'reconstruction_vertex_degrees.pdf',
    'reconstruction_average_degree.pdf',
    'reconstruction_medians.pdf',
)
WEIGHTED_FIGURES = ('ground_truth_weights.pdf', 'reconstruction_weights.pdf')

FAMILIES = (
    Family(
        name='random_trees',
        config_cls=RandomTreesConfig,
        data={'num_vertices': 8},
        make_generator=random_trees.RandomTreesGenerator,
        num_vertices=8,
        weighted=False,
        create_reconstructor=builder.create_medians_network,
        main=random_trees.main,
        figures=BINARY_FIGURES,
    ),
    Family(
        name='scale_free',
        config_cls=ScaleFreeConfig,
        data={'num_vertices': 8, 'm': 2},
        make_generator=scale_free.ScaleFreeGenerator,
        num_vertices=8,
        weighted=False,
        create_reconstructor=builder.create_medians_network,
        main=scale_free.main,
        figures=BINARY_FIGURES,
    ),
    Family(
        name='arms',
        config_cls=ArmsConfig,
        data={'num_vertices': 7},
        make_generator=arms.ArmsGenerator,
        num_vertices=7,
        weighted=False,
        create_reconstructor=builder.create_medians_network,
        main=arms.main,
        figures=BINARY_FIGURES,
    ),
    Family(
        name='gridworld',
        config_cls=GridWorldConfig,
        data={'grid_size': 3, 'num_walls': 2},
        make_generator=gridworld.GridWorldGenerator,
        num_vertices=9,
        weighted=True,
        create_reconstructor=builder.create_transition_network,
        main=gridworld.main,
        figures=WEIGHTED_FIGURES,
    ),
)

BINARY_FAMILIES = tuple(f for f in FAMILIES if not f.weighted)
WEIGHTED_FAMILIES = tuple(f for f in FAMILIES if f.weighted)


def tiny_config(
    family: Family | None = None,
    num_vertices: int = 8,
    stage_block: str = 'set_transformer',
    out_dir: str = '.',
) -> Config:
    """A config small enough that every network builds, trains and evaluates in seconds."""

    cfg = family.config_cls() if family is not None else Config()
    if family is not None:
        for name, value in family.data.items():
            setattr(cfg.data, name, value)
        num_vertices = family.num_vertices

    cfg.model.input_size = num_vertices
    cfg.model.embedding.stage_block = stage_block
    cfg.model.embedding.embedding_size = 8
    cfg.model.embedding.hidden_size = 16
    cfg.model.embedding.num_hidden_layers = 1
    cfg.model.embedding.num_heads = 2
    cfg.model.embedding.num_inducing = 4
    cfg.model.allo.num_dims = 4
    cfg.model.allo.eigfn_hidden_size = 16
    cfg.model.allo.eigfn_num_hidden_layers = 1
    cfg.model.allo.betas_hidden_size = 8
    cfg.model.allo.betas_num_hidden_layers = 1
    cfg.model.reconstructor.hidden_size = 8
    cfg.model.reconstructor.num_hidden_layers = 1

    cfg.training.num_iterations = 4
    cfg.training.chunk_size = 2
    cfg.training.batch_size = 4
    cfg.training.num_samples = 16
    cfg.training.reconstructor_num_iterations = 4

    cfg.evaluation.num_eval_graphs = 4
    cfg.evaluation.num_eval_samples = 16
    cfg.evaluation.num_plot_graphs = 2

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


@pytest.fixture(params=FAMILIES, ids=lambda f: f.name)
def family(request) -> Family:
    return request.param


@pytest.fixture(params=BINARY_FAMILIES, ids=lambda f: f.name)
def binary_family(request) -> Family:
    return request.param


@pytest.fixture(params=WEIGHTED_FAMILIES, ids=lambda f: f.name)
def weighted_family(request) -> Family:
    return request.param


@pytest.fixture(params=['pointwise', 'set_transformer'])
def stage_block(request) -> str:
    return request.param


@pytest.fixture
def embedding_network(stage_block: str) -> GraphEmbedding:
    """An untrained embedding over 8-vertex graphs, for each kind of stage block."""

    return create_embedding_network(tiny_config(stage_block=stage_block), jax.random.key(1))
