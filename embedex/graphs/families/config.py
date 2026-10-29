from dataclasses import dataclass, field

from embedex.config import load_config as _load_config
from embedex.config import parse_args as _parse_args
from embedex.graphs.config import Config, DataConfig


@dataclass
class ArmsDataConfig(DataConfig):
    num_vertices: int = 13


@dataclass
class RandomTreesDataConfig(DataConfig):
    num_vertices: int = 10


@dataclass
class ScaleFreeDataConfig(DataConfig):
    num_vertices: int = 10
    m: int = 2


@dataclass
class GridWorldDataConfig(DataConfig):
    grid_size: int = 4
    num_walls: int = 3


@dataclass
class ArmsConfig(Config):
    data: ArmsDataConfig = field(default_factory=ArmsDataConfig)
    model_dir: str = 'models/graphs/arms'
    figure_dir: str = 'figures/graphs/arms'


@dataclass
class RandomTreesConfig(Config):
    data: RandomTreesDataConfig = field(default_factory=RandomTreesDataConfig)
    model_dir: str = 'models/graphs/random_trees'
    figure_dir: str = 'figures/graphs/random_trees'


@dataclass
class ScaleFreeConfig(Config):
    data: ScaleFreeDataConfig = field(default_factory=ScaleFreeDataConfig)
    model_dir: str = 'models/graphs/scale_free'
    figure_dir: str = 'figures/graphs/scale_free'


@dataclass
class GridWorldConfig(Config):
    data: GridWorldDataConfig = field(default_factory=GridWorldDataConfig)
    model_dir: str = 'models/graphs/gridworld'
    figure_dir: str = 'figures/graphs/gridworld'


def load_config[T](cfg_cls: type[T], path: str) -> T:
    return _load_config(cfg_cls, path)


def parse_args[T](cfg_cls: type[T]) -> T:
    return _parse_args(cfg_cls)
