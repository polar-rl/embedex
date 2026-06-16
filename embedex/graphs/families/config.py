from dataclasses import dataclass, field

from embedex.config import load_config as _load_config
from embedex.config import parse_args as _parse_args
from embedex.graphs.config import Config, DataConfig


@dataclass
class ArmsDataConfig(DataConfig):
    num_vertices: int = 13


@dataclass
class ArmsConfig(Config):
    data: ArmsDataConfig = field(default_factory=ArmsDataConfig)
    model_dir: str = 'models/graphs/arms'
    figure_dir: str = 'figures/graphs/arms'


def load_config[T](cfg_cls: type[T], path: str) -> T:
    return _load_config(cfg_cls, path)


def parse_args[T](cfg_cls: type[T]) -> T:
    return _parse_args(cfg_cls)
