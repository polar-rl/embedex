from dataclasses import dataclass, field

from embedex.config import load_config as _load_config
from embedex.config import parse_args as _parse_args
from embedex.functions.config import Config, DataConfig


@dataclass
class PolynomialConfig(DataConfig):
    x_range: tuple[float, float] = (-4.0, 4.0)
    coeff_range: tuple[float, float] = (-5.0, 5.0)
    max_degree: int = 3


@dataclass
class DerivativesConfig(Config):
    data: PolynomialConfig = field(default_factory=PolynomialConfig)

    model_dir: str = 'models/functions/derivatives'
    figure_dir: str = 'figures/functions/derivatives'


@dataclass
class AntiderivativesConfig(Config):
    data: PolynomialConfig = field(default_factory=PolynomialConfig)

    model_dir: str = 'models/functions/antiderivatives'
    figure_dir: str = 'figures/functions/antiderivatives'


def load_config[T](cfg_cls: type[T], path: str) -> T:
    return _load_config(cfg_cls, path)


def parse_args[T](cfg_cls: type[T]) -> T:
    return _parse_args(cfg_cls)
