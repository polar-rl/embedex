from dataclasses import dataclass, field

from embedex.config import load_config as _load_config
from embedex.config import parse_args as _parse_args
from embedex.functions.config import Config, DataConfig


@dataclass
class PolynomialConfig(DataConfig):
    x_range: tuple[float, float] = (-4.0, 4.0)
    coeff_range: tuple[float, float] = (-5.0, 5.0)
    min_degree: int = 0
    max_degree: int = 3


@dataclass
class LogisticDataConfig(DataConfig):
    t_range: tuple[float, float] = (0.0, 5.0)
    y0_range: tuple[float, float] = (0.1, 2.0)
    r_range: tuple[float, float] = (0.5, 3.0)
    K_range: tuple[float, float] = (1.0, 5.0)


@dataclass
class DampedOscillatorDataConfig(DataConfig):
    t_range: tuple[float, float] = (0.0, 10.0)
    x0_range: tuple[float, float] = (-2.0, 2.0)
    v0_range: tuple[float, float] = (-2.0, 2.0)
    gamma_range: tuple[float, float] = (0.1, 1.0)
    omega_range: tuple[float, float] = (0.5, 3.0)
    coupling_range: tuple[float, float] = (0.5, 1.5)
    num_masses: int = 1


@dataclass
class LotkaVolterraDataConfig(DataConfig):
    t_range: tuple[float, float] = (0.0, 10.0)
    x0_range: tuple[float, float] = (0.5, 2.0)
    y0_range: tuple[float, float] = (0.5, 2.0)
    alpha_range: tuple[float, float] = (0.8, 1.5)  # Prey growth
    beta_range: tuple[float, float] = (0.8, 1.5)  # Predation
    gamma_range: tuple[float, float] = (0.8, 1.5)  # Predator death
    delta_range: tuple[float, float] = (0.8, 1.5)  # Predator growth from predation


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


@dataclass
class LogisticConfig(Config):
    data: LogisticDataConfig = field(default_factory=LogisticDataConfig)

    model_dir: str = 'models/functions/logistic'
    figure_dir: str = 'figures/functions/logistic'


@dataclass
class DampedOscillatorConfig(Config):
    data: DampedOscillatorDataConfig = field(default_factory=DampedOscillatorDataConfig)

    model_dir: str = 'models/functions/damped_oscillator'
    figure_dir: str = 'figures/functions/damped_oscillator'


@dataclass
class LotkaVolterraConfig(Config):
    data: LotkaVolterraDataConfig = field(default_factory=LotkaVolterraDataConfig)

    model_dir: str = 'models/functions/lotka_volterra'
    figure_dir: str = 'figures/functions/lotka_volterra'


def load_config[T](cfg_cls: type[T], path: str) -> T:
    return _load_config(cfg_cls, path)


def parse_args[T](cfg_cls: type[T]) -> T:
    return _parse_args(cfg_cls)
