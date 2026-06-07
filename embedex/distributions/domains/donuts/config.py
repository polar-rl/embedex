from dataclasses import dataclass, field

from embedex.config import load_config as _load_config
from embedex.config import parse_args as _parse_args
from embedex.distributions.config import ModelConfig, TrainingConfig


@dataclass
class DonutDistributionConfig:
    mean_std: float = 1.0
    radius_range: tuple[float, float] = (0.5, 2.0)
    width_range: tuple[float, float] = (0.01, 0.75)


@dataclass
class VisualizationConfig:
    grid_size: int = 512
    density_threshold: float = 0.02
    scatter_size: float = 20.0
    num_contour_levels: int = 10


@dataclass
class ExperimentConfig:
    width_analysis: bool = True
    width_analysis_num_distributions: int = 128
    width_analysis_num_samples: int = 512

    interpolation: bool = True
    interpolation_num_steps: int = 5
    interpolation_num_samples: int = 512

    linear_functionals: bool = True
    linear_functionals_num_samples: int = 1024

    gaussian_ood: bool = True
    gaussian_ood_num_samples: int = 512

    property_directions: bool = True
    property_directions_num_pairs: int = 50
    property_directions_num_samples: int = 256
    property_directions_num_alpha_steps: int = 7

    embedding_visualization: bool = True
    embedding_visualization_num_distributions: int = 512
    embedding_visualization_num_samples: int = 256

    compositional: bool = True
    compositional_num_samples: int = 512

    property_reconstruction: bool = True
    property_reconstruction_num_samples: int = 128
    property_reconstruction_num_generated: int = 512
    property_reconstruction_num_steps: int = 5


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    distribution: DonutDistributionConfig = field(default_factory=DonutDistributionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)

    model_dir: str = 'models/distributions/donuts'
    figure_dir: str = 'figures/distributions/donuts'
    run_name: str = 'baseline'
    model_ckpt: str | None = None
    seed: int = 0


def load_config(path: str) -> Config:
    return _load_config(Config, path)


def parse_args() -> Config:
    return _parse_args(Config)
