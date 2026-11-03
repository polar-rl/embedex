from dataclasses import dataclass, field

from embedex.config import load_config as _load_config
from embedex.config import parse_args as _parse_args
from embedex.distributions.config import Config, DataConfig


@dataclass
class DonutsDataConfig(DataConfig):
    mean_std: float = 1.0
    radius_range: tuple[float, float] = (0.5, 2.0)
    width_range: tuple[float, float] = (0.01, 0.75)


@dataclass
class DonutsEvaluationConfig:
    width_analysis: bool = True
    width_analysis_num_distributions: int = 128
    width_analysis_num_samples: int = 512

    interpolation: bool = True
    interpolation_num_steps: int = 6
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

    parallelogram: bool = True
    parallelogram_num_samples: int = 512


@dataclass
class DonutsVisualizationConfig:
    grid_size: int = 512
    density_threshold: float = 0.02
    scatter_size: float = 20.0
    num_contour_levels: int = 10


@dataclass
class MRFDataConfig(DataConfig):
    num_spins: int = 12
    num_states: int = 2
    coupling_scale: float = 1.0
    bias_scale: float = 0.5
    # Graph topology: 'full' | 'random_edge_fixed_prob' | 'random_edge_random_prob'
    graph_type: str = 'random_edge_fixed_prob'
    edge_prob: float = 0.5
    edge_prob_min: float = 0.1
    edge_prob_max: float = 0.9


@dataclass
class MRFVisualizationConfig:
    num_embedding_samples: int = 128
    num_generated_samples: int = 1024
    grid_num_distributions: int = 8
    sample_strip_num: int = 50


@dataclass
class DonutsConfig(Config):
    data: DonutsDataConfig = field(default_factory=DonutsDataConfig)
    evaluation: DonutsEvaluationConfig = field(default_factory=DonutsEvaluationConfig)
    visualization: DonutsVisualizationConfig = field(default_factory=DonutsVisualizationConfig)

    model_dir: str = 'models/distributions/donuts'
    figure_dir: str = 'figures/distributions/donuts'


@dataclass
class MRFConfig(Config):
    data: MRFDataConfig = field(default_factory=MRFDataConfig)
    visualization: MRFVisualizationConfig = field(default_factory=MRFVisualizationConfig)

    model_dir: str = 'models/distributions/markov_random_fields'
    figure_dir: str = 'figures/distributions/markov_random_fields'


def load_config[T](cfg_cls: type[T], path: str) -> T:
    return _load_config(cfg_cls, path)


def parse_args[T](cfg_cls: type[T]) -> T:
    return _parse_args(cfg_cls)
