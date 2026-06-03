from dataclasses import dataclass, field

from embedex.config import load_config as _load_config
from embedex.config import parse_args as _parse_args
from embedex.distributions.config import ModelConfig, TrainingConfig


@dataclass
class MRFDistributionConfig:
    num_spins: int = 12
    coupling_scale: float = 1.0
    bias_scale: float = 0.5


@dataclass
class EvaluationConfig:
    num_distributions: int = 256
    num_embedding_samples: int = 128
    num_logprob_evals: int = 8


@dataclass
class VisualizationConfig:
    num_embedding_samples: int = 128
    num_generated_samples: int = 1024
    grid_num_distributions: int = 8
    sample_strip_num: int = 50


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    distribution: MRFDistributionConfig = field(default_factory=MRFDistributionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    model_dir: str = 'models/distributions/markov_random_fields'
    figure_dir: str = 'figures/distributions/markov_random_fields'
    run_name: str = 'dfm'
    model_ckpt: str | None = None
    seed: int = 0


def load_config(path: str) -> Config:
    return _load_config(Config, path)


def parse_args() -> Config:
    return _parse_args(Config)
