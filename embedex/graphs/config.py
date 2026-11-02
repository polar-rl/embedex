from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    stage_block: str = 'set_transformer'  # 'pointwise' or 'set_transformer'
    embedding_size: int = 64
    num_stages: int = 2
    hidden_size: int = 64
    num_hidden_layers: int = 2
    # Set Transformer specific
    num_heads: int = 4
    num_inducing: int = 8


@dataclass
class AlloConfig:
    num_dims: int = 8
    eigfn_hidden_size: int = 128
    eigfn_num_hidden_layers: int = 2
    betas_hidden_size: int = 32
    betas_num_hidden_layers: int = 2
    penalty_coeff: float = 10.0
    dual_reg: float = 1e-3


@dataclass
class ReconstructorConfig:
    hidden_size: int = 64
    num_hidden_layers: int = 3


@dataclass
class ModelConfig:
    input_size: int = 10  # one-hot vertex dimension

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    allo: AlloConfig = field(default_factory=AlloConfig)
    reconstructor: ReconstructorConfig = field(default_factory=ReconstructorConfig)


@dataclass
class TrainingConfig:
    # Optimization
    learning_rate: float = 1e-3
    lr_schedule: str = 'cosine'  # 'constant' or 'cosine'
    grad_clip_norm: float = 0.0
    # Training loop
    num_iterations: int = 100_000
    chunk_size: int = 256
    batch_size: int = 32
    num_samples: int = 200
    # Reconstructor stage
    reconstructor_learning_rate: float = 1e-3
    reconstructor_num_iterations: int = 100_000
    # Checkpointing
    embedding_ckpt_path: str = ''


@dataclass
class EvaluationConfig:
    num_eval_graphs: int = 256
    num_eval_samples: int = 200
    num_plot_graphs: int = 4


@dataclass
class DataConfig:
    pass


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    project: str = 'belief-projects/default'
    run_name: str = 'baseline'
    model_dir: str = 'models/graphs'
    figure_dir: str = 'figures/graphs'
    seed: int = 0
    debug_mode: bool = False
