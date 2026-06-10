from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    stage_block: str = 'dot_product'  # 'concat' or 'dot_product'
    num_stages: int = 1
    hidden_size: int = 128
    num_hidden_layers: int = 2
    activation: str = 'relu'  # 'relu' or 'gelu'


@dataclass
class EvaluatorConfig:
    decoder: str = 'attention'  # 'concat' or 'attention'
    qk_size: int = 48
    kv_size: int = 24
    hidden_size: int = 128
    num_hidden_layers: int = 2
    activation: str = 'relu'  # 'relu' or 'gelu'


@dataclass
class OperatorConfig:
    kind: str = 'mlp'  # 'mlp' or 'residual'
    num_blocks: int = 3
    ff_multiplier: int = 2
    hidden_size: int = 128
    num_hidden_layers: int = 2
    activation: str = 'relu'  # 'relu' or 'gelu'


@dataclass
class ModelConfig:
    input_size: int = 1
    output_size: int = 1
    embedding_size: int = 64

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)


@dataclass
class TrainingConfig:
    # Optimization
    learning_rate: float = 1e-3
    lr_schedule: str = 'constant'  # 'constant' or 'cosine' (decays to 0 over num_iterations)
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0  # global-norm clip before Adam; 0 disables
    # Training loop
    num_iterations: int = 1000
    batch_size: int = 64
    num_samples: int = 256
    loss_fn: str = 'huber'  # 'mse', 'huber', or 'rel_l2'
    # Operator stage
    train_operator: bool = True  # if False, skip the operator stage entirely
    train_operator_jointly: bool = False
    operator_num_iterations: int = 1000
    operator_weight_decay: float = 0.0
    # Evaluation and checkpointing
    eval_every: int = 0
    embedding_ckpt_path: str = ''  # if set, load and skip embedding training, else train and save


@dataclass
class EvaluationConfig:
    num_eval_functions: int = 1024
    num_eval_samples: int = 256
    num_eval_plots: int = 3


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
    model_dir: str = 'models/functions'
    figure_dir: str = 'figures/functions'
    seed: int = 0
    debug_mode: bool = False
