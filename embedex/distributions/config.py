from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EmbeddingConfig:
    block_type: Literal['pointwise', 'set_transformer'] = 'pointwise'
    embedding_dim: int = 8
    hidden_size: int = 64
    num_layers: int = 3
    num_rounds: int = 3
    # Set Transformer specific
    num_heads: int = 4
    num_inducing: int = 16
    pool_type: Literal['mean', 'attention'] = 'mean'
    pool_num_heads: int = 4
    pool_num_queries: int = 1
    relational: bool = False
    relational_mode: Literal['none', 'learned', 'raw_agree', 'raw_edge'] | None = None
    rel_d_e: int = 16
    rel_num_layers: int = 3
    rel_num_heads: int = 4
    rel_node_dim: int = 0
    num_classes: int = 2

    @property
    def rel_mode(self) -> Literal['none', 'learned', 'raw_agree', 'raw_edge']:
        if self.relational_mode is None:
            return 'learned' if self.relational else 'none'
        if self.relational and self.relational_mode != 'learned':
            raise ValueError(
                f'`relational=True` contradicts `relational_mode={self.relational_mode!r}`; set one or the other.'
            )
        return self.relational_mode


@dataclass
class GeneratorConfig:
    flow_type: Literal['nf', 'fm', 'dfm', 'ar'] = 'nf'
    conditioning: Literal['concat', 'film', 'hyper'] = 'concat'
    # Normalizing Flow
    nf_dequant_type: Literal['learned', 'uniform'] = 'learned'
    nf_condition_dequant: bool = True
    nf_dequant_hidden_size: int = 64
    nf_dequant_num_layers: int = 2
    nf_hidden_size: int = 32
    nf_num_layers: int = 3
    nf_num_coupling_layers: int = 8
    nf_num_flow_params: int = 5
    # Flow Matching
    fm_time_embed_dim: int = 16
    fm_hidden_size: int = 128
    fm_num_layers: int = 4
    fm_num_steps: int = 100
    fm_sigma_min: float = 1e-4
    # Discrete Flow Matching
    dfm_num_classes: int = 10
    dfm_hidden_size: int = 128
    dfm_num_layers: int = 4
    dfm_time_embed_dim: int = 16
    dfm_num_steps: int = 100
    dfm_denoiser: Literal['mlp', 'attention'] = 'mlp'
    dfm_attn_token_dim: int = 64
    dfm_attn_num_heads: int = 4
    dfm_attn_num_layers: int = 4
    dfm_attn_ff_multiplier: int = 4
    dfm_attn_num_cond_tokens: int = 0  # 0 = auto-infer from conditioning_dim
    # Autoregressive (Causal Transformer)
    ar_num_classes: int = 2
    ar_token_dim: int = 48
    ar_num_heads: int = 4
    ar_num_layers: int = 5
    ar_ff_multiplier: int = 4


@dataclass
class ModelConfig:
    input_dim: int = 2
    likelihood_dim: int = 0  # per-sample likelihood features handed to embed() beside the samples
    likelihood_index: tuple[int, ...] | None = None
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)


@dataclass
class TrainingConfig:
    # Optimization
    learning_rate: float = 1e-3
    end_learning_rate: float = 1e-5
    warmup_steps: int = 0
    clip_grad_norm: float | None = None  # optax.clip_by_global_norm threshold (None = off)
    # Training loop
    num_iterations: int = 30000
    batch_size: int = 32
    num_samples: int = 128
    # Logging and checkpointing
    log_interval: int = 100
    checkpoint_interval: int = 1000
    # Misc
    seed: int = 0


@dataclass
class DataConfig:
    pass


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    run_name: str = 'baseline'
    model_dir: str = 'models/distributions'
    figure_dir: str = 'figures/distributions'
    model_ckpt: str | None = None
    seed: int = 0
