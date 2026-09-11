"""Config dataclasses for distribution embedding models and training."""

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
    # Node-relational stream (TWO_AXIS): outer-product sample pool -> node-axis edge-biased
    # attention -> per-node N-dim Jhat row, delivered per node. The node-token width is N (not a
    # tunable), so there is no d_model knob. num_classes is set from distribution.num_states.
    #
    # relational_mode selects which stream is built. 'learned' is the full stream above;
    # 'raw_agree' and 'raw_edge' are the ablation arms that ask whether the edge-biased
    # attention earns its keep over the raw pairwise statistic. All three emit out_dim = N^2
    # and are delivered per variable, so conditioning width is identical across arms.
    # relational: bool is the deprecated alias the learned stream was configured with before
    # the arms existed; every fusion3 checkpoint carries it, so it stays readable. Resolve
    # both through rel_mode rather than reading either field directly.
    relational: bool = False
    relational_mode: Literal['none', 'learned', 'raw_agree', 'raw_edge'] | None = None
    rel_d_e: int = 16
    rel_num_layers: int = 3
    rel_num_heads: int = 4
    num_classes: int = 2

    @property
    def rel_mode(self) -> Literal['none', 'learned', 'raw_agree', 'raw_edge']:
        """The stream to build, resolving the deprecated `relational` alias.

        A config predating the arms sets only `relational`; a config naming an arm sets only
        `relational_mode`. Both set and disagreeing is a mistake worth raising on, because
        `build_dataclass` ignores unknown keys and would otherwise build the wrong model in
        silence.
        """
        if self.relational_mode is None:
            return 'learned' if self.relational else 'none'
        if self.relational and self.relational_mode != 'learned':
            raise ValueError(
                f'relational=True contradicts relational_mode={self.relational_mode!r}; '
                'set one or the other.'
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
    # Autoregressive (causal transformer)
    ar_num_classes: int = 2
    ar_token_dim: int = 48
    ar_num_heads: int = 4
    ar_num_layers: int = 5
    ar_ff_multiplier: int = 4


@dataclass
class ModelConfig:
    input_dim: int = 2
    output_dim: int = 2
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
