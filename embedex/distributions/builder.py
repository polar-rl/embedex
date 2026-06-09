import distrax
import equinox as eqx
import jax

from embedex.config import build_dataclass
from embedex.distributions.conditioning import (
    ConcatMLP,
    CrossAttentionBlock,
    FiLMMLP,
    HyperMLP,
)
from embedex.distributions.config import ModelConfig
from embedex.distributions.embedding import (
    DistributionEmbedding,
    FlowEmbedding,
    PointwiseBlock,
    SetTransformerBlock,
)
from embedex.distributions.flows.dequantization import Dequantize
from embedex.distributions.flows.discrete_flow_matching import (
    AttentionDenoiserNet,
    DiscreteDenoiserNet,
    DiscreteFlowMatchingModel,
)
from embedex.distributions.flows.flow_matching import FlowMatchingModel, VelocityNet
from embedex.distributions.flows.flowee import Flow
from embedex.distributions.flows.normalizing_flow import (
    Coupling,
    NormalizingFlowModel,
    ParameterizedNLSq,
    create_mask,
)


def create_model(config: ModelConfig, dequantize: bool, key: jax.Array) -> DistributionEmbedding:
    embedding_key, flow_key = jax.random.split(key, 2)

    embedding_blocks = create_embedding_blocks(config, embedding_key)
    generator = create_generative_network(
        config, config.embedding.embedding_dim, dequantize, flow_key
    )

    return FlowEmbedding(embedding_blocks, generator)


def build_from_config(cfg: dict, dequantize: bool, key: jax.Array) -> DistributionEmbedding:
    return create_model(build_dataclass(ModelConfig, cfg['model']), dequantize=dequantize, key=key)


def create_embedding_blocks(
    config: ModelConfig,
    key: jax.Array,
) -> list[PointwiseBlock | SetTransformerBlock]:
    emb = config.embedding
    num_rounds = emb.num_rounds

    blocks = []
    for i in range(num_rounds):
        key, block_key = jax.random.split(key, 2)
        in_size = config.input_dim if i == 0 else config.input_dim + emb.embedding_dim

        if emb.block_type == 'set_transformer':
            blocks.append(
                SetTransformerBlock(
                    in_dim=in_size,
                    out_dim=emb.embedding_dim,
                    num_heads=emb.num_heads,
                    num_inducing=emb.num_inducing,
                    ff_dim=emb.hidden_size,
                    key=block_key,
                )
            )
        else:
            blocks.append(
                PointwiseBlock(
                    eqx.nn.MLP(
                        in_size=in_size,
                        out_size=emb.embedding_dim,
                        width_size=emb.hidden_size,
                        depth=emb.num_layers,
                        key=block_key,
                    )
                )
            )

    return blocks


def create_generative_network(
    config: ModelConfig,
    conditioning_dim: int,
    dequantize: bool,
    key: jax.Array,
) -> Flow:
    gen = config.generator

    if gen.flow_type == 'dfm':
        return create_dfm_model(config, conditioning_dim=conditioning_dim, key=key)
    elif gen.flow_type == 'fm':
        if dequantize:
            raise NotImplementedError('Flow matching + dequant not implemented. Use DFM.')
        return create_fm_model(config, conditioning_dim=conditioning_dim, key=key)
    else:
        return create_nf_model(
            config, conditioning_dim=conditioning_dim, dequantize=dequantize, key=key
        )


def create_fm_model(
    config: ModelConfig,
    conditioning_dim: int,
    key: jax.Array,
) -> FlowMatchingModel:
    gen = config.generator
    velocity_net = create_velocity_net(
        input_dim=config.input_dim,
        conditioning_dim=conditioning_dim,
        hidden_size=gen.fm_hidden_size,
        num_layers=gen.fm_num_layers,
        time_embed_dim=gen.fm_time_embed_dim,
        conditioning=gen.conditioning,
        key=key,
    )

    model = FlowMatchingModel(
        velocity_net=velocity_net,
        input_dim=config.input_dim,
        num_steps=gen.fm_num_steps,
        sigma_min=gen.fm_sigma_min,
    )
    model.add_prior(distrax.Normal(0.0, 1.0), (config.input_dim,))

    return model


def create_velocity_net(
    input_dim: int,
    conditioning_dim: int,
    hidden_size: int,
    num_layers: int,
    time_embed_dim: int,
    conditioning: str,
    key: jax.Array,
) -> VelocityNet:
    net = create_conditioned_mlp(
        input_dim=input_dim + time_embed_dim,
        conditioning_dim=conditioning_dim,
        out_size=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        conditioning=conditioning,
        key=key,
    )

    return VelocityNet(time_embed_dim=time_embed_dim, net=net)


def create_dfm_model(
    config: ModelConfig,
    conditioning_dim: int,
    key: jax.Array,
) -> DiscreteFlowMatchingModel:
    gen = config.generator
    num_variables = config.input_dim
    num_classes = gen.dfm_num_classes

    if gen.dfm_denoiser == 'attention':
        denoiser_net = create_attention_denoiser_net(
            num_classes=num_classes,
            num_variables=num_variables,
            conditioning_dim=conditioning_dim,
            token_dim=gen.dfm_attn_token_dim,
            num_heads=gen.dfm_attn_num_heads,
            num_layers=gen.dfm_attn_num_layers,
            time_embed_dim=gen.dfm_time_embed_dim,
            ff_multiplier=gen.dfm_attn_ff_multiplier,
            num_cond_tokens=gen.dfm_attn_num_cond_tokens,
            key=key,
        )
    else:
        denoiser_net = create_discrete_denoiser_net(
            num_classes=num_classes,
            num_variables=num_variables,
            conditioning_dim=conditioning_dim,
            hidden_size=gen.dfm_hidden_size,
            num_layers=gen.dfm_num_layers,
            time_embed_dim=gen.dfm_time_embed_dim,
            conditioning=gen.conditioning,
            key=key,
        )

    return DiscreteFlowMatchingModel(
        denoiser_net=denoiser_net,
        num_classes=num_classes,
        num_variables=num_variables,
        num_steps=gen.dfm_num_steps,
    )


def create_discrete_denoiser_net(
    num_classes: int,
    num_variables: int,
    conditioning_dim: int,
    hidden_size: int,
    num_layers: int,
    time_embed_dim: int,
    conditioning: str,
    key: jax.Array,
) -> DiscreteDenoiserNet:
    simplex_dim = num_variables * num_classes

    net = create_conditioned_mlp(
        input_dim=simplex_dim + time_embed_dim,
        conditioning_dim=conditioning_dim,
        out_size=simplex_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        conditioning=conditioning,
        key=key,
    )

    return DiscreteDenoiserNet(
        time_embed_dim=time_embed_dim,
        num_classes=num_classes,
        num_variables=num_variables,
        net=net,
    )


def create_attention_denoiser_net(
    num_classes: int,
    num_variables: int,
    conditioning_dim: int,
    token_dim: int,
    num_heads: int,
    num_layers: int,
    time_embed_dim: int,
    ff_multiplier: int,
    num_cond_tokens: int,
    key: jax.Array,
) -> AttentionDenoiserNet:
    keys = jax.random.split(key, num_layers + 8)

    PARAMS_PER_ENTITY = 7
    if num_cond_tokens <= 0:
        if conditioning_dim >= PARAMS_PER_ENTITY:
            num_cond_tokens = conditioning_dim // PARAMS_PER_ENTITY
        else:
            num_cond_tokens = max(1, conditioning_dim)
    cond_token_raw_dim = (
        conditioning_dim // num_cond_tokens if num_cond_tokens > 0 else conditioning_dim
    )

    class_embeddings = jax.random.normal(keys[0], (num_classes, token_dim)) * 0.02
    variable_embeddings = jax.random.normal(keys[1], (num_variables, token_dim)) * 0.02
    time_proj = eqx.nn.Linear(time_embed_dim, token_dim, key=keys[2])

    cond_token_proj = eqx.nn.Linear(cond_token_raw_dim, token_dim, key=keys[3])
    cond_time_proj = eqx.nn.Linear(time_embed_dim, token_dim, key=keys[4])
    cond_global_proj = eqx.nn.Linear(conditioning_dim, token_dim, key=keys[5 + num_layers])
    cond_pos_embeddings = (
        jax.random.normal(keys[6 + num_layers], (num_cond_tokens, token_dim)) * 0.02
    )

    blocks = tuple(
        CrossAttentionBlock(
            token_dim=token_dim,
            cond_token_dim=token_dim,
            num_heads=num_heads,
            key=keys[5 + i],
            ff_multiplier=ff_multiplier,
        )
        for i in range(num_layers)
    )

    final_norm = eqx.nn.LayerNorm(token_dim)
    output_proj = eqx.nn.Linear(token_dim, num_classes, key=keys[7 + num_layers])

    return AttentionDenoiserNet(
        time_embed_dim=time_embed_dim,
        num_classes=num_classes,
        num_variables=num_variables,
        token_dim=token_dim,
        num_cond_tokens=num_cond_tokens,
        cond_token_raw_dim=cond_token_raw_dim,
        class_embeddings=class_embeddings,
        variable_embeddings=variable_embeddings,
        time_proj=time_proj,
        cond_token_proj=cond_token_proj,
        cond_time_proj=cond_time_proj,
        cond_global_proj=cond_global_proj,
        cond_pos_embeddings=cond_pos_embeddings,
        blocks=blocks,
        final_norm=final_norm,
        output_proj=output_proj,
    )


def create_nf_model(
    config: ModelConfig,
    conditioning_dim: int,
    dequantize: bool,
    key: jax.Array,
) -> NormalizingFlowModel:
    gen = config.generator

    flow_key, dq_key = jax.random.split(key, 2)
    keys = jax.random.split(flow_key, gen.nf_num_coupling_layers)

    layers: list[Flow] = []

    if dequantize:
        dequant_cond_dim = conditioning_dim if gen.nf_condition_dequant else 0
        layers.append(
            create_dequantization_layer(
                config.input_dim,
                conditioning_dim=dequant_cond_dim,
                hidden_size=gen.nf_dequant_hidden_size,
                num_layers=gen.nf_dequant_num_layers,
                key=dq_key,
            )
        )

    for layer_key in keys:
        param_net = create_conditioned_mlp(
            config.input_dim,
            conditioning_dim,
            config.input_dim * gen.nf_num_flow_params,
            gen.nf_hidden_size,
            gen.nf_num_layers,
            conditioning=gen.conditioning,
            key=layer_key,
        )

        layers.append(
            Coupling(
                create_mask((config.input_dim,), (1,)),
                param_net,
                ParameterizedNLSq((config.input_dim,)),
                dual=True,
            )
        )

    flow_network = NormalizingFlowModel(layers)
    flow_network.add_prior(distrax.Normal(0.0, 1.0), (config.input_dim,))

    return flow_network


def create_dequantization_layer(
    input_dim: int,
    conditioning_dim: int,
    hidden_size: int,
    num_layers: int,
    key: jax.Array,
) -> Dequantize:
    net = eqx.nn.MLP(
        in_size=input_dim + conditioning_dim,
        out_size=2 * input_dim,
        width_size=hidden_size,
        depth=num_layers,
        key=key,
    )

    return Dequantize(net=net, input_dim=input_dim, conditioning_dim=conditioning_dim)


def create_conditioned_mlp(
    input_dim: int,
    conditioning_dim: int,
    out_size: int,
    hidden_size: int,
    num_layers: int,
    conditioning: str,
    key: jax.Array,
) -> ConcatMLP | FiLMMLP | HyperMLP:
    if conditioning == 'film':
        return FiLMMLP(
            input_dim,
            conditioning_dim,
            out_size,
            hidden_size,
            num_layers,
            key=key,
        )
    elif conditioning == 'hyper':
        return HyperMLP(
            input_dim,
            conditioning_dim,
            out_size,
            hidden_size,
            num_layers,
            key=key,
        )
    else:
        return ConcatMLP(
            input_dim + conditioning_dim,
            out_size,
            hidden_size,
            num_layers,
            key=key,
        )
