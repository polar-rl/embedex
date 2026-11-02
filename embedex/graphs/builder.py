import equinox as eqx
import jax
import optax

from embedex.base_embedding_blocks import PointwiseBlock as BasePointwiseBlock
from embedex.base_embedding_blocks import SetTransformerBlock as BaseSetTransformerBlock
from embedex.config import build_dataclass
from embedex.graphs.allo import ConditionedBetas, ConditionedEigenfunctions
from embedex.graphs.config import Config, EmbeddingConfig, ModelConfig
from embedex.graphs.embedding import GraphEmbedding, PointwiseBlock, SetTransformerBlock
from embedex.graphs.reconstructor import MediansNetwork, TransitionNetwork


def create_embedding_network(cfg: Config, key: jax.Array) -> GraphEmbedding:
    blocks_key, eigenfunctions_key, betas_key = jax.random.split(key, 3)
    embedding_blocks = create_embedding_blocks(
        cfg.model.embedding, cfg.model.input_size, blocks_key
    )

    return GraphEmbedding(
        embedding_blocks,
        ConditionedEigenfunctions(
            eqx.nn.MLP(
                cfg.model.input_size + cfg.model.embedding.embedding_size,
                cfg.model.allo.num_dims,
                cfg.model.allo.eigfn_hidden_size,
                cfg.model.allo.eigfn_num_hidden_layers,
                key=eigenfunctions_key,
            )
        ),
        ConditionedBetas(
            eqx.nn.MLP(
                2 + cfg.model.embedding.embedding_size,
                1,
                cfg.model.allo.betas_hidden_size,
                cfg.model.allo.betas_num_hidden_layers,
                key=betas_key,
            ),
            cfg.model.allo.num_dims,
        ),
        cfg.model.allo.penalty_coeff,
        cfg.model.allo.dual_reg,
    )


def build_from_config(cfg: dict, key: jax.Array) -> GraphEmbedding:
    return create_embedding_network(Config(model=build_dataclass(ModelConfig, cfg['model'])), key)


def create_embedding_blocks(
    embedding: EmbeddingConfig, input_size: int, key: jax.Array
) -> list[eqx.Module]:
    blocks, block_keys = [], jax.random.split(key, embedding.num_stages)

    for i, block_key in enumerate(block_keys):
        in_size = input_size + (embedding.embedding_size if i > 0 else 0)

        if embedding.stage_block == 'pointwise':
            blocks.append(
                PointwiseBlock(
                    BasePointwiseBlock(
                        eqx.nn.MLP(
                            in_size,
                            embedding.embedding_size,
                            embedding.hidden_size,
                            embedding.num_hidden_layers,
                            key=block_key,
                        )
                    )
                )
            )
        elif embedding.stage_block == 'set_transformer':
            blocks.append(
                SetTransformerBlock(
                    BaseSetTransformerBlock(
                        in_size,
                        embedding.embedding_size,
                        embedding.num_heads,
                        embedding.num_inducing,
                        embedding.hidden_size,
                        block_key,
                    )
                )
            )
        else:
            raise ValueError(f'Unknown stage_block: {embedding.stage_block!r}')

    return blocks


def create_medians_network(cfg: Config, key: jax.Array) -> MediansNetwork:
    return MediansNetwork(
        eqx.nn.MLP(
            cfg.model.input_size + cfg.model.embedding.embedding_size,
            1,
            cfg.model.reconstructor.hidden_size,
            cfg.model.reconstructor.num_hidden_layers,
            key=key,
        )
    )


def create_transition_network(cfg: Config, key: jax.Array) -> TransitionNetwork:
    return TransitionNetwork(
        eqx.nn.MLP(
            1 + cfg.model.embedding.embedding_size,
            1,
            cfg.model.reconstructor.hidden_size,
            cfg.model.reconstructor.num_hidden_layers,
            key=key,
        )
    )


def create_optimizer(cfg: Config, num_steps: int) -> optax.GradientTransformation:
    if cfg.training.lr_schedule == 'constant':
        lr = cfg.training.learning_rate
    elif cfg.training.lr_schedule == 'cosine':
        lr = optax.cosine_decay_schedule(
            init_value=cfg.training.learning_rate,
            decay_steps=num_steps,
            alpha=0.0,
        )
    else:
        raise ValueError(f'Unknown learning rate schedule: {cfg.training.lr_schedule!r}')

    opt = optax.optimistic_adam_v2(lr)

    if cfg.training.grad_clip_norm > 0:
        opt = optax.chain(optax.clip_by_global_norm(cfg.training.grad_clip_norm), opt)

    return opt
