from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from embedex.config import build_dataclass
from embedex.functions.config import (
    Config,
    EmbeddingConfig,
    EvaluatorConfig,
    ModelConfig,
    OperatorConfig,
)
from embedex.functions.embedding import ConcatStageBlock, DotProductStageBlock, FunctionEmbedding
from embedex.functions.evaluator import AttentionDecoder, ConcatDecoder
from embedex.functions.generator import Generator
from embedex.functions.operator import MlpOperator, OperatorMapping, ResidualOperator


def create_embedding_network(cfg: Config, key: jax.Array) -> FunctionEmbedding:
    encoder_key, decoder_key = jax.random.split(key, 2)

    encoder_activation = _get_activation(cfg.model.embedding.activation)
    decoder_activation = _get_activation(cfg.model.evaluator.activation)

    embedding_blocks = create_embedding_blocks(
        cfg.model, cfg.model.embedding, encoder_activation, encoder_key
    )
    evaluator = create_evaluator(cfg.model, cfg.model.evaluator, decoder_activation, decoder_key)

    return FunctionEmbedding(embedding_blocks, evaluator, _get_loss(cfg.training.loss_fn))


def build_from_config(cfg: dict, key: jax.Array) -> FunctionEmbedding:
    return create_embedding_network(build_dataclass(Config, cfg), key)


def create_embedding_blocks(
    model: ModelConfig, embedding: EmbeddingConfig, activation: Callable, key: jax.Array
) -> list[eqx.Module]:
    stage_keys = jax.random.split(key, embedding.num_stages)
    blocks: list[eqx.Module] = []

    for i, stage_key in enumerate(stage_keys):
        if embedding.stage_block == 'concat':
            in_size = model.input_size + model.output_size + (model.embedding_size if i > 0 else 0)
            blocks.append(
                ConcatStageBlock(
                    eqx.nn.MLP(
                        in_size,
                        model.embedding_size,
                        embedding.hidden_size,
                        embedding.num_hidden_layers,
                        activation=activation,
                        key=stage_key,
                    ),
                )
            )
        elif embedding.stage_block == 'dot_product':
            x_key, y_key, pool_key = jax.random.split(stage_key, 3)

            pool_net = None
            if i > 0:
                pool_net = eqx.nn.MLP(
                    model.embedding_size,
                    model.embedding_size,
                    embedding.hidden_size,
                    embedding.num_hidden_layers,
                    activation=activation,
                    key=pool_key,
                )

            blocks.append(
                DotProductStageBlock(
                    eqx.nn.MLP(
                        model.input_size,
                        model.embedding_size,
                        embedding.hidden_size,
                        embedding.num_hidden_layers,
                        activation=activation,
                        key=x_key,
                    ),
                    eqx.nn.MLP(
                        model.output_size,
                        model.embedding_size,
                        embedding.hidden_size,
                        embedding.num_hidden_layers,
                        activation=activation,
                        key=y_key,
                    ),
                    pool_net,
                )
            )
        else:
            raise ValueError(f'Unknown stage_block: {embedding.stage_block!r}')

    return blocks


def create_evaluator(
    model: ModelConfig, evaluator: EvaluatorConfig, activation: Callable, key: jax.Array
) -> eqx.Module:
    if evaluator.decoder == 'concat':
        return ConcatDecoder(
            net=eqx.nn.MLP(
                model.input_size + model.embedding_size,
                model.output_size,
                evaluator.hidden_size,
                evaluator.num_hidden_layers,
                activation=activation,
                key=key,
            ),
        )

    if evaluator.decoder == 'attention':
        return AttentionDecoder(
            model.input_size,
            model.embedding_size,
            evaluator.qk_size,
            evaluator.kv_size,
            model.output_size,
            evaluator.hidden_size,
            evaluator.num_hidden_layers,
            activation,
            key,
        )

    raise ValueError(f'Unknown decoder: {evaluator.decoder!r}')


def create_operator_network(cfg: Config, generator: Generator, key: jax.Array) -> OperatorMapping:
    in_size = generator.operator_input_size(cfg)
    operator_network = create_operator(cfg.model.operator, in_size, cfg.model.embedding_size, key)

    return OperatorMapping(operator_network)


def create_operator(
    operator: OperatorConfig, in_size: int, out_size: int, key: jax.Array
) -> eqx.Module:
    activation = _get_activation(operator.activation)

    if operator.kind == 'residual':
        return ResidualOperator(
            in_size,
            out_size,
            operator.hidden_size,
            operator.num_blocks,
            operator.ff_multiplier,
            activation,
            key,
        )
    if operator.kind == 'mlp':
        return MlpOperator(
            in_size,
            out_size,
            operator.hidden_size,
            operator.num_hidden_layers,
            activation,
            key,
        )

    raise ValueError(f'Unknown operator: {operator.kind!r}')


def create_optimizer(
    cfg: Config, weight_decay: float, num_steps: int
) -> optax.GradientTransformation:
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

    base = optax.adamw(lr, weight_decay=weight_decay)

    if cfg.training.grad_clip_norm > 0:
        return optax.chain(optax.clip_by_global_norm(cfg.training.grad_clip_norm), base)

    return base


def _get_activation(name: str) -> Callable:
    if name == 'relu':
        return jax.nn.relu
    if name == 'gelu':
        return jax.nn.gelu

    raise ValueError(f'Unknown activation: {name!r}')


def _get_loss(name: str) -> Callable:
    if name == 'huber':

        def _huber_loss(y_pred, y_target):
            return jnp.mean(optax.losses.huber_loss(y_pred, y_target))

        return _huber_loss
    if name == 'mse':

        def _mse_loss(y_pred, y_target):
            return jnp.mean(optax.losses.squared_error(y_pred, y_target))

        return _mse_loss
    if name == 'rel_l2':

        def _rel_l2_loss(y_pred, y_target):
            numerator = jnp.sqrt(jnp.sum((y_pred - y_target) ** 2))
            denominator = jnp.maximum(jnp.sqrt(jnp.sum(y_target**2)), 1e-8)
            return numerator / denominator

        return _rel_l2_loss
    if name == 'softmax_ce':

        def _softmax_ce_loss(y_pred, y_target):
            log_p = jax.nn.log_softmax(y_pred, axis=0)
            q = jax.nn.softmax(y_target, axis=0)
            return -jnp.sum(q * log_p)

        return _softmax_ce_loss

    raise ValueError(f'Unknown loss function: {name!r}')
