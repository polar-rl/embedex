import equinox as eqx
import jax
import jax.numpy as jnp

from embedex.functions.embedding import FunctionEmbedding
from embedex.functions.generator import Generator
from embedex.functions.operator import OperatorMapping


@eqx.filter_jit
def evaluate_mse(embedding_network: FunctionEmbedding, xs: jax.Array, ys: jax.Array) -> float:
    def _single_mse(x, y):
        x_embed, x_test = jnp.split(x, 2, axis=0)
        y_embed, y_test = jnp.split(y, 2, axis=0)

        z = embedding_network.embed(x_embed, y_embed)
        y_pred = embedding_network.eval(x_test, z)

        return _mse(y_pred, y_test)

    return jnp.mean(jax.vmap(_single_mse)(xs, ys))


@eqx.filter_jit
def evaluate_rel_l2(embedding_network: FunctionEmbedding, xs: jax.Array, ys: jax.Array) -> float:
    def _single_rel_l2(x, y):
        x_embed, x_test = jnp.split(x, 2, axis=0)
        y_embed, y_test = jnp.split(y, 2, axis=0)

        z = embedding_network.embed(x_embed, y_embed)
        y_pred = embedding_network.eval(x_test, z)

        return _rel_l2(y_pred, y_test)

    return jnp.mean(jax.vmap(_single_rel_l2)(xs, ys))


@eqx.filter_jit
def evaluate_operator_mse(
    embedding_network: FunctionEmbedding,
    operator_network: OperatorMapping,
    generator: Generator,
    xs: jax.Array,
    ys: jax.Array,
    extras: jax.Array,
) -> float:
    def _single_mse(x, y, e):
        x_embed, x_test = jnp.split(x, 2, axis=0)
        y_embed, _ = jnp.split(y, 2, axis=0)
        e_embed = jnp.split(e, 2, axis=0)[0] if x.shape[0] == e.shape[0] else e

        source_embedding = generator.operator_input(embedding_network, x_embed, y_embed, e_embed)
        pred_embedding = operator_network(source_embedding)
        y_pred = embedding_network.eval(x_test, pred_embedding)

        y_target = generator.operator_ground_truth(x, y, e)
        _, y_test = jnp.split(y_target, 2, axis=0)

        return _mse(y_pred, y_test)

    return jnp.mean(jax.vmap(_single_mse)(xs, ys, extras))


@eqx.filter_jit
def evaluate_operator_rel_l2(
    embedding_network: FunctionEmbedding,
    operator_network: OperatorMapping,
    generator: Generator,
    xs: jax.Array,
    ys: jax.Array,
    extras: jax.Array,
) -> float:
    def _single_rel_l2(x, y, e):
        x_embed, x_test = jnp.split(x, 2, axis=0)
        y_embed, _ = jnp.split(y, 2, axis=0)
        e_embed = jnp.split(e, 2, axis=0)[0] if x.shape[0] == e.shape[0] else e

        source_embedding = generator.operator_input(embedding_network, x_embed, y_embed, e_embed)
        pred_embedding = operator_network(source_embedding)
        y_pred = embedding_network.eval(x_test, pred_embedding)

        y_target = generator.operator_ground_truth(x, y, e)
        _, y_test = jnp.split(y_target, 2, axis=0)

        return _rel_l2(y_pred, y_test)

    return jnp.mean(jax.vmap(_single_rel_l2)(xs, ys, extras))


def _mse(y_pred: jax.Array, y_target: jax.Array) -> jax.Array:
    return jnp.mean((y_pred - y_target) ** 2)


def _rel_l2(y_pred: jax.Array, y_target: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum((y_pred - y_target) ** 2)) / jnp.maximum(
        jnp.sqrt(jnp.sum(y_target**2)), 1e-8
    )
