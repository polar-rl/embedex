from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from embedex.base_embedding_blocks import PointwiseBlock
from embedex.base_embedding_blocks import SetTransformerBlock as BaseSetTransformerBlock


class PointwiseConcatBlock(eqx.Module):
    pointwise: PointwiseBlock

    def __call__(self, xs: jax.Array, ys: jax.Array, pool: jax.Array | None) -> jax.Array:
        if pool is None:
            stage_input = jnp.concatenate([xs, ys], axis=-1)
        else:
            pool_broadcast = jnp.broadcast_to(pool, (xs.shape[0], pool.shape[0]))
            stage_input = jnp.concatenate([xs, ys, pool_broadcast], axis=-1)

        return self.pointwise(stage_input, jnp.ones(stage_input.shape[0]))


class DotProductBlock(eqx.Module):
    x_net: eqx.nn.MLP
    y_net: eqx.nn.MLP
    pool_net: eqx.nn.MLP | None

    def __call__(self, xs: jax.Array, ys: jax.Array, pool: jax.Array | None) -> jax.Array:
        zs = jax.vmap(self.x_net)(xs) * jax.vmap(self.y_net)(ys)

        if pool is not None:
            zs = zs * self.pool_net(pool)

        return zs


class SetTransformerBlock(eqx.Module):
    set_transformer: BaseSetTransformerBlock

    def __call__(self, xs: jax.Array, ys: jax.Array, pool: jax.Array | None) -> jax.Array:
        if pool is None:
            stage_input = jnp.concatenate([xs, ys], axis=-1)
        else:
            pool_broadcast = jnp.broadcast_to(pool, (xs.shape[0], pool.shape[0]))
            stage_input = jnp.concatenate([xs, ys, pool_broadcast], axis=-1)

        return self.set_transformer(stage_input, jnp.ones(stage_input.shape[0]))


class SetTransformerDotProductBlock(eqx.Module):
    dot_product: DotProductBlock
    set_transformer: BaseSetTransformerBlock

    def __call__(self, xs: jax.Array, ys: jax.Array, pool: jax.Array | None) -> jax.Array:
        stage_input = self.dot_product(xs, ys, pool)

        return self.set_transformer(stage_input, jnp.ones(stage_input.shape[0]))


class FunctionEmbedding(eqx.Module):
    """Embeds sets of (x, y) samples to create a latent representation of the underlying function.

    `embed` runs the samples through the stack of stage blocks, pooling after
    each so that block `k > 0` sees the previous round's set-level summary (a
    single block recovers a plain per-sample encoder); the final pool is the
    embedding. `eval` then reads function values back out of it via the evaluator.

    Block 0: block_0(xs, ys, None)        -> zs_0; pool_0 = mean(zs_0)
    Block k: block_k(xs, ys, pool_{k-1})  -> zs_k; pool_k = mean(zs_k)
    """

    embedding_blocks: list[eqx.Module]
    evaluator: eqx.Module
    loss_fn: Callable = eqx.field(static=True)

    def embed(self, xs: jax.Array, ys: jax.Array) -> jax.Array:
        pool = None

        for block in self.embedding_blocks:
            zs = block(xs, ys, pool)
            pool = jnp.mean(zs, axis=0)

        return pool

    def eval(self, xs: jax.Array, z: jax.Array) -> jax.Array:
        return jax.vmap(self.evaluator, in_axes=(0, None))(xs, z)

    def loss(self, xs: jax.Array, ys: jax.Array, key: jax.Array) -> jax.Array:
        perm = jax.random.permutation(key, xs.shape[0])
        xs, ys = xs[perm], ys[perm]

        xs_embed, xs_test = jnp.split(xs, 2, axis=0)
        ys_embed, ys_test = jnp.split(ys, 2, axis=0)

        z = self.embed(xs_embed, ys_embed)
        ys_pred = self.eval(xs_test, z)

        return self.loss_fn(ys_pred, ys_test)
