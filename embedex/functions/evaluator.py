from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp


class ConcatDecoder(eqx.Module):
    """Predicts function values for a function represented by embedding `z` at input `x`."""

    net: eqx.nn.MLP

    def __call__(self, x: jax.Array, z: jax.Array) -> jax.Array:
        return self.net(jnp.concatenate((x, z), axis=-1))


class AttentionDecoder(eqx.Module):
    """Predicts function values for a function represented by embedding `z` at input `x`.

    Projects input `x` into a query. The embedding `z` is projected into synthetic
    key and value matrices, then attended over with single-headed attention.
    """

    q_proj: eqx.nn.MLP
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    scale: float
    kv_size: int

    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        qk_size: int,
        kv_size: int,
        v_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        activation: Callable,
        key: jax.Array,
    ) -> None:
        q_key, k_key, v_key = jax.random.split(key, 3)
        self.q_proj = eqx.nn.MLP(
            input_size, qk_size, hidden_size, num_hidden_layers, activation=activation, key=q_key
        )
        self.k_proj = eqx.nn.Linear(embedding_size, kv_size * qk_size, key=k_key)
        self.v_proj = eqx.nn.Linear(embedding_size, kv_size * v_size, key=v_key)
        self.scale = 1.0 / jnp.sqrt(qk_size)
        self.kv_size = kv_size

    def __call__(self, x: jax.Array, z: jax.Array) -> jax.Array:
        q = self.q_proj(x)[jnp.newaxis, :]  # (1, qk_size)
        k = self.k_proj(z).reshape(self.kv_size, -1)  # (kv_size, qk_size)
        v = self.v_proj(z).reshape(self.kv_size, -1)  # (kv_size, v_size)
        weights = jax.nn.softmax(self.scale * q @ jnp.transpose(k), axis=1)  # (1, kv_size)

        return (weights @ v)[0]  # (v_size,)
