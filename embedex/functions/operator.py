from collections.abc import Callable

import equinox as eqx
import jax


class MlpOperator(eqx.Module):
    net: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        activation: Callable,
        key: jax.Array,
    ) -> None:
        self.net = eqx.nn.MLP(
            in_size, out_size, hidden_size, num_hidden_layers, activation=activation, key=key
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.net(x)


class ResidualBlock(eqx.Module):
    ln: eqx.nn.LayerNorm
    ff: eqx.nn.MLP

    def __init__(self, hidden_size: int, ff_dim: int, activation: Callable, key: jax.Array) -> None:
        self.ln = eqx.nn.LayerNorm(hidden_size)
        self.ff = eqx.nn.MLP(
            hidden_size, hidden_size, ff_dim, depth=1, activation=activation, key=key
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.ff(self.ln(x))


class ResidualOperator(eqx.Module):
    in_proj: eqx.nn.Linear
    blocks: list[ResidualBlock]
    out_ln: eqx.nn.LayerNorm
    out_proj: eqx.nn.Linear

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        num_blocks: int,
        ff_multiplier: int,
        activation: Callable,
        key: jax.Array,
    ) -> None:
        keys = jax.random.split(key, num_blocks + 2)

        self.in_proj = eqx.nn.Linear(in_size, hidden_size, key=keys[0])
        self.blocks = [
            ResidualBlock(hidden_size, hidden_size * ff_multiplier, activation, keys[i])
            for i in range(1, num_blocks + 1)
        ]
        self.out_ln = eqx.nn.LayerNorm(hidden_size)
        self.out_proj = eqx.nn.Linear(hidden_size, out_size, key=keys[-1])

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.in_proj(x)

        for block in self.blocks:
            x = block(x)

        return self.out_proj(self.out_ln(x))


class OperatorMapping(eqx.Module):
    net: eqx.Module

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.net(x)
