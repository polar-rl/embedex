import equinox as eqx
import jax
import jax.numpy as jnp


class ConditionedEigenfunctions(eqx.Module):
    net: eqx.nn.MLP

    def __call__(self, vertex: jax.Array, z: jax.Array) -> jax.Array:
        return self.net(jnp.concatenate([vertex, z]))


class ConditionedBetas(eqx.Module):
    net: eqx.nn.MLP
    num_dims: int = eqx.field(static=True)

    def __call__(self, z: jax.Array) -> jax.Array:
        n = self.num_dims
        rows, cols = jnp.tril_indices(n)

        pairs = jnp.stack([rows, cols], axis=-1).astype(jnp.float32) / n
        z_broadcast = jnp.broadcast_to(z[jnp.newaxis, :], (rows.shape[0], z.shape[0]))
        inputs = jnp.concatenate([pairs, z_broadcast], axis=1)
        betas = jax.vmap(self.net)(inputs).reshape(-1)

        return jnp.zeros((n, n)).at[rows, cols].set(betas)
