import equinox as eqx
import jax
import jax.numpy as jnp

from embedex.base_embedding_blocks import PointwiseBlock as BasePointwiseBlock
from embedex.base_embedding_blocks import SetTransformerBlock as BaseSetTransformerBlock
from embedex.graphs.allo import ConditionedBetas, ConditionedEigenfunctions


class PointwiseBlock(eqx.Module):
    pointwise: BasePointwiseBlock

    def __call__(self, edges: jax.Array, pool: jax.Array | None) -> jax.Array:
        if pool is None:
            block_input = edges
        else:
            pool_broadcast = jnp.broadcast_to(pool, (edges.shape[0], pool.shape[0]))
            block_input = jnp.concatenate([edges, pool_broadcast], axis=-1)

        return self.pointwise(block_input, jnp.ones(block_input.shape[0]))


class SetTransformerBlock(eqx.Module):
    set_transformer: BaseSetTransformerBlock

    def __call__(self, edges: jax.Array, pool: jax.Array | None) -> jax.Array:
        if pool is None:
            block_input = edges
        else:
            pool_broadcast = jnp.broadcast_to(pool, (edges.shape[0], pool.shape[0]))
            block_input = jnp.concatenate([edges, pool_broadcast], axis=-1)

        return self.set_transformer(block_input, jnp.ones(block_input.shape[0]))


class GraphEmbedding(eqx.Module):
    embedding_blocks: list[eqx.Module]
    eigenfunctions: ConditionedEigenfunctions
    betas: ConditionedBetas
    penalty_coeff: float = eqx.field(static=True)
    dual_reg: float = eqx.field(static=True)

    def embed(self, edges: jax.Array) -> jax.Array:
        # Symmetric per-edge encoding: each edge token is e_u + e_v
        edges = jnp.sum(edges.reshape(edges.shape[0], 2, -1), axis=1)

        pool = None
        for block in self.embedding_blocks:
            zs = block(edges, pool)
            pool = jnp.mean(zs, axis=0)

        return pool

    def eigenvalues(self, z: jax.Array) -> jax.Array:
        return -jnp.diag(self.betas(z)) / 2

    def representation(self, vertices: jax.Array, z: jax.Array) -> jax.Array:
        uvs = jax.vmap(lambda v: self.eigenfunctions(v, z))(vertices)
        return uvs[:, 1:] / jnp.sqrt(self.eigenvalues(z))[jnp.newaxis, 1:]

    def laplacian(self, vertices: jax.Array, z: jax.Array) -> jax.Array:
        uvs = jax.vmap(lambda v: self.eigenfunctions(v, z))(vertices)
        return (uvs @ jnp.diag(self.eigenvalues(z)) @ jnp.transpose(uvs)) / vertices.shape[0]

    def distance_matrix(self, vertices: jax.Array, z: jax.Array) -> jax.Array:
        representation = self.representation(vertices, z)
        dist = representation @ jnp.transpose(representation)

        return jnp.diag(dist)[:, jnp.newaxis] + jnp.diag(dist)[jnp.newaxis, :] - 2 * dist

    def loss(self, edges: jax.Array, key: jax.Array) -> jax.Array:
        perm = jax.random.permutation(key, edges.shape[0])
        edges = edges[perm]

        edges_train, edges_test = jnp.split(edges, 2, axis=0)
        z = self.embed(edges_train)
        uvs_test = jax.vmap(jax.vmap(lambda v: self.eigenfunctions(v, z)))(edges_test)

        uvs = uvs_test[:, 0]
        uvs_adj = uvs_test[:, 1]
        uvs_stop = jax.lax.stop_gradient(uvs)

        betas = self.betas(jax.lax.stop_gradient(z))

        num_dims = uvs.shape[1]
        recon = jnp.sum(jnp.mean((uvs - uvs_adj) ** 2, axis=0))

        gram = (jnp.transpose(uvs) @ uvs_stop) / uvs.shape[0]
        residual = gram - jnp.eye(num_dims)

        tril = jnp.tril(jnp.ones((num_dims, num_dims)))
        betas_term = jnp.sum(betas * residual * tril)
        prox_term = self.penalty_coeff * jnp.sum(residual**2 * tril)
        reg_term = self.dual_reg * jnp.mean(betas**2)

        return recon + betas_term + prox_term - reg_term
