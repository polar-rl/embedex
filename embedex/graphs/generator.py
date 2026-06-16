from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp


class Generator(ABC):
    @abstractmethod
    def _sample_one(self, num_samples: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    @eqx.filter_jit
    def _sample_edges_from_adj(self, adj: jax.Array, num_samples: int, key: jax.Array) -> jax.Array:
        """Sample edges from an adjacency matrix.

        First vertex is sampled with probability proportional to its degree,
        then a neighbor is sampled uniformly from the vertex's neighbors.
        """

        n = adj.shape[0]
        deg = jnp.sum(adj, axis=1)

        key, v_key = jax.random.split(key, 2)
        vs = jax.random.choice(v_key, n, (num_samples,), p=deg / jnp.sum(deg))

        keys = jax.random.split(key, num_samples)
        vs_adj = jax.vmap(lambda p, k: jax.random.choice(k, n, p=p))(adj[vs], keys)

        return jnp.eye(n)[jnp.stack((vs, vs_adj), axis=-1)]

    @eqx.filter_jit
    def sample(
        self, batch_size: int, num_samples: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return jax.vmap(self._sample_one, in_axes=(None, 0))(
            num_samples, jax.random.split(key, batch_size)
        )
