import equinox as eqx
import jax
import jax.numpy as jnp

from embedex.base_embedding_blocks import PointwiseBlock, SetTransformerBlock  # noqa: F401
from embedex.distributions.flows import flowee


class DistributionEmbedding(eqx.Module):
    """Embeds samples to create a latent representation of their distribution.

    Uses a multi-round architecture where each block processes the original
    samples concatenated with the previous round's pooled context. This allows
    later rounds to compute features relative to the set-level summary (e.g.
    spread around the estimated center).

    Each block maps (N, D_in) -> (N, D_out). The block type determines how samples interact.

    Block 0: block_0(xs)               -> zs_0, pool_0 = weighted_mean(zs_0)
    Block k: block_k(xs || pool_{k-1}) -> zs_k, pool_k = weighted_mean(zs_k)
    """

    embedding_blocks: list[eqx.Module]
    generator: flowee.Flow

    def generate(
        self,
        z: jax.Array,
        num_samples: int,
        params: jax.Array | None = None,
        *,
        key: jax.Array,
    ) -> jax.Array:
        keys = jax.random.split(key, num_samples)
        return jax.vmap(self.generator.sample, in_axes=(None, None, 0))(z, params, keys)

    def embed(self, xs: jax.Array, weights: jax.Array) -> tuple[jax.Array, jax.Array]:
        n = xs.shape[0]
        w = weights.reshape((-1, 1)) / jnp.sum(weights)

        pool = None
        for block in self.embedding_blocks:
            if pool is None:
                block_input = xs
            else:
                pool_broadcast = jnp.broadcast_to(pool, (n, pool.shape[0]))
                block_input = jnp.concatenate([xs, pool_broadcast], axis=-1)

            zs = block(block_input, weights)
            pool = jnp.sum(zs * w, axis=0)

        return pool, jnp.linalg.norm(pool)

    def loss(
        self,
        xs: jax.Array,
        weights: jax.Array,
        shuffle: bool = True,
        params: jax.Array | None = None,
        *,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if shuffle:
            key, shuffle_key = jax.random.split(key, 2)
            sinds = jax.random.permutation(shuffle_key, jnp.arange(xs.shape[0]))
            xs, weights = xs[sinds], weights[sinds]

        xs_train, xs_test = jnp.split(xs, 2, axis=0)
        w_train, w_test = jnp.split(weights, 2, axis=0)

        z, norm = self.embed(xs_train, w_train)
        keys = jax.random.split(key, xs_test.shape[0])
        loss, aux1, aux2 = jax.vmap(self.generator.loss, in_axes=(0, None, None, 0))(
            xs_test, z, params, keys
        )

        mean_loss = jnp.sum(loss * w_test) / (jnp.sum(w_test) + 1e-30)

        return mean_loss, jnp.mean(aux1), jnp.mean(aux2), norm


class FlowEmbedding(DistributionEmbedding):
    def log_prob(
        self,
        xs: jax.Array,
        z: jax.Array,
        reps: int = 1,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute log-probability of each row of `xs` under the embedded distribution.

        When the generator's `log_prob` is stochastic (e.g. dequantization noise in a
        normalizing flow), `reps > 1` runs that many independent passes and averages
        them in probability space (`logsumexp` over reps minus `log(reps)`), giving
        a lower-variance estimate.
        """

        def _compute_log_prob(key) -> tuple[jax.Array, jax.Array, jax.Array]:
            keys = jax.random.split(key, xs.shape[0])
            return jax.vmap(self.generator.log_prob, in_axes=(0, None, None, 0))(xs, z, None, keys)

        if key is None:
            return jax.vmap(self.generator.log_prob, in_axes=(0, None, None, None))(
                xs, z, None, None
            )

        if reps <= 1:
            return _compute_log_prob(key)

        log_prob_x, log_prob_z, ldj = jax.vmap(_compute_log_prob)(jax.random.split(key, reps))
        log_prob_x = jax.nn.logsumexp(log_prob_x, axis=0) - jnp.log(reps)

        return log_prob_x, jnp.mean(log_prob_z, axis=0), jnp.mean(ldj, axis=0)
