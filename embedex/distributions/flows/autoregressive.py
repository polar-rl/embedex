"""Autoregressive categorical generator conditioned on an embedding.

Models p(x | z) = prod_i p(x_i | x_<i, z) exactly via a causal transformer:
teacher-forced cross-entropy training, exact log-prob, exact ancestral sampling.
Serves as the ceiling for "neural renderer given z" — if it closes more of the
conditional gap than masked DFM, the DFM mechanism limited rendering; if it caps
at the same place, amortized inference over the family is the wall.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

import embedex.distributions.flows.flowee as flowee


class ARBlock(eqx.Module):
    """Pre-norm causal multi-head self-attention + FFN over the variable tokens."""

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    ff: eqx.nn.MLP
    num_heads: int = eqx.field(static=True)
    token_dim: int = eqx.field(static=True)

    def __call__(self, h: jax.Array) -> jax.Array:
        n, d = h.shape
        dh = d // self.num_heads
        hn = jax.vmap(self.ln1)(h)
        qs = jax.vmap(self.q_proj)(hn).reshape(n, self.num_heads, dh)
        ks = jax.vmap(self.k_proj)(hn).reshape(n, self.num_heads, dh)
        vs = jax.vmap(self.v_proj)(hn).reshape(n, self.num_heads, dh)

        scores = jnp.einsum('ihd,jhd->hij', qs, ks) / jnp.sqrt(dh)
        causal = jnp.tril(jnp.ones((n, n)))  # position i attends to j <= i
        scores = jnp.where(causal[None] > 0, scores, -1e30)
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum('hij,jhd->ihd', attn, vs).reshape(n, d)

        h = h + jax.vmap(self.o_proj)(ctx)
        h = h + jax.vmap(self.ff)(jax.vmap(self.ln2)(h))
        return h


class CausalARNet(eqx.Module):
    """Causal transformer giving p(x_i | x_<i, z) logits for every i in one pass."""

    bos_embedding: jax.Array  # (token_dim,)
    class_embeddings: jax.Array  # (num_classes, token_dim)
    pos_embeddings: jax.Array  # (num_variables, token_dim)
    cond_proj: eqx.nn.Linear  # conditioning_dim -> token_dim
    blocks: tuple
    final_norm: eqx.nn.LayerNorm
    out_proj: eqx.nn.Linear  # token_dim -> num_classes
    num_classes: int = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)

    def __call__(self, x_int: jax.Array, sideinfo: jax.Array | None) -> jax.Array:
        """x_int: (num_variables,) int. Returns logits (num_variables, num_classes).

        Teacher-forced inputs: position 0 sees BOS, position i (>=1) sees x_{i-1};
        the causal mask makes logits[i] depend only on x_<i (and z).
        """
        prev = self.class_embeddings[x_int[:-1]]  # (N-1, D)
        tok_in = jnp.concatenate([self.bos_embedding[None, :], prev], axis=0)  # (N, D)
        h = tok_in + self.pos_embeddings
        if sideinfo is not None:
            h = h + self.cond_proj(sideinfo)[None, :]
        for block in self.blocks:
            h = block(h)
        return jax.vmap(self.out_proj)(jax.vmap(self.final_norm)(h))


class AutoregressiveModel(flowee.Flow):
    """Exact autoregressive categorical model over `num_variables` discrete coords."""

    net: eqx.Module
    num_classes: int = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)

    def _to_int(self, x: jax.Array) -> jax.Array:
        return jnp.clip(jnp.round(x).astype(jnp.int32), 0, self.num_classes - 1)

    def _forward(self, x, ldj, sideinfo=None, params=None, key=None):
        raise NotImplementedError('AutoregressiveModel overrides loss/sample/log_prob directly')

    def _inverse(self, x, ldj, sideinfo=None, params=None, key=None):
        raise NotImplementedError('AutoregressiveModel overrides loss/sample/log_prob directly')

    def loss(self, x, sideinfo=None, params=None, key=None):
        x_int = self._to_int(x)
        logits = self.net(x_int, sideinfo)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.sum(log_probs[jnp.arange(self.num_variables), x_int])
        return ce, 0.0, 0.0

    def log_prob(self, x, sideinfo=None, params=None, key=None):
        x_int = self._to_int(x)
        logits = self.net(x_int, sideinfo)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        lp = jnp.sum(log_probs[jnp.arange(self.num_variables), x_int])
        return lp, lp, 0.0

    def sample(self, sideinfo, params, key):
        keys = jax.random.split(key, self.num_variables)

        def body(i, x_built):
            logits = self.net(x_built, sideinfo)
            xi = jax.random.categorical(keys[i], logits[i])
            return x_built.at[i].set(xi.astype(x_built.dtype))

        x0 = jnp.zeros(self.num_variables, dtype=jnp.int32)
        x_final = jax.lax.fori_loop(0, self.num_variables, body, x0)
        return x_final.astype(jnp.float32)
