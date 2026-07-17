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

    def step(
        self, h_i: jax.Array, k_cache: jax.Array, v_cache: jax.Array, i: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        dh = self.token_dim // self.num_heads
        hn = self.ln1(h_i)
        q = self.q_proj(hn).reshape(self.num_heads, dh)
        k_cache = k_cache.at[i].set(self.k_proj(hn).reshape(self.num_heads, dh))
        v_cache = v_cache.at[i].set(self.v_proj(hn).reshape(self.num_heads, dh))

        scores = jnp.einsum('hd,jhd->hj', q, k_cache) / jnp.sqrt(dh)
        mask = jnp.arange(k_cache.shape[0]) <= i
        scores = jnp.where(mask[None, :], scores, -1e30)
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum('hj,jhd->hd', attn, v_cache).reshape(-1)

        h_i = h_i + self.o_proj(ctx)
        h_i = h_i + self.ff(self.ln2(h_i))
        return h_i, k_cache, v_cache


class CausalARNet(eqx.Module):
    """Causal transformer giving p(x_i | x_<i, z) logits for every i in one pass."""

    bos_embedding: jax.Array
    class_embeddings: jax.Array
    pos_embeddings: jax.Array
    cond_proj: eqx.nn.Linear
    blocks: tuple
    final_norm: eqx.nn.LayerNorm
    out_proj: eqx.nn.Linear
    num_classes: int = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    rel_node_dim: int = eqx.field(static=True, default=0)
    moment_dim: int = eqx.field(static=True, default=0)

    def _static_inputs(self, sideinfo: jax.Array | None) -> jax.Array:
        h = self.pos_embeddings
        if sideinfo is not None:
            if self.rel_node_dim > 0:
                # z = [mu ; Jhat(N*N)]; deliver [mu ; row_i] to node i's slot (node-aligned)
                n = self.rel_node_dim
                mu = sideinfo[: self.moment_dim]
                jrows = sideinfo[self.moment_dim :].reshape(n, n)
                pernode = jnp.concatenate(
                    [jnp.broadcast_to(mu, (n, self.moment_dim)), jrows], axis=-1
                )
                h = h + jax.vmap(self.cond_proj)(pernode)
            else:
                h = h + self.cond_proj(sideinfo)[None, :]
        return h

    def __call__(self, x_int: jax.Array, sideinfo: jax.Array | None) -> jax.Array:
        prev = self.class_embeddings[x_int[:-1]]  # (N-1, D)
        tok_in = jnp.concatenate([self.bos_embedding[None, :], prev], axis=0)  # (N, D)
        h = tok_in + self._static_inputs(sideinfo)
        for block in self.blocks:
            h = block(h)
        return jax.vmap(self.out_proj)(jax.vmap(self.final_norm)(h))


class AutoregressiveModel(flowee.Flow):
    """Exact autoregressive categorical model over `num_variables` discrete coords."""

    net: eqx.Module
    num_classes: int = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)

    def _to_int(self, x: jax.Array) -> jax.Array:
        return jnp.clip(jnp.round(x.astype(jnp.float32)).astype(jnp.int32), 0, self.num_classes - 1)

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
        net = self.net
        static_in = net._static_inputs(sideinfo)
        num_heads = net.blocks[0].num_heads
        dh = net.blocks[0].token_dim // num_heads
        cache0 = jnp.zeros((len(net.blocks), self.num_variables, num_heads, dh))

        def body(i, carry):
            x, kc, vc = carry
            prev_tok = jnp.where(
                i == 0, net.bos_embedding, net.class_embeddings[x[jnp.maximum(i - 1, 0)]]
            )
            h_i = prev_tok + static_in[i]
            for layer, block in enumerate(net.blocks):
                h_i, k_l, v_l = block.step(h_i, kc[layer], vc[layer], i)
                kc = kc.at[layer].set(k_l)
                vc = vc.at[layer].set(v_l)
            logits_i = net.out_proj(net.final_norm(h_i))
            xi = jax.random.categorical(keys[i], logits_i)
            return x.at[i].set(xi.astype(x.dtype)), kc, vc

        x0 = jnp.zeros(self.num_variables, dtype=jnp.int32)
        x_final, _, _ = jax.lax.fori_loop(0, self.num_variables, body, (x0, cache0, cache0))
        return x_final.astype(jnp.float32)
