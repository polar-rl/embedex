import equinox as eqx
import jax
import jax.numpy as jnp


class PointwiseBlock(eqx.Module):
    """Wraps an MLP to operate on a set: (N, D_in) -> (N, D_out)."""

    mlp: eqx.nn.MLP

    def __call__(self, xs: jax.Array, weights: jax.Array) -> jax.Array:
        return jax.vmap(self.mlp)(xs)


class SetTransformerBlock(eqx.Module):
    """Induced Set Attention Block (ISAB) from the Set Transformer paper.

    Uses a small set of learned inducing points to compress the input set,
    then broadcasts back — giving O(NM) complexity instead of O(N²).

    Forward pass:
        H = MAB(I, X)   # (M, D) — compress N inputs to M inducing points
        Z = MAB(X, H)   # (N, D) — each input attends to the compressed set

    Where MAB(Q, K) = LayerNorm(Q + MHA(Q, K, K)) followed by a pointwise FFN.
    """

    q_proj_1: eqx.nn.Linear
    k_proj_1: eqx.nn.Linear
    v_proj_1: eqx.nn.Linear
    o_proj_1: eqx.nn.Linear
    ln1_1: eqx.nn.LayerNorm
    ff_1: eqx.nn.MLP
    ln2_1: eqx.nn.LayerNorm

    q_proj_2: eqx.nn.Linear
    k_proj_2: eqx.nn.Linear
    v_proj_2: eqx.nn.Linear
    o_proj_2: eqx.nn.Linear
    ln1_2: eqx.nn.LayerNorm
    ff_2: eqx.nn.MLP
    ln2_2: eqx.nn.LayerNorm

    in_proj: eqx.nn.Linear | None
    out_proj: eqx.nn.Linear | None

    inducing_points: jax.Array
    num_heads: int

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        num_inducing: int,
        ff_dim: int,
        key: jax.Array,
    ) -> None:
        keys = jax.random.split(key, 15)
        d = out_dim

        if in_dim != d:
            self.in_proj = eqx.nn.Linear(in_dim, d, key=keys[0])
        else:
            self.in_proj = None

        self.inducing_points = jax.random.normal(keys[1], (num_inducing, d)) * 0.01
        self.num_heads = num_heads

        self.q_proj_1 = eqx.nn.Linear(d, d, key=keys[2])
        self.k_proj_1 = eqx.nn.Linear(d, d, key=keys[3])
        self.v_proj_1 = eqx.nn.Linear(d, d, key=keys[4])
        self.o_proj_1 = eqx.nn.Linear(d, d, key=keys[5])
        self.ln1_1 = eqx.nn.LayerNorm(d)
        self.ff_1 = eqx.nn.MLP(d, d, ff_dim, 1, key=keys[6])
        self.ln2_1 = eqx.nn.LayerNorm(d)

        self.q_proj_2 = eqx.nn.Linear(d, d, key=keys[7])
        self.k_proj_2 = eqx.nn.Linear(d, d, key=keys[8])
        self.v_proj_2 = eqx.nn.Linear(d, d, key=keys[9])
        self.o_proj_2 = eqx.nn.Linear(d, d, key=keys[10])
        self.ln1_2 = eqx.nn.LayerNorm(d)
        self.ff_2 = eqx.nn.MLP(d, d, ff_dim, 1, key=keys[11])
        self.ln2_2 = eqx.nn.LayerNorm(d)

        if d != out_dim:
            self.out_proj = eqx.nn.Linear(d, out_dim, key=keys[12])
        else:
            self.out_proj = None

    def _mha(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        q_proj: eqx.nn.Linear,
        k_proj: eqx.nn.Linear,
        v_proj: eqx.nn.Linear,
        o_proj: eqx.nn.Linear,
        key_log_w: jax.Array | None = None,
    ) -> jax.Array:
        d = q.shape[-1]
        head_dim = d // self.num_heads

        Q = jax.vmap(q_proj)(q).reshape(-1, self.num_heads, head_dim)
        K = jax.vmap(k_proj)(k).reshape(-1, self.num_heads, head_dim)
        V = jax.vmap(v_proj)(v).reshape(-1, self.num_heads, head_dim)

        Q, K, V = (
            jnp.transpose(Q, (1, 0, 2)),
            jnp.transpose(K, (1, 0, 2)),
            jnp.transpose(V, (1, 0, 2)),
        )
        scale = jnp.sqrt(head_dim).astype(Q.dtype)
        logits = jnp.matmul(Q, jnp.transpose(K, (0, 2, 1))) / scale
        if key_log_w is not None:
            logits = logits + key_log_w[None, None, :].astype(logits.dtype)
        attn = jax.nn.softmax(logits, axis=-1)
        out = jnp.matmul(attn, V)

        out = jnp.transpose(out, (1, 0, 2)).reshape(-1, d)
        return jax.vmap(o_proj)(out)

    def _mab(
        self,
        q: jax.Array,
        kv: jax.Array,
        q_proj: eqx.nn.Linear,
        k_proj: eqx.nn.Linear,
        v_proj: eqx.nn.Linear,
        o_proj: eqx.nn.Linear,
        ln1: eqx.nn.LayerNorm,
        ff: eqx.nn.MLP,
        ln2: eqx.nn.LayerNorm,
        key_log_w: jax.Array | None = None,
    ) -> jax.Array:
        h = jax.vmap(ln1)(
            q + self._mha(q, kv, kv, q_proj, k_proj, v_proj, o_proj, key_log_w=key_log_w)
        )
        return jax.vmap(ln2)(h + jax.vmap(ff)(h))

    def __call__(self, xs: jax.Array, weights: jax.Array) -> jax.Array:
        if self.in_proj is not None:
            xs = jax.vmap(self.in_proj)(xs)

        w = weights / (jnp.sum(weights) + 1e-30)
        n = w.shape[0]
        key_log_w = jnp.log(jnp.clip(w * n, min=1e-30))

        inducing_points = jnp.broadcast_to(self.inducing_points, self.inducing_points.shape)
        h = self._mab(
            inducing_points,
            xs,
            self.q_proj_1,
            self.k_proj_1,
            self.v_proj_1,
            self.o_proj_1,
            self.ln1_1,
            self.ff_1,
            self.ln2_1,
            key_log_w=key_log_w,
        )
        out = self._mab(
            xs,
            h,
            self.q_proj_2,
            self.k_proj_2,
            self.v_proj_2,
            self.o_proj_2,
            self.ln1_2,
            self.ff_2,
            self.ln2_2,
        )

        if self.out_proj is not None:
            out = jax.vmap(self.out_proj)(out)

        return out
