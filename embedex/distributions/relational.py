import equinox as eqx
import jax
import jax.numpy as jnp


def _largest_divisor_leq(n: int, k: int) -> int:
    for d in range(min(k, n), 0, -1):
        if n % d == 0:
            return d
    return 1


class EdgeAttn(eqx.Module):
    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    o: eqx.nn.Linear
    ebias: eqx.nn.Linear
    ln1: eqx.nn.LayerNorm
    ff: eqx.nn.MLP
    ln2: eqx.nn.LayerNorm
    nh: int = eqx.field(static=True)

    def __init__(self, dm: int, de: int, nh: int, key: jax.Array) -> None:
        ks = jax.random.split(key, 6)
        self.q = eqx.nn.Linear(dm, dm, key=ks[0])
        self.k = eqx.nn.Linear(dm, dm, key=ks[1])
        self.v = eqx.nn.Linear(dm, dm, key=ks[2])
        self.o = eqx.nn.Linear(dm, dm, key=ks[3])
        self.ebias = eqx.nn.Linear(de, nh, key=ks[4])
        self.ln1 = eqx.nn.LayerNorm(dm)
        self.ff = eqx.nn.MLP(dm, dm, 2 * dm, 1, key=ks[5])
        self.ln2 = eqx.nn.LayerNorm(dm)
        self.nh = nh

    def __call__(self, h: jax.Array, R: jax.Array) -> jax.Array:
        N, dm = h.shape
        hd = dm // self.nh
        Q = jax.vmap(self.q)(h).reshape(N, self.nh, hd).transpose(1, 0, 2)
        K = jax.vmap(self.k)(h).reshape(N, self.nh, hd).transpose(1, 0, 2)
        V = jax.vmap(self.v)(h).reshape(N, self.nh, hd).transpose(1, 0, 2)
        bias = jax.vmap(jax.vmap(self.ebias))(R).transpose(2, 0, 1)  # (nh, N, N)
        logits = jnp.matmul(Q, K.transpose(0, 2, 1)) / jnp.sqrt(hd) + bias
        a = jax.nn.softmax(logits, axis=-1)
        out = jnp.matmul(a, V).transpose(1, 0, 2).reshape(N, dm)
        h = jax.vmap(self.ln1)(h + jax.vmap(self.o)(out))
        return jax.vmap(self.ln2)(h + jax.vmap(self.ff)(h))


def _normalized_weights(xs: jax.Array, weights: jax.Array | None) -> jax.Array:
    if weights is None:
        return jnp.full(xs.shape[0], 1.0 / xs.shape[0])
    return weights / (jnp.sum(weights) + 1e-30)


def _as_class_index(xs: jax.Array, num_classes: int) -> jax.Array:
    return jnp.clip(jnp.round(xs.astype(jnp.float32)).astype(jnp.int32), 0, num_classes - 1)


class RawAgreeStream(eqx.Module):
    """Ablation arm rawA: variable i is conditioned on row i of the agreement matrix.

    A_ij = sum_s w_s 1[x_i^(s) == x_j^(s)], so A_ii = 1 once the weights normalize. The
    stream has no learned parameters at all: it is the pairwise statistic itself, delivered
    the way the learned stream delivers its readout. It collapses the q x q co-occurrence to
    a single agreement scalar per pair, which is why it is run alongside RawEdgeStream rather
    than alone.
    """

    num_variables: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)

    @property
    def out_dim(self) -> int:
        return self.num_variables * self.num_variables

    def __call__(self, xs: jax.Array, weights: jax.Array | None = None) -> jax.Array:
        x_int = _as_class_index(xs, self.num_classes)
        w = _normalized_weights(xs, weights)
        agree = (x_int[:, :, None] == x_int[:, None, :]).astype(w.dtype)
        return jnp.einsum('s,sij->ij', w, agree).reshape(-1)


class RawEdgeStream(eqx.Module):
    """Ablation arm rawE: the learned stream with the EdgeAttn blocks deleted.

    Keeps the cell embedding, the weighted outer-product sample pool and the edge projection,
    then reads each pair down to one number with a shared linear map. The only difference
    from RelationalStream is the attention, so the two arms separate the richness of the
    statistic from the explaining-away step that turns correlations into couplings.
    """

    cell_embed: jax.Array
    Rproj: eqx.nn.Linear
    edge_out: eqx.nn.Linear
    num_variables: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    d_e: int = eqx.field(static=True)

    def __init__(self, num_variables, num_classes, d_e, key):
        ks = jax.random.split(key, 3)
        self.cell_embed = jax.random.normal(ks[0], (num_classes, d_e)) * 0.1
        self.Rproj = eqx.nn.Linear(d_e * d_e, d_e, key=ks[1])
        self.edge_out = eqx.nn.Linear(d_e, 1, key=ks[2])
        self.num_variables = num_variables
        self.num_classes = num_classes
        self.d_e = d_e

    @property
    def out_dim(self) -> int:
        return self.num_variables * self.num_variables

    def __call__(self, xs: jax.Array, weights: jax.Array | None = None) -> jax.Array:
        x_int = _as_class_index(xs, self.num_classes)
        w = _normalized_weights(xs, weights)
        N = self.num_variables
        e = self.cell_embed[x_int]
        R = jnp.einsum('s,sid,sje->ijde', w, e, e)
        Rp = jax.vmap(jax.vmap(self.Rproj))(R.reshape(N, N, -1))
        return jax.vmap(jax.vmap(self.edge_out))(Rp).reshape(-1)


class RelationalStream(eqx.Module):
    cell_embed: jax.Array
    Rproj: eqx.nn.Linear
    node_init: eqx.nn.Linear
    blocks: list
    num_variables: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    d_e: int = eqx.field(static=True)

    def __init__(self, num_variables, num_classes, d_e, num_layers, num_heads, key):
        ks = jax.random.split(key, num_layers + 3)
        dm = num_variables
        nh = _largest_divisor_leq(dm, num_heads)
        self.cell_embed = jax.random.normal(ks[0], (num_classes, d_e)) * 0.1
        self.Rproj = eqx.nn.Linear(d_e * d_e, d_e, key=ks[1])
        self.node_init = eqx.nn.Linear(d_e, dm, key=ks[2])
        self.blocks = [EdgeAttn(dm, d_e, nh, ks[3 + i]) for i in range(num_layers)]
        self.num_variables = num_variables
        self.num_classes = num_classes
        self.d_e = d_e

    @property
    def out_dim(self) -> int:
        return self.num_variables * self.num_variables

    def __call__(self, xs: jax.Array, weights: jax.Array | None = None) -> jax.Array:
        x_int = jnp.clip(
            jnp.round(xs.astype(jnp.float32)).astype(jnp.int32), 0, self.num_classes - 1
        )
        e = self.cell_embed[x_int]
        N = self.num_variables
        if weights is None:
            w = jnp.full(xs.shape[0], 1.0 / xs.shape[0])
        else:
            w = weights / (jnp.sum(weights) + 1e-30)
        R = jnp.einsum('s,sid,sje->ijde', w, e, e)
        Rp = jax.vmap(jax.vmap(self.Rproj))(R.reshape(N, N, -1))
        h = jax.vmap(self.node_init)(jnp.sum(e * w[:, None, None], axis=0))
        for blk in self.blocks:
            h = blk(h, Rp)
        return h.reshape(-1)
