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


class LikelihoodGate(eqx.Module):
    """Per-(sample, variable) weights from per-sample likelihood features.

    Variable i reads likelihood column `index[i]` (its own factor: the site's residual, the
    ghost's match, or the one shared scalar) and gates sample s by sigmoid(a_i * l + b_i).
    The weights are normalized over the samples for each variable, so with a scalar feature
    and a gate that closes on unlikely samples they are the posterior weights of the set;
    with no features every sample weighs the same.
    """

    a: jax.Array
    b: jax.Array
    num_variables: int = eqx.field(static=True)
    index: tuple[int, ...] | None = eqx.field(static=True)

    def __init__(self, num_variables: int, index: tuple[int, ...] | None = None):
        self.a = jnp.zeros(num_variables)
        self.b = jnp.zeros(num_variables)
        self.num_variables = num_variables
        if index is not None and len(index) != num_variables:
            raise ValueError(
                f'likelihood_index has {len(index)} entries for {num_variables} variables'
            )
        self.index = None if index is None else tuple(int(i) for i in index)

    def columns(self, d: int) -> jax.Array:
        """(N,) likelihood column each variable reads."""
        N = self.num_variables
        if self.index is not None:
            return jnp.asarray(self.index, jnp.int32)
        if d == 1:
            return jnp.zeros(N, jnp.int32)
        if d == N:
            return jnp.arange(N, dtype=jnp.int32)
        raise ValueError(
            f'{d} likelihood features for {N} variables need an explicit likelihood_index'
        )

    def __call__(self, num_samples: int, likelihood: jax.Array | None) -> jax.Array:
        """(n, N) sample weights, each column summing to one."""
        N = self.num_variables
        if likelihood is None:
            return jnp.full((num_samples, N), 1.0 / num_samples)
        lik = likelihood.astype(jnp.float32)[:, self.columns(likelihood.shape[-1])]
        g = jax.nn.sigmoid(self.a * lik + self.b)
        return g / (jnp.sum(g, axis=0, keepdims=True) + 1e-30)


def pair_weights(w: jax.Array) -> jax.Array:
    """(n, N, N) weights of sample s for the pair (i, j): the geometric mean of the two
    variables' sample weights, normalized over the samples. Equal per-variable weights
    give back the plain per-sample weight."""
    sw = jnp.sqrt(w)
    p = sw[:, :, None] * sw[:, None, :]
    return p / (jnp.sum(p, axis=0, keepdims=True) + 1e-30)


def _as_class_index(xs: jax.Array, num_variables: int, num_classes: int) -> jax.Array:
    xs = xs[:, :num_variables]
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

    def __call__(self, xs: jax.Array, likelihood: jax.Array | None = None) -> jax.Array:
        del likelihood  # the ablation arms weigh every sample the same
        x_int = _as_class_index(xs, self.num_variables, self.num_classes)
        agree = (x_int[:, :, None] == x_int[:, None, :]).astype(jnp.float32)
        return jnp.mean(agree, axis=0).reshape(-1)


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

    def __call__(self, xs: jax.Array, likelihood: jax.Array | None = None) -> jax.Array:
        del likelihood  # the ablation arms weigh every sample the same
        x_int = _as_class_index(xs, self.num_variables, self.num_classes)
        N = self.num_variables
        e = self.cell_embed[x_int]
        R = jnp.einsum('sid,sje->ijde', e, e) / xs.shape[0]
        Rp = jax.vmap(jax.vmap(self.Rproj))(R.reshape(N, N, -1))
        return jax.vmap(jax.vmap(self.edge_out))(Rp).reshape(-1)


class RelationalStream(eqx.Module):
    cell_embed: jax.Array
    Rproj: eqx.nn.Linear
    node_init: eqx.nn.Linear
    blocks: list
    gate: LikelihoodGate
    num_variables: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    d_e: int = eqx.field(static=True)
    node_dim: int = eqx.field(static=True)

    def __init__(
        self, num_variables, num_classes, d_e, num_layers, num_heads, key, node_dim=0, index=None
    ):
        ks = jax.random.split(key, num_layers + 3)
        dm = node_dim or num_variables
        nh = _largest_divisor_leq(dm, num_heads)
        self.cell_embed = jax.random.normal(ks[0], (num_classes, d_e)) * 0.1
        self.Rproj = eqx.nn.Linear(d_e * d_e, d_e, key=ks[1])
        self.node_init = eqx.nn.Linear(d_e, dm, key=ks[2])
        self.blocks = [EdgeAttn(dm, d_e, nh, ks[3 + i]) for i in range(num_layers)]
        self.gate = LikelihoodGate(num_variables, index)
        self.num_variables = num_variables
        self.num_classes = num_classes
        self.d_e = d_e
        self.node_dim = dm

    @property
    def out_dim(self) -> int:
        return self.num_variables * self.node_dim

    def __call__(self, xs: jax.Array, likelihood: jax.Array | None = None) -> jax.Array:
        x_int = _as_class_index(xs, self.num_variables, self.num_classes)
        e = self.cell_embed[x_int]
        N = self.num_variables
        w = self.gate(xs.shape[0], likelihood)
        R = jnp.einsum('sij,sid,sje->ijde', pair_weights(w), e, e)
        Rp = jax.vmap(jax.vmap(self.Rproj))(R.reshape(N, N, -1))
        h = jax.vmap(self.node_init)(jnp.sum(e * w[:, :, None], axis=0))
        for blk in self.blocks:
            h = blk(h, Rp)
        return h.reshape(-1)
