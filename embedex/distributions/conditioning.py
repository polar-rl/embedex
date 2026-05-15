import equinox as eqx
import jax
import jax.numpy as jnp


class ConcatMLP(eqx.Module):
    """eqx.nn.MLP that accepts an optional conditioning vector via concatenation.

    Implements the concat-conditioning branch of the polymorphic conditioned-MLP
    interface, alongside `FiLMMLP` and `HyperMLP`.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        key: jax.Array,
    ) -> None:
        self.mlp = eqx.nn.MLP(in_size, out_size, width_size, depth, key=key)

    def __call__(self, x: jax.Array, sideinfo: jax.Array | None = None) -> jax.Array:
        if sideinfo is not None:
            x = jnp.concatenate([x, sideinfo])
        return self.mlp(x)


class FiLMMLP(eqx.Module):
    """MLP with FiLM conditioning.

    Instead of concatenating sideinfo with the input, sideinfo modulates each
    hidden layer via learned affine parameters (gamma, beta):

        h_l = relu(W_l @ h_{l-1} + b_l)
        h_l = (1 + gamma_l) * h_l + beta_l

    where (gamma_l, beta_l) = Linear_l(sideinfo).  The (1 + gamma) form means
    the default modulation is the identity.
    """

    layers: tuple[eqx.nn.Linear, ...]
    film_layers: tuple[eqx.nn.Linear, ...]
    final_layer: eqx.nn.Linear

    def __init__(
        self,
        in_size: int,
        conditioning_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        key: jax.Array,
    ) -> None:
        keys = jax.random.split(key, 2 * depth + 1)

        layers = []
        film_layers = []
        for i in range(depth):
            in_dim = in_size if i == 0 else width_size
            layers.append(eqx.nn.Linear(in_dim, width_size, key=keys[i]))
            film_layers.append(
                eqx.nn.Linear(conditioning_size, 2 * width_size, key=keys[depth + i])
            )

        self.layers = tuple(layers)
        self.film_layers = tuple(film_layers)
        self.final_layer = eqx.nn.Linear(
            width_size if depth > 0 else in_size, out_size, key=keys[2 * depth]
        )

    def __call__(self, x: jax.Array, sideinfo: jax.Array | None = None) -> jax.Array:
        x = jnp.ravel(x)
        if sideinfo is not None:
            sideinfo = jnp.ravel(sideinfo)

        for layer, film_layer in zip(self.layers, self.film_layers, strict=False):
            x = layer(x)
            if sideinfo is not None:
                film_params = film_layer(sideinfo)
                gamma, beta = jnp.split(film_params, 2)
                x = (1.0 + gamma) * x + beta
            x = jax.nn.relu(x)

        return self.final_layer(x)


class HyperMLP(eqx.Module):
    """MLP with hypernetwork conditioning on the first layer.

    The conditioning signal generates the first layer's weight matrix and bias
    via a small hypernetwork MLP.

    Subsequent hidden layers are standard (unconditioned).
    """

    hyper_net: eqx.nn.MLP
    default_first_layer: eqx.nn.Linear
    layers: tuple[eqx.nn.Linear, ...]
    final_layer: eqx.nn.Linear
    in_size: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        conditioning_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        key: jax.Array,
    ) -> None:
        keys = jax.random.split(key, depth + 3)

        self.in_size = in_size
        self.width_size = width_size
        first_layer_size = width_size * in_size + width_size
        self.hyper_net = eqx.nn.MLP(
            in_size=conditioning_size,
            out_size=first_layer_size,
            width_size=width_size,
            depth=1,
            key=keys[0],
        )

        self.default_first_layer = eqx.nn.Linear(in_size, width_size, key=keys[1])

        remaining = []
        for i in range(1, depth):
            remaining.append(eqx.nn.Linear(width_size, width_size, key=keys[i + 1]))
        self.layers = tuple(remaining)

        self.final_layer = eqx.nn.Linear(
            width_size if depth > 0 else in_size, out_size, key=keys[depth + 2]
        )

    def __call__(self, x: jax.Array, sideinfo: jax.Array | None = None) -> jax.Array:
        x = jnp.ravel(x)

        if sideinfo is not None:
            sideinfo = jnp.ravel(sideinfo)
            params = self.hyper_net(sideinfo)
            W = params[: self.width_size * self.in_size].reshape(self.width_size, self.in_size)
            b = params[self.width_size * self.in_size :]
            x = jax.nn.relu(W @ x + b)
        else:
            x = jax.nn.relu(self.default_first_layer(x))

        for layer in self.layers:
            x = jax.nn.relu(layer(x))

        return self.final_layer(x)


def batch_layer_norm(tokens: jax.Array, norm: eqx.nn.LayerNorm) -> jax.Array:
    """Apply LayerNorm to a batch of tokens without vmap.

    Operates directly on `(N, D)` tensors using the weight/bias from an
    `eqx.nn.LayerNorm` instance, avoiding Equinox pytree-tracing overhead.
    """

    mean = tokens.mean(axis=-1, keepdims=True)
    var = tokens.var(axis=-1, keepdims=True)
    h = (tokens - mean) / jnp.sqrt(var + norm.eps)
    if norm.use_weight:
        h = h * norm.weight
    if norm.use_bias:
        h = h + norm.bias
    return h


def batch_linear(x: jax.Array, linear: eqx.nn.Linear) -> jax.Array:
    """Apply a Linear layer to a batch of vectors without vmap.

    Computes `x @ W^T + b` directly for `x` of shape `(N, in)`.
    """

    out = x @ linear.weight.T
    if linear.bias is not None:
        out = out + linear.bias
    return out


class CrossAttentionBlock(eqx.Module):
    """Self-attention among tokens + cross-attention to conditioning tokens.

    Each block applies:

    1. **Self-attention** with pre-norm among the `num_variables` tokens so
       variables can communicate with each other.
    2. **Cross-attention** with pre-norm where variable tokens attend to
       conditioning tokens (entity parameters), giving each variable direct
       access to the full conditioning information at every layer.
    3. **Feed-forward** MLP with pre-norm.

    All sub-layers use residual connections.  Time is injected additively
    into the token representations before the first block (by the caller).
    """

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    sa_norm: eqx.nn.LayerNorm
    sa_qkv_proj: eqx.nn.Linear
    sa_out_proj: eqx.nn.Linear
    xa_norm_q: eqx.nn.LayerNorm
    xa_norm_kv: eqx.nn.LayerNorm
    xa_q_proj: eqx.nn.Linear
    xa_kv_proj: eqx.nn.Linear
    xa_out_proj: eqx.nn.Linear
    ff_norm: eqx.nn.LayerNorm
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear

    def __init__(
        self,
        token_dim: int,
        cond_token_dim: int,
        num_heads: int,
        *,
        ff_multiplier: int,
        key: jax.Array,
    ) -> None:
        if token_dim % num_heads != 0:
            raise ValueError(
                f'token_dim ({token_dim}) must be divisible by num_heads ({num_heads})'
            )
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads

        keys = jax.random.split(key, 8)

        self.sa_norm = eqx.nn.LayerNorm(token_dim)
        self.sa_qkv_proj = eqx.nn.Linear(token_dim, 3 * token_dim, key=keys[0])
        self.sa_out_proj = eqx.nn.Linear(token_dim, token_dim, key=keys[1])

        self.xa_norm_q = eqx.nn.LayerNorm(token_dim)
        self.xa_norm_kv = eqx.nn.LayerNorm(cond_token_dim)
        self.xa_q_proj = eqx.nn.Linear(token_dim, token_dim, key=keys[2])
        self.xa_kv_proj = eqx.nn.Linear(cond_token_dim, 2 * token_dim, key=keys[3])
        self.xa_out_proj = eqx.nn.Linear(token_dim, token_dim, key=keys[4])

        self.ff_norm = eqx.nn.LayerNorm(token_dim)
        ff_dim = ff_multiplier * token_dim
        self.ff1 = eqx.nn.Linear(token_dim, ff_dim, key=keys[5])
        self.ff2 = eqx.nn.Linear(ff_dim, token_dim, key=keys[6])

    def _multihead_attn(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
    ) -> jax.Array:
        """Compute multi-head attention given projected q, k, v."""

        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_heads, self.head_dim)
        v = v.reshape(-1, self.num_heads, self.head_dim)

        scale = self.head_dim**-0.5
        attn = jnp.einsum('thd,shd->hts', q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum('hts,shd->thd', attn, v)
        return out.reshape(-1, self.num_heads * self.head_dim)

    def __call__(
        self,
        tokens: jax.Array,
        cond_tokens: jax.Array,
    ) -> jax.Array:
        """Apply self-attn → cross-attn → FF block.

        Args:
            tokens: `(num_variables, token_dim)` variable representations.
            cond_tokens: `(num_cond_tokens, cond_token_dim)` conditioning
                representations (e.g. entity parameter tokens).

        Returns:
            `(num_variables, token_dim)` updated variable representations.
        """

        h = batch_layer_norm(tokens, self.sa_norm)
        qkv = batch_linear(h, self.sa_qkv_proj)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        sa_out = self._multihead_attn(q, k, v)
        sa_out = batch_linear(sa_out, self.sa_out_proj)
        tokens = tokens + sa_out

        h_q = batch_layer_norm(tokens, self.xa_norm_q)
        h_kv = batch_layer_norm(cond_tokens, self.xa_norm_kv)
        q = batch_linear(h_q, self.xa_q_proj)
        kv = batch_linear(h_kv, self.xa_kv_proj)
        k, v = jnp.split(kv, 2, axis=-1)
        xa_out = self._multihead_attn(q, k, v)
        xa_out = batch_linear(xa_out, self.xa_out_proj)
        tokens = tokens + xa_out

        h = batch_layer_norm(tokens, self.ff_norm)
        h = batch_linear(h, self.ff1)
        h = jax.nn.gelu(h)
        h = batch_linear(h, self.ff2)
        tokens = tokens + h

        return tokens
