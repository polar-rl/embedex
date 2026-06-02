import equinox as eqx
import jax
import jax.numpy as jnp

from embedex.distributions.flows.flowee import Flow


class Dequantize(Flow):
    """Learned dequantization noise q_φ(u | x, sideinfo) as a flow layer.

    Uses a small MLP to predict per-dimension location and scale of a
    logistic distribution, then pushes through sigmoid to bound the
    output to (-0.5, 0.5)^d.

    When conditioning_dim > 0, sideinfo is concatenated with x before
    feeding into the MLP, allowing distribution-dependent noise.

    Forward:  x_int → x_int + u,  ldj += -log q(u | x)
    Inverse:  x_cont → floor(x_cont)
    """

    net: eqx.nn.MLP
    input_dim: int
    conditioning_dim: int

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        assert key is not None, 'Dequantize requires a PRNG key'

        d = self.input_dim
        x_flat = x.reshape(-1)
        if self.conditioning_dim > 0 and sideinfo is not None:
            net_input = jnp.concatenate([x_flat, sideinfo.reshape(-1)])
        else:
            net_input = x_flat
        net_params = self.net(net_input)
        mu, log_s = net_params[:d], net_params[d:]

        log_s = jnp.clip(log_s, -5.0, 3.0)
        s = jnp.exp(log_s)

        eps = jax.random.logistic(key, (d,))
        v = mu + s * eps
        v = jnp.clip(v, -5.0, 5.0)

        u = jax.nn.sigmoid(v) - 0.5

        log_logistic = -eps - 2.0 * jax.nn.softplus(-eps)
        log_det_eps_to_v = jnp.sum(log_s)
        log_det_v_to_u = jnp.sum(jax.nn.log_sigmoid(v) + jax.nn.log_sigmoid(-v))
        log_q = jnp.sum(log_logistic) - log_det_eps_to_v - log_det_v_to_u

        return x + u.reshape(x.shape), ldj - log_q

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        return jnp.round(x), ldj
