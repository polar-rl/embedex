import equinox as eqx
import jax
import jax.numpy as jnp

import embedex.distributions.flows.flowee as flowee


def sinusoidal_embedding(t: jax.Array, dim: int) -> jax.Array:
    """Sinusoidal positional embedding for scalar time t in [0, 1]."""

    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / half)
    args = t * freqs
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)])


class VelocityNet(eqx.Module):
    """Time-conditioned velocity network for flow matching.

    Concatenates x_t with a sinusoidal time embedding, then passes through
    a conditioned network with sideinfo.
    """

    time_embed_dim: int = eqx.field(static=True)
    net: eqx.Module

    def __call__(
        self,
        x_t: jax.Array,
        t: jax.Array,
        sideinfo: jax.Array | None = None,
    ) -> jax.Array:
        t_embed = sinusoidal_embedding(t, self.time_embed_dim)
        xt_input = jnp.concatenate([jnp.ravel(x_t), t_embed])
        if sideinfo is not None:
            return self.net(xt_input, sideinfo)
        return self.net(xt_input)


class FlowMatchingModel(flowee.Flow):
    """Continuous flow matching generative model.

    Learns a velocity field v_theta(x_t, t, z) that transports a standard
    Gaussian prior to the data distribution conditioned on embedding z.

    Training uses the conditional flow matching (CFM) objective with
    optimal transport interpolation.
    """

    velocity_net: VelocityNet
    input_dim: int = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    sigma_min: float = eqx.field(static=True)

    def sample(
        self,
        sideinfo: jax.Array | None,
        params: jax.Array | None,
        key: jax.Array,
    ) -> jax.Array:
        assert self.prior is not None, 'Must call `add_prior` before calling `sample`'
        sample_key, inverse_key = jax.random.split(key, 2)
        z = self.prior.sample(seed=sample_key, sample_shape=self.prior_shape)
        return self._inverse(z, 0, sideinfo=sideinfo, params=params, key=inverse_key)[0]

    def _inverse(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Noise -> data: integrate ODE from t=0 to t=1 (midpoint method)."""

        dt = 1.0 / self.num_steps
        ts = jnp.linspace(0.0, 1.0 - dt, self.num_steps)

        def step(x_t, t):
            v1 = self.velocity_net(x_t, t, sideinfo)
            x_mid = x_t + 0.5 * dt * v1
            v_mid = self.velocity_net(x_mid, t + 0.5 * dt, sideinfo)
            return x_t + dt * v_mid, None

        x_1, _ = jax.lax.scan(step, x, ts)
        return x_1, ldj

    def _forward(
        self,
        x: jax.Array,
        ldj: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Data -> noise: integrate ODE from t=1 to t=0 (midpoint method)."""

        dt = 1.0 / self.num_steps
        ts = jnp.linspace(1.0, dt, self.num_steps)

        def step(x_t, t):
            v1 = self.velocity_net(x_t, t, sideinfo)
            x_mid = x_t - 0.5 * dt * v1
            v_mid = self.velocity_net(x_mid, t - 0.5 * dt, sideinfo)
            return x_t - dt * v_mid, None

        x_0, _ = jax.lax.scan(step, x, ts)
        return x_0, ldj

    def loss(
        self,
        x: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Conditional flow matching loss.

        Samples t ~ U(0,1), z ~ N(0,I), computes interpolant x_t and
        target velocity u_t, returns MSE ||v_theta(x_t, t) - u_t||^2.
        """

        assert key is not None
        t_key, noise_key = jax.random.split(key, 2)

        t = jax.random.uniform(t_key)
        z = jax.random.normal(noise_key, x.shape)

        x_t = (1 - (1 - self.sigma_min) * t) * z + t * x
        u_t = x - (1 - self.sigma_min) * z

        v_t = self.velocity_net(x_t, t, sideinfo)

        mse = jnp.sum((v_t - u_t) ** 2)
        return mse, 0.0, 0.0

    def log_prob(
        self,
        x: jax.Array,
        sideinfo: jax.Array | None = None,
        params: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Log-probability via the continuous change-of-variables formula.

        Solves the augmented ODE backward from t=1 (data) to t=0 (noise),
        accumulating the divergence of the velocity field.
        """

        dt = 1.0 / self.num_steps
        ts = jnp.linspace(1.0, dt, self.num_steps)

        if key is not None:
            eps = jax.random.normal(key, x.shape)

            def step(carry, t):
                x_t, log_det = carry

                def v_fn(x_):
                    return self.velocity_net(x_, t, sideinfo)

                v, vjp_fn = jax.vjp(v_fn, x_t)
                (dvdx_eps,) = vjp_fn(eps)
                tr_dvdx = jnp.sum(dvdx_eps * eps)

                x_t = x_t - dt * v
                log_det = log_det + dt * tr_dvdx
                return (x_t, log_det), None
        else:

            def step(carry, t):
                x_t, log_det = carry

                def v_fn(x_):
                    return self.velocity_net(x_, t, sideinfo)

                v = v_fn(x_t)
                jac = jax.jacfwd(v_fn)(x_t)
                tr_dvdx = jnp.trace(jac)

                x_t = x_t - dt * v
                log_det = log_det + dt * tr_dvdx
                return (x_t, log_det), None

        (z, trace_integral), _ = jax.lax.scan(step, (x, 0.0), ts)

        log_prob_z = -0.5 * jnp.sum(z**2) - 0.5 * x.shape[-1] * jnp.log(2 * jnp.pi)
        ldj = -trace_integral
        log_prob_x = log_prob_z + ldj

        return log_prob_x, log_prob_z, ldj
