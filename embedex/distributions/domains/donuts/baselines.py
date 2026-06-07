import equinox as eqx
import jax
import jax.numpy as jnp

from embedex.distributions.domains.donuts.donut import Donut, donut_moments

_LOG_FLOOR = 1e-30


class KDEBaseline(eqx.Module):
    samples: jax.Array
    kernel_cov: jax.Array

    @classmethod
    def fit(cls, samples: jax.Array) -> 'KDEBaseline':
        n, d = samples.shape
        factor = n ** (-1.0 / (d + 4))

        cov = jnp.cov(samples.T) + 1e-6 * jnp.eye(d)
        return cls(samples=samples, kernel_cov=cov * factor**2)

    def sample(self, num_samples: int, key: jax.Array) -> jax.Array:
        idx_key, noise_key = jax.random.split(key, 2)

        n, d = self.samples.shape
        idx = jax.random.randint(idx_key, (num_samples,), 0, n)
        chol = jnp.linalg.cholesky(self.kernel_cov)
        noise = jax.random.normal(noise_key, (num_samples, d)) @ chol.T

        return self.samples[idx] + noise

    def log_prob(self, xs: jax.Array) -> jax.Array:
        n, d = self.samples.shape
        chol = jnp.linalg.cholesky(self.kernel_cov)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        log_norm = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det) - jnp.log(n)

        def per_point(x: jax.Array) -> jax.Array:
            diff = x - self.samples
            sol = jax.scipy.linalg.cho_solve((chol, True), diff.T).T
            quad = jnp.sum(diff * sol, axis=-1)
            return jax.nn.logsumexp(-0.5 * quad) + log_norm

        return jax.vmap(per_point)(xs)


class OracleDonutFit(eqx.Module):
    """Method-of-moments donut fit; the true family is hard-coded.

    Fit: center = sample mean; radius = mean distance to center; the radial
    distribution is U[r(1 - w/2), r(1 + w/2)], whose std is r*w/sqrt(12), so
    width = sqrt(12) * std(distances) / radius.
    """

    donut: Donut
    smoothing: jax.Array

    @classmethod
    def fit(cls, samples: jax.Array) -> 'OracleDonutFit':
        n = samples.shape[0]

        center, radius, width = donut_moments(samples)
        dists = jnp.linalg.norm(samples - center, axis=-1)
        dists_std = jnp.std(dists)

        iqr = jnp.percentile(dists, 75) - jnp.percentile(dists, 25)
        spread = jnp.minimum(dists_std, iqr / 1.34) + 1e-4
        smoothing = 0.9 * spread * n ** (-1.0 / 5)

        donut = Donut(mean=center, radius=radius, width=width)
        return cls(donut=donut, smoothing=smoothing)

    def sample(self, num_samples: int, key: jax.Array) -> jax.Array:
        return self.donut.sample(num_samples, key)

    def log_prob(self, xs: jax.Array) -> jax.Array:
        radius, width = self.donut.radius, self.donut.width
        lo = radius * (1.0 - 0.5 * width)
        hi = radius * (1.0 + 0.5 * width)
        length = jnp.maximum(hi - lo, 1e-8)

        dists = jnp.linalg.norm(xs - self.donut.mean, axis=-1)
        dists = jnp.maximum(dists, 1e-8)

        # Radial density: U[lo, hi] convolved with N(0, smoothing^2).
        cdf_hi = jax.scipy.stats.norm.cdf((hi - dists) / self.smoothing)
        cdf_lo = jax.scipy.stats.norm.cdf((lo - dists) / self.smoothing)
        radial_density = (cdf_hi - cdf_lo) / length

        density = radial_density / (2.0 * jnp.pi * dists)
        return jnp.log(jnp.maximum(density, _LOG_FLOOR))
