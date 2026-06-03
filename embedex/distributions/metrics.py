import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def sliced_wasserstein_distance(
    x: jax.Array,
    y: jax.Array,
    num_projections: int,
    key: jax.Array,
) -> jax.Array:
    """Sliced Wasserstein-2 distance between two (N, D) sample sets.

    Both *x* and *y* must have the same number of rows N.
    """

    directions = jax.random.normal(key, (num_projections, x.shape[-1]))
    directions = directions / jnp.linalg.norm(directions, keepdims=True, axis=-1)

    proj_x = x @ directions.T
    proj_y = y @ directions.T

    proj_x = jnp.sort(proj_x, axis=0)
    proj_y = jnp.sort(proj_y, axis=0)

    return jnp.sqrt(jnp.mean((proj_x - proj_y) ** 2))


def js_divergence(p: jax.Array, q: jax.Array) -> jax.Array:
    """Jensen-Shannon divergence between two PMFs over the same support."""

    eps = 1e-12
    p = (p + eps) / jnp.sum(p + eps)
    q = (q + eps) / jnp.sum(q + eps)
    m = 0.5 * (p + q)
    kl_pm = jnp.sum(p * jnp.log(p / m))
    kl_qm = jnp.sum(q * jnp.log(q / m))
    result = 0.5 * kl_pm + 0.5 * kl_qm
    return jnp.where(jnp.isfinite(result), result, 1.0)
