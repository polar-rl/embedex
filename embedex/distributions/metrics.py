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
