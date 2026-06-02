from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class Donut(eqx.Module):
    mean: jax.Array
    radius: jax.Array
    width: jax.Array

    def sample(self, num_samples: int, key: jax.Array) -> jax.Array:
        direction_key, offset_key = jax.random.split(key, 2)

        a = jax.random.normal(direction_key, (num_samples, 2))
        a = a / jnp.linalg.norm(a, keepdims=True, axis=-1)
        b = jax.random.uniform(offset_key, (num_samples, 1), minval=-0.5, maxval=0.5)

        return self.mean + self.radius * a * (1.0 + self.width * b)

    @property
    def params(self) -> jax.Array:
        return jnp.concatenate(jax.tree.leaves(jax.tree.map(jnp.atleast_1d, self)))

    @classmethod
    def random(
        cls,
        mean_std: float,
        radius_range: tuple[float, float],
        width_range: tuple[float, float],
        key: jax.Array,
    ) -> Donut:
        mean_key, radius_key, width_key = jax.random.split(key, 3)

        mean = jax.random.normal(mean_key, (2,)) * mean_std
        radius = jax.random.uniform(radius_key, minval=radius_range[0], maxval=radius_range[1])
        width = jax.random.uniform(width_key, minval=width_range[0], maxval=width_range[1])

        return cls(mean=mean, radius=radius, width=width)


def random_donuts(
    num_distributions: int,
    mean_std: float,
    radius_range: tuple[float, float],
    width_range: tuple[float, float],
    key: jax.Array,
) -> Donut:
    """Generate a batch of random donuts as a single batched pytree."""

    return jax.vmap(Donut.random, in_axes=(None, None, None, 0))(
        mean_std, radius_range, width_range, jax.random.split(key, num_distributions)
    )


def sample_batch(
    donuts: Donut | list[Donut],
    num_samples: int,
    key: jax.Array,
) -> jax.Array:
    """Sample from a batch of donuts.

    Accepts either a batched pytree (from random_donuts) or a list
    of individual Donut instances (from manual construction).
    """

    if isinstance(donuts, list):
        donuts = jax.tree.map(lambda *xs: jnp.stack(xs), *donuts)

    return jax.vmap(Donut.sample, in_axes=(0, None, 0))(
        donuts, num_samples, jax.random.split(key, donuts.mean.shape[0])
    )


PAPER_DONUTS = [
    Donut(mean=jnp.array([-2.7, 0.5]), radius=jnp.float32(2.0), width=jnp.float32(0.45)),
    Donut(mean=jnp.array([0.25, 1.2]), radius=jnp.float32(1.4), width=jnp.float32(0.15)),
    Donut(mean=jnp.array([0.0, -0.5]), radius=jnp.float32(1.6), width=jnp.float32(0.3)),
]
