from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def all_states(num_spins: int) -> jax.Array:
    """Enumerate every binary configuration of the field, shape (2**N, N)."""

    indices = jnp.arange(2**num_spins, dtype=jnp.int32)
    bits = ((indices[:, None] >> jnp.arange(num_spins)[None, :]) & 1).astype(jnp.float32)
    return bits


def num_params(num_spins: int) -> int:
    """Number of free parameters: one coupling per edge plus one bias per node."""

    return num_spins * (num_spins - 1) // 2 + num_spins


class MRF(eqx.Module):
    """A pairwise Markov random field over binary variables (an Ising model).

    P(x | J, h) = (1/Z) exp( sum_{i<j} J_ij x_i x_j + sum_i h_i x_i ),  x_i in {0, 1}

    The field lives on a graph whose weighted adjacency matrix is `J` (zeros
    encode conditional independencies) with per-node biases `h`. A sample is a
    state of the field: one assignment to every node. Currently binary and
    fully connected; categorical (Potts) and random-graph variants will extend
    this without changing the interface.
    """

    J: jax.Array
    h: jax.Array
    num_spins: int = eqx.field(static=True)

    @property
    def states(self) -> jax.Array:
        return all_states(self.num_spins)

    def log_pmf(self) -> jax.Array:
        states = self.states
        field_term = states @ self.h
        coupling_term = 0.5 * jnp.sum((states @ self.J) * states, axis=-1)
        energies = field_term + coupling_term
        return energies - jax.nn.logsumexp(energies)

    def pmf(self) -> jax.Array:
        return jnp.exp(self.log_pmf())

    def sample(self, num_samples: int, key: jax.Array) -> jax.Array:
        """Draw states of the field, shape (num_samples, N)."""

        indices = jax.random.categorical(key, self.log_pmf(), shape=(num_samples,))
        return self.states[indices]

    def single_marginals(self) -> jax.Array:
        """P(x_i = 1) vector, shape (N,)."""

        return self.states.T @ self.pmf()

    def pairwise_marginals(self) -> jax.Array:
        """P(x_i = 1, x_j = 1) matrix, shape (N, N)."""

        probs = self.pmf()
        return (self.states * probs[:, None]).T @ self.states

    @property
    def params(self) -> jax.Array:
        rows, cols = jnp.triu_indices(self.num_spins, k=1)
        return jnp.concatenate([self.J[rows, cols], self.h])

    @classmethod
    def random(
        cls,
        num_spins: int,
        coupling_scale: float,
        bias_scale: float,
        key: jax.Array,
    ) -> MRF:
        """Random field with Gaussian couplings and biases."""

        j_key, h_key = jax.random.split(key, 2)

        j_raw = jax.random.normal(j_key, (num_spins, num_spins)) * coupling_scale
        J = (j_raw + j_raw.T) / 2
        J = J.at[jnp.diag_indices(num_spins)].set(0.0)

        h = jax.random.normal(h_key, (num_spins,)) * bias_scale

        return cls(J=J, h=h, num_spins=num_spins)


def random_mrfs(
    num_distributions: int,
    num_spins: int,
    coupling_scale: float,
    bias_scale: float,
    key: jax.Array,
) -> MRF:
    """Generate a batch of random MRFs as a single batched pytree."""

    return jax.vmap(MRF.random, in_axes=(None, None, None, 0))(
        num_spins, coupling_scale, bias_scale, jax.random.split(key, num_distributions)
    )


def sample_batch(
    mrfs: MRF | list[MRF],
    num_samples: int,
    key: jax.Array,
) -> jax.Array:
    """Sample from a batch of MRFs.

    Accepts either a batched pytree (from random_mrfs) or a list of individual
    MRF instances (from manual construction).
    """

    if isinstance(mrfs, list):
        mrfs = jax.tree.map(lambda *xs: jnp.stack(xs), *mrfs)

    return jax.vmap(MRF.sample, in_axes=(0, None, 0))(
        mrfs, num_samples, jax.random.split(key, mrfs.J.shape[0])
    )
