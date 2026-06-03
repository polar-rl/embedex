from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp


def all_states(num_spins: int, num_states: int) -> jax.Array:
    total = num_states**num_spins
    indices = jnp.arange(total, dtype=jnp.int32)
    powers = num_states ** jnp.arange(num_spins, dtype=jnp.int32)
    return ((indices[:, None] // powers[None, :]) % num_states).astype(jnp.int32)


def _adjacency(num_spins: int, edge_prob: jax.Array | float, key: jax.Array) -> jax.Array:
    draws = jax.random.bernoulli(key, edge_prob, (num_spins, num_spins)).astype(jnp.float32)
    upper = jnp.triu(draws, k=1)
    return upper + upper.T


class MRF(eqx.Module):
    """A pairwise Markov random field over categorical variables."""

    h: jax.Array
    J: jax.Array
    num_spins: int = eqx.field(static=True)
    num_states: int = eqx.field(static=True)

    @property
    def states(self) -> jax.Array:
        return all_states(self.num_spins, self.num_states)

    def _onehot(self) -> jax.Array:
        return jax.nn.one_hot(self.states, self.num_states)

    def log_pmf(self) -> jax.Array:
        onehot = self._onehot()
        field = jnp.einsum('mic,ic->m', onehot, self.h)
        coupling = 0.5 * jnp.einsum('mia,mjb,ijab->m', onehot, onehot, self.J)
        energies = field + coupling
        return energies - jax.nn.logsumexp(energies)

    def pmf(self) -> jax.Array:
        return jnp.exp(self.log_pmf())

    def sample(self, num_samples: int, key: jax.Array) -> jax.Array:
        indices = jax.random.categorical(key, self.log_pmf(), shape=(num_samples,))
        return self.states[indices].astype(jnp.float32)

    def single_marginals(self) -> jax.Array:
        return jnp.einsum('m,mic->ic', self.pmf(), self._onehot())

    def pairwise_marginals(self) -> jax.Array:
        onehot = self._onehot()
        return jnp.einsum('m,mia,mjb->ijab', self.pmf(), onehot, onehot)

    def agreement_marginals(self) -> jax.Array:
        onehot = self._onehot()
        return jnp.einsum('m,mic,mjc->ij', self.pmf(), onehot, onehot)

    @classmethod
    def random(
        cls,
        *,
        num_spins: int,
        num_states: int,
        coupling_scale: float,
        bias_scale: float,
        edge_prob: float,
        edge_prob_min: float,
        edge_prob_max: float,
        graph_type: str,
        key: jax.Array,
    ) -> 'MRF':
        """Random MRF with signed Gaussian potentials on a random graph.

        graph_type:
            'full'                    -> complete graph (edge_prob ignored)
            'random_edge_fixed_prob'  -> each edge present with probability edge_prob
            'random_edge_random_prob' -> per-field density ~ U(edge_prob_min, edge_prob_max)
        """

        h_key, j_key, p_key, edge_key = jax.random.split(key, 4)

        h = jax.random.normal(h_key, (num_spins, num_states)) * bias_scale

        j_raw = jax.random.normal(j_key, (num_spins, num_spins, num_states, num_states))
        j_raw = j_raw * coupling_scale
        j_sym = (j_raw + jnp.transpose(j_raw, (1, 0, 3, 2))) / 2

        if graph_type == 'full':
            adjacency = 1.0 - jnp.eye(num_spins)
        elif graph_type == 'random_edge_fixed_prob':
            adjacency = _adjacency(num_spins, edge_prob, edge_key)
        elif graph_type == 'random_edge_random_prob':
            prob = jax.random.uniform(p_key, minval=edge_prob_min, maxval=edge_prob_max)
            adjacency = _adjacency(num_spins, prob, edge_key)
        else:
            raise ValueError(
                f'Unknown graph_type {graph_type!r}; expected one of '
                "'full', 'random_edge_fixed_prob', 'random_edge_random_prob'."
            )

        J = j_sym * adjacency[:, :, None, None]

        return cls(h=h, J=J, num_spins=num_spins, num_states=num_states)


def random_mrfs(
    num_distributions: int,
    num_spins: int,
    num_states: int,
    coupling_scale: float,
    bias_scale: float,
    edge_prob: float,
    key: jax.Array,
    *,
    graph_type: str = 'random_edge_fixed_prob',
    edge_prob_min: float = 0.1,
    edge_prob_max: float = 0.9,
) -> MRF:
    make = partial(
        MRF.random,
        num_spins=num_spins,
        num_states=num_states,
        coupling_scale=coupling_scale,
        bias_scale=bias_scale,
        edge_prob=edge_prob,
        edge_prob_min=edge_prob_min,
        edge_prob_max=edge_prob_max,
        graph_type=graph_type,
    )
    keys = jax.random.split(key, num_distributions)
    return jax.vmap(lambda k: make(key=k))(keys)


def sample_batch(
    mrfs: MRF | list[MRF],
    num_samples: int,
    key: jax.Array,
) -> jax.Array:
    if isinstance(mrfs, list):
        mrfs = jax.tree.map(lambda *xs: jnp.stack(xs), *mrfs)

    return jax.vmap(MRF.sample, in_axes=(0, None, 0))(
        mrfs, num_samples, jax.random.split(key, mrfs.h.shape[0])
    )
