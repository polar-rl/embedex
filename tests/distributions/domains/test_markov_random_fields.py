import jax
import numpy as np
import pytest

from embedex.distributions.domains.markov_random_fields.mrf import (
    MRF,
    all_states,
    random_mrfs,
    sample_batch,
)
from embedex.distributions.domains.markov_random_fields.pmfs import empirical_pmf
from embedex.distributions.metrics import js_divergence


def make_mrf(
    num_spins: int = 4,
    num_states: int = 2,
    graph_type: str = 'random_edge_fixed_prob',
    edge_prob: float = 0.5,
    seed: int = 0,
) -> MRF:
    return MRF.random(
        num_spins=num_spins,
        num_states=num_states,
        coupling_scale=1.0,
        bias_scale=0.5,
        edge_prob=edge_prob,
        edge_prob_min=0.1,
        edge_prob_max=0.9,
        graph_type=graph_type,
        key=jax.random.key(seed),
    )


def num_edges(mrf: MRF) -> int:
    strength = np.linalg.norm(np.array(mrf.J), axis=(2, 3))
    return int((np.triu(strength, k=1) > 0).sum())


@pytest.mark.parametrize(('num_spins', 'num_states'), [(4, 2), (3, 3)])
def test_all_states_enumerates_every_configuration(num_spins, num_states):
    states = np.array(all_states(num_spins, num_states))

    assert states.shape == (num_states**num_spins, num_spins)
    assert states.min() == 0
    assert states.max() == num_states - 1
    assert np.unique(states, axis=0).shape[0] == num_states**num_spins


@pytest.mark.parametrize(('num_spins', 'num_states'), [(3, 2), (3, 3)])
def test_log_pmf_matches_brute_force(num_spins, num_states):
    mrf = make_mrf(num_spins, num_states)
    h, J = np.array(mrf.h), np.array(mrf.J)
    states = np.array(mrf.states)

    energies = []
    for x in states:
        field = sum(h[i, x[i]] for i in range(num_spins))
        coupling = 0.5 * sum(
            J[i, j, x[i], x[j]] for i in range(num_spins) for j in range(num_spins)
        )
        energies.append(field + coupling)
    energies = np.array(energies)
    shifted = energies - energies.max()
    expected = shifted - np.log(np.exp(shifted).sum())

    np.testing.assert_allclose(np.array(mrf.log_pmf()), expected, atol=1e-5)


def test_coupling_tables_symmetric_with_zero_diagonal():
    mrf = make_mrf(num_spins=5, num_states=3)
    J = np.array(mrf.J)

    np.testing.assert_allclose(J, np.transpose(J, (1, 0, 3, 2)), atol=1e-6)
    np.testing.assert_allclose(J[np.arange(5), np.arange(5)], 0.0)


def test_graph_types_control_edge_sets():
    n = 6
    max_edges = n * (n - 1) // 2

    assert num_edges(make_mrf(num_spins=n, graph_type='full')) == max_edges
    assert num_edges(make_mrf(num_spins=n, edge_prob=0.0)) == 0
    assert num_edges(make_mrf(num_spins=n, edge_prob=1.0)) == max_edges


@pytest.mark.parametrize(('num_spins', 'num_states'), [(4, 2), (3, 3)])
def test_marginals_are_consistent(num_spins, num_states):
    mrf = make_mrf(num_spins, num_states)
    pmf = np.array(mrf.pmf())
    states = np.array(mrf.states)

    single = np.array(mrf.single_marginals())
    assert single.shape == (num_spins, num_states)
    np.testing.assert_allclose(single.sum(axis=1), 1.0, atol=1e-5)
    for i in range(num_spins):
        for c in range(num_states):
            np.testing.assert_allclose(single[i, c], pmf[states[:, i] == c].sum(), atol=1e-5)

    pairwise = np.array(mrf.pairwise_marginals())
    assert pairwise.shape == (num_spins, num_spins, num_states, num_states)
    np.testing.assert_allclose(pairwise.sum(axis=(2, 3)), 1.0, atol=1e-5)
    # Marginalizing node j out of the pairwise table recovers node i's marginals.
    np.testing.assert_allclose(
        pairwise.sum(axis=3),
        np.broadcast_to(single[:, None, :], (num_spins, num_spins, num_states)),
        atol=1e-5,
    )

    agreement = np.array(mrf.agreement_marginals())
    assert agreement.shape == (num_spins, num_spins)
    np.testing.assert_allclose(agreement, agreement.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(agreement), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.einsum('ijaa->ij', pairwise), agreement, atol=1e-5)


def test_sampling_matches_marginals():
    num_samples = 20000
    mrf = make_mrf(num_spins=4, num_states=3, seed=1)
    samples = np.array(mrf.sample(num_samples, key=jax.random.key(2)))

    assert samples.shape == (num_samples, 4)
    np.testing.assert_allclose(samples, np.round(samples))  # integer-valued floats
    assert samples.min() >= 0
    assert samples.max() <= 2

    single = np.array(mrf.single_marginals())
    for c in range(3):
        np.testing.assert_allclose((samples == c).mean(axis=0), single[:, c], atol=0.02)


def test_random_mrfs_and_sample_batch_shapes():
    num_distributions, num_spins, num_states, num_samples = 8, 4, 2, 16
    mrfs = random_mrfs(num_distributions, num_spins, num_states, 1.0, 0.5, 0.5, jax.random.key(0))

    assert mrfs.h.shape == (num_distributions, num_spins, num_states)
    assert mrfs.J.shape == (num_distributions, num_spins, num_spins, num_states, num_states)
    assert mrfs.num_spins == num_spins
    assert mrfs.num_states == num_states
    assert not np.allclose(np.array(mrfs.J[0]), np.array(mrfs.J[1]))

    samples = sample_batch(mrfs, num_samples, jax.random.key(1))
    assert samples.shape == (num_distributions, num_samples, num_spins)


def test_empirical_pmf_recovers_true_pmf():
    mrf = make_mrf(num_spins=4, num_states=2, seed=3)
    samples = mrf.sample(50000, key=jax.random.key(4))
    pmf = empirical_pmf(samples, 4, 2)

    assert float(js_divergence(mrf.pmf(), pmf)) < 0.01
