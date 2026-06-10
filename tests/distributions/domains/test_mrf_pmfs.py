import numpy as np
import pytest

from embedex.distributions.domains.markov_random_fields.pmfs import (
    empirical_pmf,
    js_divergence,
    mean_field_pmf,
    pmf_metrics,
    state_indices,
)


def enumerate_states(num_spins: int, num_states: int) -> np.ndarray:
    indices = np.arange(num_states**num_spins)
    powers = num_states ** np.arange(num_spins)
    return (indices[:, None] // powers[None, :]) % num_states


def test_state_indices_inverts_state_enumeration():
    states = enumerate_states(4, 3)
    np.testing.assert_array_equal(state_indices(states, 4, 3), np.arange(3**4))


def test_empirical_pmf_counts_and_drops_non_finite():
    samples = np.array([[0, 0], [1, 0], [1, 0], [np.nan, 0]])
    pmf = empirical_pmf(samples, 2, 2)

    np.testing.assert_allclose(pmf, [1 / 3, 2 / 3, 0.0, 0.0])


def test_mean_field_pmf_is_product_of_marginals():
    rng = np.random.default_rng(0)
    samples = rng.integers(0, 2, size=(500, 3)).astype(float)
    states = enumerate_states(3, 2)

    pmf = mean_field_pmf(samples, states, 2)
    marginals = np.stack([np.mean(samples[:, i : i + 1] == [0, 1], axis=0) for i in range(3)])
    expected = [np.prod(marginals[np.arange(3), s]) for s in states]

    np.testing.assert_allclose(pmf, expected)
    assert pmf.sum() == pytest.approx(1.0)


def test_js_divergence_bounds_and_zeros():
    p = np.array([0.5, 0.5, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.5, 0.5])

    assert js_divergence(p, p) == pytest.approx(0.0)
    assert js_divergence(p, q) == pytest.approx(np.log(2.0))
    assert np.isfinite(js_divergence(p, np.array([0.25, 0.25, 0.25, 0.25])))


def test_pmf_metrics_identical_pmfs():
    states = enumerate_states(3, 2)
    pmf = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])

    metrics = pmf_metrics(pmf, pmf, states, 2, top_k=4)

    assert metrics['jsd'] == pytest.approx(0.0)
    assert metrics['class_mae'] == pytest.approx(0.0)
    assert metrics['agree_mae'] == pytest.approx(0.0)
    assert metrics['mass_outside'] == pytest.approx(0.0, abs=0.01)
    assert metrics['mode_coverage'] == 1.0


def test_pmf_metrics_flags_missed_mode():
    states = enumerate_states(3, 2)
    true_pmf = np.zeros(8)
    true_pmf[[0, 7]] = 0.5
    model_pmf = np.zeros(8)
    model_pmf[0] = 0.5
    model_pmf[1:7] = 0.5 / 6

    metrics = pmf_metrics(true_pmf, model_pmf, states, 2, top_k=2)

    assert metrics['mass_outside'] == pytest.approx(0.5)
    assert metrics['mode_coverage'] == pytest.approx(0.5)
    assert metrics['jsd'] > 0.1
