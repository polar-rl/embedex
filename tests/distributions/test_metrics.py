"""Distances between distributions, given as samples or as probability mass functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.distributions.domains.markov_random_fields import pmfs
from embedex.distributions.metrics import js_divergence, sliced_wasserstein_distance


@pytest.fixture
def samples(key) -> jax.Array:
    return jax.random.normal(key, (256, 2)) * jnp.array([1.0, 2.0])


class TestSlicedWasserstein:
    def test_is_zero_between_reorderings_of_one_set(self, samples, key):
        shuffled = samples[jax.random.permutation(key, len(samples))]

        assert float(sliced_wasserstein_distance(samples, shuffled, 64, key)) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_is_symmetric(self, samples, key):
        other = samples**2

        assert float(sliced_wasserstein_distance(samples, other, 64, key)) == pytest.approx(
            float(sliced_wasserstein_distance(other, samples, 64, key)), rel=1e-6
        )

    def test_measures_a_shift_in_one_dimension_exactly(self, key):
        xs = jax.random.normal(key, (128, 1))

        assert float(sliced_wasserstein_distance(xs, xs + 0.7, 8, key)) == pytest.approx(0.7)

    def test_averages_a_shift_over_directions(self, samples, key):
        # Projected onto a random unit direction u, a shift by b moves every point by b.u, and
        # E[(b.u)^2] = |b|^2 / 2 in two dimensions.
        shift = jnp.array([3.0, -4.0])

        distance = sliced_wasserstein_distance(samples, samples + shift, 20000, key)

        assert float(distance) == pytest.approx(5.0 / np.sqrt(2.0), rel=0.02)


class TestJensenShannon:
    P = jnp.array([0.5, 0.5, 0.0, 0.0])
    Q = jnp.array([0.0, 0.0, 0.5, 0.5])

    @pytest.mark.parametrize(
        'divergence', [js_divergence, pmfs.js_divergence], ids=['jax', 'numpy']
    )
    def test_is_zero_for_equal_and_log_2_for_disjoint_pmfs(self, divergence):
        assert float(divergence(self.P, self.P)) == pytest.approx(0.0, abs=1e-6)
        assert float(divergence(self.P, self.Q)) == pytest.approx(np.log(2.0), rel=1e-5)

    def test_both_implementations_agree(self, key):
        p, q = jax.random.dirichlet(key, jnp.ones(16), (2,))

        assert float(js_divergence(p, q)) == pytest.approx(
            pmfs.js_divergence(np.asarray(p), np.asarray(q)), rel=1e-4
        )
        assert float(js_divergence(p, q)) == pytest.approx(float(js_divergence(q, p)), rel=1e-6)
