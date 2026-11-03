"""Donuts: rings in the plane, sampled at a uniform angle and a uniform radial offset, and the
baselines the paper compares the embedding against."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.distributions.domains.donuts.baselines import KDEBaseline, OracleDonutFit
from embedex.distributions.domains.donuts.donut import (
    Donut,
    donut_moments,
    random_donuts,
    sample_batch,
)

MEAN, RADIUS, WIDTH = (1.0, -2.0), 1.5, 0.4


@pytest.fixture
def ring() -> Donut:
    return Donut(mean=jnp.array(MEAN), radius=jnp.float32(RADIUS), width=jnp.float32(WIDTH))


@pytest.fixture
def samples(ring, key) -> np.ndarray:
    return np.asarray(ring.sample(20000, key))


def integrate(log_density, half_width: float = 6.0, num_points: int = 481) -> float:
    """The integral of a 2D density over a square around `MEAN`, evaluated in chunks."""

    grid = jnp.linspace(-half_width, half_width, num_points)
    points = jnp.stack(jnp.meshgrid(grid, grid), -1).reshape(num_points, num_points, 2)
    densities = jax.lax.map(lambda row: jnp.exp(log_density(row + jnp.array(MEAN))), points)
    return float(jnp.sum(densities) * (grid[1] - grid[0]) ** 2)


class TestDonut:
    def test_samples_fill_the_ring_uniformly(self, samples):
        offsets = samples - np.array(MEAN)
        distances = np.linalg.norm(offsets, axis=-1)
        angles = np.arctan2(offsets[:, 1], offsets[:, 0])

        inner, outer = RADIUS * (1 - WIDTH / 2), RADIUS * (1 + WIDTH / 2)
        assert inner - 1e-5 <= distances.min() and distances.max() <= outer + 1e-5
        radial = np.histogram(distances, bins=8, range=(inner, outer))[0] / len(samples)
        angular = np.histogram(angles, bins=16, range=(-np.pi, np.pi))[0] / len(samples)
        np.testing.assert_allclose(radial, 1 / 8, atol=0.01)
        np.testing.assert_allclose(angular, 1 / 16, atol=0.006)

    def test_moments_recover_the_donut(self, samples):
        center, radius, width = donut_moments(jnp.asarray(samples))

        np.testing.assert_allclose(center, MEAN, atol=0.02)
        assert float(radius) == pytest.approx(RADIUS, rel=0.01)
        assert float(width) == pytest.approx(WIDTH, rel=0.02)

    def test_parameters_are_the_center_radius_and_width(self, ring):
        np.testing.assert_allclose(ring.params, [*MEAN, RADIUS, WIDTH])

    def test_random_donuts_stay_in_their_ranges(self, domains, key):
        data = domains['donuts'].config().data

        donuts = random_donuts(4096, data.mean_std, data.radius_range, data.width_range, key)

        assert (
            data.radius_range[0]
            <= donuts.radius.min()
            <= donuts.radius.max()
            <= data.radius_range[1]
        )
        assert (
            data.width_range[0] <= donuts.width.min() <= donuts.width.max() <= data.width_range[1]
        )
        np.testing.assert_allclose(donuts.mean.std(0), data.mean_std, rtol=0.05)

    def test_a_batch_samples_each_donut_on_its_own(self, ring, key):
        other = Donut(mean=jnp.array([-3.0, 0.0]), radius=jnp.float32(0.5), width=jnp.float32(0.1))
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), ring, other)

        batch = sample_batch([ring, other], 512, key)

        assert batch.shape == (2, 512, 2)
        np.testing.assert_array_equal(batch, sample_batch(stacked, 512, key))
        for donut, xs in zip([ring, other], batch, strict=True):
            np.testing.assert_allclose(donut_moments(xs)[1], donut.radius, rtol=0.02)


class TestBaselines:
    @pytest.fixture
    def fit_samples(self, ring, key) -> jax.Array:
        return ring.sample(2048, jax.random.fold_in(key, 1))

    @pytest.fixture
    def kde(self, fit_samples) -> KDEBaseline:
        return KDEBaseline.fit(fit_samples)

    @pytest.fixture
    def oracle(self, fit_samples) -> OracleDonutFit:
        return OracleDonutFit.fit(fit_samples)

    @pytest.mark.parametrize('baseline', ['kde', 'oracle'])
    def test_density_integrates_to_one(self, baseline, request):
        model = request.getfixturevalue(baseline)

        assert integrate(model.log_prob) == pytest.approx(1.0, abs=0.01)

    def test_kde_samples_are_data_points_plus_kernel_noise(self, kde, fit_samples, key):
        xs = kde.sample(50000, key)

        np.testing.assert_allclose(xs.mean(0), fit_samples.mean(0), atol=0.02)
        expected = jnp.cov(fit_samples.T) + kde.kernel_cov
        np.testing.assert_allclose(jnp.cov(xs.T), expected, rtol=0.03, atol=0.02)

    def test_kde_bandwidth_follows_scotts_rule(self, kde, fit_samples):
        # In d = 2 dimensions the kernel is the data covariance scaled by n^(-2 / (d + 4)).
        expected = jnp.cov(fit_samples.T) * len(fit_samples) ** (-1 / 3)

        np.testing.assert_allclose(kde.kernel_cov, expected, rtol=1e-4)

    def test_oracle_recovers_the_donut(self, oracle):
        np.testing.assert_allclose(oracle.donut.mean, MEAN, atol=0.08)
        assert float(oracle.donut.radius) == pytest.approx(RADIUS, rel=0.01)
        assert float(oracle.donut.width) == pytest.approx(WIDTH, rel=0.05)

    def test_oracle_samples_its_fitted_donut(self, oracle, key):
        center, radius, width = donut_moments(oracle.sample(20000, key))

        np.testing.assert_allclose(center, oracle.donut.mean, atol=0.02)
        assert float(radius) == pytest.approx(float(oracle.donut.radius), rel=0.01)
        assert float(width) == pytest.approx(float(oracle.donut.width), rel=0.02)

    def test_oracle_density_is_close_to_the_true_density(self, oracle, samples):
        # The true density is uniform over the ring's radial band, spread over a circle of
        # circumference 2 pi d: 1 / (2 pi d * radius * width). The oracle smooths its edges, so
        # it scores held-out samples slightly below the truth.
        distances = np.linalg.norm(samples - np.array(MEAN), axis=-1)
        true_log_density = -np.log(2 * np.pi * distances * RADIUS * WIDTH)

        gap = np.mean(true_log_density) - float(oracle.log_prob(samples).mean())
        assert 0.0 < gap < 0.1

    def test_oracle_scores_held_out_donut_samples_above_the_kde(self, kde, oracle, samples):
        assert float(oracle.log_prob(samples).mean()) > float(kde.log_prob(samples).mean())
