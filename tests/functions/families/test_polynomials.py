"""Random polynomials, and the differential and antidifferential operators on them."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.functions.builder import create_embedding_network
from embedex.functions.families import derivatives
from embedex.functions.families.antiderivatives import PolynomialAntiderivativeGenerator
from embedex.functions.families.derivatives import (
    PolynomialDerivativeGenerator,
    PolynomialGenerator,
)

BATCH, SAMPLES = 16, 32


def fit(x: jax.Array, y: jax.Array, degree: int) -> tuple[np.ndarray, float]:
    """Least-squares polynomial coefficients (highest degree first, as numpy orders them) and
    the largest residual relative to the largest value."""

    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    coeffs = np.polyfit(x, y, degree)
    return coeffs, np.abs(y - np.polyval(coeffs, x)).max() / np.abs(y).max()


def degree(x: jax.Array, y: jax.Array) -> int:
    """The lowest degree of a polynomial through the samples, up to float32 precision."""

    return next(d for d in range(5) if fit(x, y, d)[1] < 1e-4)


def test_evaluate_gives_values_and_derivatives():
    coeffs = jnp.array([1.0, -2.0, 0.5, 3.0])  # 1 - 2x + x^2 / 2 + 3x^3
    x = jnp.linspace(-4.0, 4.0, 9)

    y, dy = PolynomialGenerator((-4.0, 4.0), (-5.0, 5.0), 0, 3).evaluate(x, coeffs)

    np.testing.assert_allclose(y, np.polyval(coeffs[::-1], x), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dy, np.polyval(np.polyder(coeffs[::-1]), x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('expected', range(4))
def test_sample_degree_gives_polynomials_of_exactly_that_degree(expected, key):
    generator = PolynomialGenerator((-4.0, 4.0), (-5.0, 5.0), 0, 3)
    xs, ys, dys = generator.sample_degree(expected, BATCH, SAMPLES, key)

    for x, y, dy in zip(xs[..., 0], ys[..., 0], dys[..., 0], strict=True):
        coeffs, _ = fit(x, y, expected)

        assert degree(x, y) == expected
        np.testing.assert_allclose(np.polyval(np.polyder(coeffs), x), dy, rtol=1e-3, atol=1e-3)


def test_sample_draws_every_degree_between_the_bounds(key):
    xs, ys, _ = PolynomialGenerator((-4.0, 4.0), (-5.0, 5.0), 1, 2).sample(128, SAMPLES, key)

    degrees = {degree(x, y) for x, y in zip(xs[..., 0], ys[..., 0], strict=True)}

    assert degrees == {1, 2}


def test_inputs_cover_the_whole_range(key):
    xs, _, _ = PolynomialGenerator((-1.0, 3.0), (-5.0, 5.0), 0, 3).sample(64, SAMPLES, key)

    assert float(xs.min()) == pytest.approx(-1.0, abs=0.05)
    assert float(xs.max()) == pytest.approx(3.0, abs=0.05)


class TestOperatorTasks:
    @pytest.fixture
    def setup(self, families, make_config, key):
        family = families['derivatives']
        cfg = make_config(family)
        xs, ys, dys = family.generator().sample(2, SAMPLES, key)
        return cfg, create_embedding_network(cfg, key), xs[0], ys[0], dys[0]

    def test_differentiation_maps_an_embedding_to_the_derivative(self, setup):
        cfg, embedding, x, y, dy = setup
        generator = PolynomialDerivativeGenerator((-4.0, 4.0), (-5.0, 5.0), 1, 3)

        np.testing.assert_array_equal(
            generator.operator_input(embedding, x, y, dy), embedding.embed(x, y)
        )
        np.testing.assert_array_equal(generator.operator_ground_truth(x, y, dy), dy)
        assert generator.operator_input_size(cfg) == cfg.model.embedding.embedding_size

    def test_integration_maps_an_embedding_of_the_derivative_and_one_sample_to_the_function(
        self, setup
    ):
        cfg, embedding, x, y, dy = setup
        generator = PolynomialAntiderivativeGenerator((-4.0, 4.0), (-5.0, 5.0), 0, 3)

        source = generator.operator_input(embedding, x, y, dy)

        np.testing.assert_array_equal(source[:-2], embedding.embed(x, dy))
        np.testing.assert_array_equal(source[-2:], [x[0, 0], y[0, 0]])
        np.testing.assert_array_equal(generator.operator_ground_truth(x, y, dy), y)
        assert generator.operator_input_size(cfg) == cfg.model.embedding.embedding_size + 2


def test_derivatives_train_the_operator_without_constants(
    families, make_config, tmp_path, monkeypatch, key
):
    # A constant's derivative is zero everywhere, which breaks the relative L2 loss, so the
    # operator gets its own generator that starts at `operator_min_degree`.
    monkeypatch.chdir(tmp_path)  # the experiment tracker writes its offline files here
    cfg = make_config(families['derivatives'])
    min_degrees = {}

    def run_training_pipeline(cfg, generator, operator_generator, *evaluations):
        min_degrees.update(embedding=generator.min_degree, operator=operator_generator.min_degree)
        return create_embedding_network(cfg, key), None

    monkeypatch.setattr(derivatives, 'run_training_pipeline', run_training_pipeline)
    derivatives.main(cfg)

    assert min_degrees == {'embedding': 0, 'operator': 1}
