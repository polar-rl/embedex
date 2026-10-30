"""Reconstruction and operator metrics: each embeds the first half of a function's samples and
scores the prediction on the second half."""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.functions import metrics
from embedex.functions.families.antiderivatives import PolynomialAntiderivativeGenerator
from embedex.functions.families.derivatives import PolynomialDerivativeGenerator
from embedex.functions.generator import Generator
from embedex.functions.metrics import (
    evaluate_operator_rel_l2,
    evaluate_operator_rel_max,
    evaluate_operator_rmse,
    evaluate_rel_l2,
    evaluate_rel_max,
    evaluate_rmse,
)

BATCH, SAMPLES = 16, 64

# Two functions with four samples each: two to embed, then two to score.
XS = jnp.zeros((2, 4, 1))
YS = jnp.array([[1.0, 1.0, 2.0, 4.0], [0.0, 0.0, 0.0, -1.0]])[..., jnp.newaxis]
# Predicting the mean of the embedded half, the held-out errors are (1, 3) and (0, -1).
EXPECTED = {
    'rmse': np.sqrt((5.0 + 0.5) / 2),  # root of the mean squared error over all functions
    'rel_l2': (np.sqrt(10 / 20) + 1 / 1) / 2,
    'rel_max': (3 / 4 + 1 / 1) / 2,
}


class MeanOfContext(eqx.Module):
    """Predicts the mean value of the embedded samples everywhere."""

    def embed(self, xs: jax.Array, ys: jax.Array) -> jax.Array:
        return jnp.mean(ys, axis=0)

    def eval(self, xs: jax.Array, z: jax.Array) -> jax.Array:
        return jnp.broadcast_to(z, (xs.shape[0], z.shape[0]))


class Task(Generator):
    """An operator task whose input is either an embedding of the extras or the extras
    themselves, and whose target is the function's own values."""

    def __init__(self, embed_extras: bool) -> None:
        self.embed_extras = embed_extras

    def sample(self, batch_size, num_samples, key):
        raise NotImplementedError

    def operator_input_size(self, cfg) -> int:
        raise NotImplementedError

    def operator_input(self, embedding_network, xs, ys, extra):
        return embedding_network.embed(xs, extra) if self.embed_extras else extra

    def operator_ground_truth(self, xs, ys, extra):
        return ys


class PolynomialFit(eqx.Module):
    """Embeds samples of a polynomial as its least-squares coefficients, which recovers any
    polynomial of degree at most `degree` exactly."""

    degree: int = eqx.field(static=True)

    def features(self, xs: jax.Array) -> jax.Array:
        return xs ** jnp.arange(self.degree + 1)

    def embed(self, xs: jax.Array, ys: jax.Array) -> jax.Array:
        return jnp.linalg.lstsq(self.features(xs), ys[:, 0])[0]

    def eval(self, xs: jax.Array, coeffs: jax.Array) -> jax.Array:
        return (self.features(xs) @ coeffs)[:, jnp.newaxis]


class Differentiate(eqx.Module):
    """Maps polynomial coefficients to those of the derivative."""

    def __call__(self, coeffs: jax.Array) -> jax.Array:
        return jnp.append(coeffs[1:] * jnp.arange(1, coeffs.shape[0]), 0.0)


class Integrate(eqx.Module):
    """Maps the coefficients of f' and one sample (x0, f(x0)) to the coefficients of f."""

    def __call__(self, source: jax.Array) -> jax.Array:
        coeffs, x0, y0 = source[:-2], source[-2], source[-1]
        integral = jnp.append(0.0, coeffs[:-1] / jnp.arange(1, coeffs.shape[0]))
        return integral.at[0].set(y0 - jnp.polyval(integral[::-1], x0))


class Solution(eqx.Module):
    """Decodes an ODE's parameters into its exact solution."""

    solve: Callable = eqx.field(static=True)

    def eval(self, ts: jax.Array, params: jax.Array) -> jax.Array:
        return self.solve(ts[:, 0], params)


def scores(embedding, xs, ys) -> dict[str, float]:
    return {
        'rmse': float(evaluate_rmse(embedding, xs, ys)),
        'rel_l2': float(evaluate_rel_l2(embedding, xs, ys)),
        'rel_max': float(evaluate_rel_max(embedding, xs, ys)),
    }


def operator_scores(embedding, operator, generator, xs, ys, extras) -> dict[str, float]:
    args = (embedding, operator, generator, xs, ys, extras)
    return {
        'rmse': float(evaluate_operator_rmse(*args)),
        'rel_l2': float(evaluate_operator_rel_l2(*args)),
        'rel_max': float(evaluate_operator_rel_max(*args)),
    }


def test_reconstruction_scores_the_held_out_half():
    assert scores(MeanOfContext(), XS, YS) == pytest.approx(EXPECTED)


def test_operator_embeds_per_sample_extras_from_the_held_out_half_only():
    # The extras are the values themselves, so the identity operator reproduces the embedding.
    result = operator_scores(MeanOfContext(), eqx.nn.Identity(), Task(True), XS, YS, YS)

    assert result == pytest.approx(EXPECTED)


def test_operator_receives_per_function_extras_whole():
    # Constant extras of 1 and 0 make the same predictions as the mean of each embedded half.
    params = jnp.array([[1.0], [0.0]])

    result = operator_scores(MeanOfContext(), eqx.nn.Identity(), Task(False), XS, YS, params)

    assert result == pytest.approx(EXPECTED)


def test_relative_errors_survive_a_zero_target():
    ones, zeros = jnp.ones((4, 1)), jnp.zeros((4, 1))

    assert jnp.isfinite(metrics._rel_l2(ones, zeros))
    assert jnp.isfinite(metrics._rel_max(ones, zeros))


def test_exact_polynomial_embedding_and_differentiation_score_zero(key):
    # Fitting the coefficients and differentiating them is a perfect embedding and operator.
    fit = PolynomialFit(3)
    generator = PolynomialDerivativeGenerator((-4.0, 4.0), (-5.0, 5.0), 1, 3)
    xs, ys, dys = generator.sample(BATCH, SAMPLES, key)

    reconstruction = scores(fit, xs, ys)
    operator = operator_scores(fit, Differentiate(), generator, xs, ys, dys)

    for result in (reconstruction, operator):
        assert result['rel_l2'] < 1e-4 and result['rel_max'] < 1e-4
        assert result['rmse'] < 1e-4 * float(jnp.sqrt(jnp.mean(ys**2)))


def test_exact_polynomial_integration_scores_zero(key):
    # The derivative fixes the function only up to a constant; the operator's input carries one
    # sample of the function to pin it down.
    fit = PolynomialFit(3)
    generator = PolynomialAntiderivativeGenerator((-4.0, 4.0), (-5.0, 5.0), 0, 3)
    xs, ys, dys = generator.sample(BATCH, SAMPLES, key)

    result = operator_scores(fit, Integrate(), generator, xs, ys, dys)

    assert result['rel_l2'] < 1e-4 and result['rel_max'] < 1e-4
    assert result['rmse'] < 1e-4 * float(jnp.sqrt(jnp.mean(ys**2)))


def test_exact_solution_operator_scores_zero(ode_family, key):
    generator = ode_family.generator()
    ts, ys, params = generator.sample(BATCH, SAMPLES, key)
    solution = Solution(partial(ode_family.solution, generator))

    result = operator_scores(solution, eqx.nn.Identity(), generator, ts, ys, params)

    assert result == pytest.approx({'rmse': 0.0, 'rel_l2': 0.0, 'rel_max': 0.0}, abs=1e-4)
