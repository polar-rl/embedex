"""`FunctionEmbedding`: the staged set encoder over (x, y) samples and the decoder that reads it."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.functions.builder import create_embedding_network
from embedex.functions.embedding import FunctionEmbedding
from embedex.functions.families.derivatives import PolynomialGenerator

EMBEDDING_SIZE, SAMPLES = 8, 32  # matches the `embedding_network` fixture


@pytest.fixture
def samples(key) -> tuple[jax.Array, jax.Array]:
    """Samples of two different polynomials, each of shape (2, SAMPLES, 1)."""

    xs, ys, _ = PolynomialGenerator((-4.0, 4.0), (-5.0, 5.0), 1, 3).sample(2, SAMPLES, key)
    return xs, ys


class MeanOfStage(eqx.Module):
    """A stage block whose per-sample output is y, plus the previous stage's summary if any."""

    def __call__(self, xs: jax.Array, ys: jax.Array, pool: jax.Array | None) -> jax.Array:
        return ys if pool is None else ys + pool


class Constant(eqx.Module):
    """A decoder that predicts the embedding itself at every input."""

    def __call__(self, x: jax.Array, z: jax.Array) -> jax.Array:
        return z


def squared_error(y_pred: jax.Array, y_target: jax.Array) -> jax.Array:
    return jnp.mean((y_pred - y_target) ** 2)


def test_embedding_is_a_finite_vector(embedding_network, samples):
    xs, ys = samples

    z = embedding_network.embed(xs[0], ys[0])

    assert z.shape == (EMBEDDING_SIZE,)
    assert jnp.all(jnp.isfinite(z))


def test_embedding_ignores_the_order_of_the_samples(embedding_network, samples, key):
    xs, ys = samples
    order = jax.random.permutation(key, SAMPLES)

    np.testing.assert_allclose(
        embedding_network.embed(xs[0][order], ys[0][order]),
        embedding_network.embed(xs[0], ys[0]),
        atol=1e-5,
    )


def test_different_functions_get_different_embeddings(embedding_network, samples):
    xs, ys = samples

    assert not np.allclose(
        embedding_network.embed(xs[0], ys[0]), embedding_network.embed(xs[1], ys[1])
    )


def test_eval_predicts_one_value_per_input(embedding_network, samples):
    xs, ys = samples

    predictions = embedding_network.eval(xs[1], embedding_network.embed(xs[0], ys[0]))

    assert predictions.shape == (SAMPLES, 1)


def test_each_stage_sees_the_previous_stages_summary():
    # Stage 0 averages y; stage 1 adds that average to every y and averages again.
    network = FunctionEmbedding([MeanOfStage(), MeanOfStage()], Constant(), squared_error)
    ys = jnp.array([[1.0], [2.0], [6.0]])

    np.testing.assert_allclose(network.embed(jnp.zeros((3, 1)), ys), [6.0])


def test_later_stage_blocks_use_the_summary(stage_block, make_config, key):
    # The embedding is the last stage's pool, so the first stage only matters through it.
    cfg = make_config(stage_block=stage_block)
    network = create_embedding_network(cfg, key)
    other = create_embedding_network(cfg, jax.random.key(7))
    swapped = eqx.tree_at(lambda m: m.embedding_blocks[0], network, other.embedding_blocks[0])
    xs, ys, _ = PolynomialGenerator((-4.0, 4.0), (-5.0, 5.0), 1, 3).sample(1, SAMPLES, key)

    assert not np.allclose(network.embed(xs[0], ys[0]), swapped.embed(xs[0], ys[0]))


@pytest.mark.parametrize('seed', range(5))
def test_loss_scores_held_out_samples_only(seed):
    # The decoder repeats the one embedded value, so the error is |0 - 2| whichever of the two
    # samples is held out. Embedding both would predict their mean (error 1), and scoring the
    # embedded sample would give 0.
    network = FunctionEmbedding([MeanOfStage()], Constant(), squared_error)

    loss = network.loss(jnp.zeros((2, 1)), jnp.array([[0.0], [2.0]]), jax.random.key(seed))

    assert float(loss) == pytest.approx(4.0)


def test_every_parameter_gets_a_gradient(stage_block, decoder, make_config, samples, key):
    network = create_embedding_network(make_config(stage_block=stage_block, decoder=decoder), key)
    xs, ys = samples

    loss, grads = eqx.filter_value_and_grad(lambda n: n.loss(xs[0], ys[0], key))(network)

    assert jnp.isfinite(loss) and loss >= 0.0
    for path, grad in jax.tree_util.tree_leaves_with_path(grads):
        assert jnp.any(grad != 0.0), jax.tree_util.keystr(path)
