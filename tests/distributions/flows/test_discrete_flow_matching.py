"""Discrete flow matching: the corruption process, the denoising loss, and the sampler that runs
the corruption backwards.

Hand-written denoisers pin the conventions down: with the exact posterior of the corruption
process, the sampler must reproduce the data distribution.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.distributions.flows.discrete_flow_matching import DiscreteFlowMatchingModel

NUM_VARIABLES, NUM_CLASSES = 3, 4


class FixedLogits(eqx.Module):
    """Predicts the same clean-data logits whatever it is shown."""

    logits: jax.Array

    def __call__(self, x_t, t, sideinfo=None):
        return self.logits


class ExactPosterior(eqx.Module):
    """The exact denoiser for data whose variables are independent, variable i taking class c
    with probability probs[i, c]: p(x_1 | x_t, t) is proportional to
    probs * (t [x_t == c] + (1 - t) / K)."""

    probs: jax.Array

    def __call__(self, x_t, t, sideinfo=None):
        likelihood = t * jax.nn.one_hot(x_t, NUM_CLASSES) + (1.0 - t) / NUM_CLASSES
        return jnp.log(self.probs * likelihood)


def dfm(denoiser, num_steps: int = 32) -> DiscreteFlowMatchingModel:
    return DiscreteFlowMatchingModel(denoiser, NUM_CLASSES, NUM_VARIABLES, num_steps)


@pytest.fixture
def probs(key) -> jax.Array:
    return jax.random.dirichlet(key, jnp.ones(NUM_CLASSES), (NUM_VARIABLES,))


@pytest.fixture
def x(key) -> jax.Array:
    return jax.random.randint(jax.random.fold_in(key, 1), (NUM_VARIABLES,), 0, NUM_CLASSES)


@pytest.mark.parametrize('t', [0.0, 0.3, 0.8, 1.0])
def test_corruption_keeps_each_variable_with_probability_t(t, probs, x, key):
    model = dfm(ExactPosterior(probs))

    noisy = jax.vmap(lambda k: model._corrupt(x, jnp.array(t), k))(jax.random.split(key, 20000))

    # A replaced variable can land on its own value by chance.
    agreement = np.asarray(noisy == x).mean(0)
    np.testing.assert_allclose(agreement, t + (1 - t) / NUM_CLASSES, atol=0.015)
    if t == 0.0:
        frequencies = np.asarray(jax.nn.one_hot(noisy, NUM_CLASSES)).mean(0)
        np.testing.assert_allclose(frequencies, 1 / NUM_CLASSES, atol=0.015)


def test_loss_is_the_cross_entropy_of_the_clean_data(probs, x, key):
    logits = jnp.log(probs)
    model = dfm(FixedLogits(logits))

    losses = jax.vmap(lambda k: model.loss(x.astype(jnp.float32), key=k)[0])(
        jax.random.split(key, 8)
    )

    expected = -jnp.sum(jax.nn.log_softmax(logits)[jnp.arange(NUM_VARIABLES), x])
    np.testing.assert_allclose(losses, expected, rtol=1e-6)


def test_log_prob_is_not_available(probs, x):
    # Reading the denoiser at t near 1 gave every state a probability of about 1, since a good
    # denoiser returns the clean state it is shown.
    with pytest.raises(NotImplementedError, match='samples'):
        dfm(ExactPosterior(probs)).log_prob(x.astype(jnp.float32))


def test_sampling_with_the_exact_denoiser_reproduces_the_data(probs, key):
    model = dfm(ExactPosterior(probs))

    samples = jax.vmap(lambda k: model.sample(None, None, k))(jax.random.split(key, 20000))

    frequencies = np.asarray(jax.nn.one_hot(samples.astype(jnp.int32), NUM_CLASSES)).mean(0)
    np.testing.assert_allclose(frequencies, probs, atol=0.015)


def test_deterministic_generation_ends_at_the_most_likely_state(probs):
    model = dfm(FixedLogits(jnp.log(probs)))

    generated, _ = model._inverse(jnp.zeros(NUM_VARIABLES), 0.0)

    np.testing.assert_array_equal(generated, jnp.argmax(probs, -1))


@pytest.mark.parametrize('denoiser', ['mlp', 'attention'])
def test_denoisers_read_the_tokens_the_time_and_the_embedding(denoiser, make_model, key):
    net = make_model(flow_type='dfm', input_dim=4, num_classes=3, dfm_denoiser=denoiser)
    net = net.generator.denoiser_net
    x_t, t = jnp.array([0, 1, 2, 1]), jnp.array(0.4)
    sideinfo = jax.random.normal(key, (8,))

    logits = net(x_t, t, sideinfo)

    assert logits.shape == (4, 3)
    assert not np.allclose(net(x_t.at[0].set(2), t, sideinfo), logits)
    assert not np.allclose(net(x_t, jnp.array(0.9), sideinfo), logits)
    assert not np.allclose(net(x_t, t, -sideinfo), logits)
