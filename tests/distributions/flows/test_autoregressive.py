"""The autoregressive model: an exactly normalized categorical distribution whose sampler,
with its key-value cache, agrees with the network's full causal pass."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.distributions.flows.autoregressive import AutoregressiveModel


@pytest.fixture(params=['none', 'learned'], ids=['moments', 'relational'])
def make_ar(request, make_model, key):
    """An autoregressive model over `num_variables` variables, and an embedding of random states
    to condition it on, with and without the relational stream."""

    def _make(num_variables: int, num_classes: int) -> tuple[AutoregressiveModel, jax.Array]:
        model = make_model(
            flow_type='ar',
            input_dim=num_variables,
            num_classes=num_classes,
            relational_mode=request.param,
        )
        states = jax.random.randint(key, (32, num_variables), 0, num_classes)
        z, _ = model.embed(states.astype(jnp.float32))
        return model.generator, z

    return _make


def all_states(num_variables: int, num_classes: int) -> jax.Array:
    states = itertools.product(range(num_classes), repeat=num_variables)
    return jnp.array(list(states), dtype=jnp.float32)


def test_probabilities_of_all_states_sum_to_one(make_ar):
    model, z = make_ar(4, 3)

    log_probs, _, _ = jax.vmap(lambda x: model.log_prob(x, z))(all_states(4, 3))

    assert float(jnp.sum(jnp.exp(log_probs))) == pytest.approx(1.0, abs=1e-4)


def test_each_variable_is_predicted_from_the_earlier_ones_only(make_ar, key):
    model, z = make_ar(5, 3)
    x = jax.random.randint(key, (5,), 0, 3)

    logits = model.net(x, z)

    for j in range(5):
        changed = model.net(x.at[j].set((x[j] + 1) % 3), z)
        np.testing.assert_allclose(changed[: j + 1], logits[: j + 1], atol=1e-6)
        assert j == 4 or not np.allclose(changed[j + 1 :], logits[j + 1 :])


def test_loss_is_the_negative_log_probability(make_ar, key):
    model, z = make_ar(4, 3)
    x = jax.random.randint(key, (4,), 0, 3).astype(jnp.float32)

    loss, _, _ = model.loss(x, z)
    log_prob, _, _ = model.log_prob(x, z)

    assert float(loss) == pytest.approx(-float(log_prob))


def naive_sample(model: AutoregressiveModel, z: jax.Array, key: jax.Array) -> jax.Array:
    """Samples one variable at a time, rerunning the full network on the prefix each time."""

    keys = jax.random.split(key, model.num_variables)

    def sample_variable(i, x):
        return x.at[i].set(jax.random.categorical(keys[i], model.net(x, z)[i]))

    x = jnp.zeros(model.num_variables, dtype=jnp.int32)
    return jax.lax.fori_loop(0, model.num_variables, sample_variable, x).astype(jnp.float32)


def test_cached_sampling_matches_rerunning_the_network(make_ar, key):
    model, z = make_ar(7, 2)
    keys = jax.random.split(key, 2000)

    cached = jax.vmap(lambda k: model.sample(z, None, k))(keys)
    naive = jax.vmap(lambda k: naive_sample(model, z, k))(keys)

    # Both run the blocks in bfloat16, where the two orders of computation round differently, so
    # a draw right on the boundary between two classes can go either way (about 1 in 10^4 does).
    # Attending to one cache slot too many flips about 1 in 500.
    assert float(jnp.mean(cached != naive)) < 1e-3


def test_each_variable_reads_its_own_row_of_the_relational_block(make_model, key):
    # The embedding ends with one row of pairwise values per variable, and the row of variable j
    # joins the model at variable j's position, so it cannot affect the variables before it.
    model = make_model(flow_type='ar', input_dim=5, num_classes=3, relational_mode='learned')
    states = jax.random.randint(key, (32, 5), 0, 3)
    z, _ = model.embed(states.astype(jnp.float32))
    x = states[0]
    logits = model.generator.net(x, z)

    for j in range(5):
        row = slice(8 + 5 * j, 8 + 5 * (j + 1))  # after the 8 moment features
        changed = model.generator.net(x, z.at[row].add(1.0))
        np.testing.assert_allclose(changed[:j], logits[:j], atol=1e-6)
        assert not np.allclose(changed[j], logits[j])


def test_samples_follow_the_model_probabilities(make_ar, key):
    model, z = make_ar(3, 2)
    states = all_states(3, 2)
    log_probs, _, _ = jax.vmap(lambda x: model.log_prob(x, z))(states)

    samples = jax.vmap(lambda k: model.sample(z, None, k))(jax.random.split(key, 20000))

    indices = np.asarray(samples @ jnp.array([4.0, 2.0, 1.0])).astype(int)
    frequencies = np.bincount(indices, minlength=8) / len(indices)
    np.testing.assert_allclose(frequencies, np.exp(log_probs), atol=0.015)
