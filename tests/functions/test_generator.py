"""The `Generator` contract: what every function family must produce for training and evaluation."""

import jax
import numpy as np

from embedex.functions.builder import create_embedding_network, create_operator_network

BATCH, SAMPLES = 8, 32


def test_samples_have_the_documented_shapes(family, make_config, key):
    generator = family.generator()
    xs, ys, extras = generator.sample(BATCH, SAMPLES, key)

    assert xs.shape == (BATCH, SAMPLES, 1)
    assert ys.shape == (BATCH, SAMPLES, family.output_size)
    if family.ode:  # the parameters that determine each function
        assert extras.shape == (BATCH, generator.operator_input_size(make_config(family)))
    else:  # the derivative at every sample
        assert extras.shape == (BATCH, SAMPLES, 1)


def test_samples_are_finite_and_inputs_lie_in_the_domain(family, key):
    xs, ys, extras = family.generator().sample(BATCH, SAMPLES, key)
    low, high = family.domain()

    assert np.all((np.asarray(xs) >= low) & (np.asarray(xs) <= high))
    assert all(np.all(np.isfinite(a)) for a in (xs, ys, extras))


def test_sampling_is_reproducible_and_key_dependent(family, key):
    generator = family.generator()
    samples = generator.sample(BATCH, SAMPLES, key)
    samples_again = generator.sample(BATCH, SAMPLES, key)
    other_samples = generator.sample(BATCH, SAMPLES, jax.random.key(1))

    for a, b, c in zip(samples, samples_again, other_samples, strict=True):
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)


def test_a_batch_holds_different_functions(family, key):
    _, ys, _ = family.generator().sample(BATCH, SAMPLES, key)

    assert len({np.asarray(y).tobytes() for y in ys}) == BATCH


def test_ode_samples_lie_on_the_solution_for_their_parameters(ode_family, key):
    # The operator maps these parameters to the function, so they must be the ones that
    # generated it, in the order the family's solution reads them.
    generator = ode_family.generator()
    ts, ys, params = generator.sample(BATCH, SAMPLES, key)

    solutions = jax.vmap(lambda t, p: ode_family.solution(generator, t, p))(ts[..., 0], params)

    np.testing.assert_allclose(solutions, ys, rtol=1e-4, atol=1e-5)


def test_operator_maps_its_input_to_an_embedding_of_the_target(family, make_config, key):
    cfg = make_config(family)
    generator = family.generator()
    xs, ys, extras = generator.sample(BATCH, SAMPLES, key)
    embedding = create_embedding_network(cfg, key)
    operator = create_operator_network(cfg, generator, key)

    source = generator.operator_input(embedding, xs[0], ys[0], extras[0])
    target = generator.operator_ground_truth(xs[0], ys[0], extras[0])
    prediction = embedding.eval(xs[0], operator(source))

    assert source.shape == (generator.operator_input_size(cfg),)
    assert target.shape == prediction.shape == (SAMPLES, family.output_size)
