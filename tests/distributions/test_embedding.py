"""Embedding sample sets, and the losses that train an embedding through its generator."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.distributions.embedding import AttentionPool

NUM_SAMPLES = 16


@pytest.fixture
def samples(key) -> jax.Array:
    return jax.random.normal(key, (NUM_SAMPLES, 2)) * jnp.array([1.0, 3.0]) + 1.0


def embed(network, samples: jax.Array) -> np.ndarray:
    return np.asarray(network.embed(samples)[0])


class TestEmbed:
    def test_embedding_is_a_finite_vector_returned_with_its_norm(self, embedding_network, samples):
        z, norm = embedding_network.embed(samples)

        assert z.shape == (8,)
        assert jnp.all(jnp.isfinite(z))
        assert float(norm) == pytest.approx(float(jnp.linalg.norm(z)))

    def test_embedding_ignores_the_order_of_the_samples(self, embedding_network, samples, key):
        shuffled = samples[jax.random.permutation(key, NUM_SAMPLES)]

        np.testing.assert_allclose(
            embed(embedding_network, shuffled), embed(embedding_network, samples), atol=1e-5
        )

    def test_embedding_depends_only_on_the_empirical_distribution(self, embedding_network, samples):
        # Repeating every sample leaves the empirical distribution, and so the embedding, as it is.
        repeated = jnp.tile(samples, (3, 1))

        np.testing.assert_allclose(
            embed(embedding_network, repeated), embed(embedding_network, samples), atol=1e-5
        )

    def test_different_sample_sets_get_different_embeddings(self, embedding_network, samples):
        assert not np.allclose(
            embed(embedding_network, samples), embed(embedding_network, 2.0 * samples), atol=1e-3
        )

    def test_every_round_contributes_to_the_embedding(self, embedding_network, samples, key):
        # Only the last round is read out, so the first can only matter through the summary it
        # hands to the next round.
        other_first = eqx.tree_at(
            lambda m: m.embedding_blocks[0],
            embedding_network,
            jax.tree.map(
                lambda x: x + jax.random.normal(key, x.shape) if eqx.is_inexact_array(x) else x,
                embedding_network.embedding_blocks[0],
            ),
        )

        assert not np.allclose(
            embed(other_first, samples), embed(embedding_network, samples), atol=1e-4
        )


class TestAttentionPool:
    @pytest.fixture
    def pool(self, key) -> AttentionPool:
        return AttentionPool(dim=8, num_heads=2, num_queries=3, key=key)

    def test_reads_out_one_vector_per_query(self, pool, key):
        out = pool(jax.random.normal(key, (10, 8)), jnp.ones(10))

        assert out.shape == (3 * 8,)

    def test_ignores_samples_without_weight(self, pool, key):
        zs = jax.random.normal(key, (10, 8))
        weights = jnp.array([1.0] * 6 + [0.0] * 4)

        np.testing.assert_allclose(pool(zs, weights), pool(zs[:6], jnp.ones(6)), atol=1e-5)

    def test_only_relative_weights_matter(self, pool, key):
        zs = jax.random.normal(key, (10, 8))
        weights = jnp.linspace(0.5, 2.0, 10)

        np.testing.assert_allclose(pool(zs, weights), pool(zs, 7.0 * weights), atol=1e-5)

    def test_weighting_a_sample_counts_it_several_times(self, pool, key):
        zs = jax.random.normal(key, (4, 8))
        repeated = jnp.concatenate([zs, zs[:1], zs[:1]])

        np.testing.assert_allclose(
            pool(zs, jnp.array([3.0, 1.0, 1.0, 1.0])), pool(repeated, jnp.ones(6)), atol=1e-5
        )


class TestLosses:
    def test_loss_scores_one_half_of_the_set_under_the_embedding_of_the_other(
        self, embedding_network, samples, key
    ):
        loss, _, _, norm = embedding_network.loss(samples, shuffle=False, key=key)

        z, z_norm = embedding_network.embed(samples[: NUM_SAMPLES // 2])
        log_probs, _, _ = embedding_network.log_prob(samples[NUM_SAMPLES // 2 :], z)
        assert float(loss) == pytest.approx(-float(jnp.mean(log_probs)), rel=1e-5)
        assert float(norm) == pytest.approx(float(z_norm))

    def test_shuffling_splits_the_set_at_random(self, embedding_network, samples, key):
        losses = {
            float(embedding_network.loss(samples, key=k)[0]) for k in jax.random.split(key, 8)
        }

        assert len(losses) > 1

    def test_gold_loss_scores_the_gold_sample_under_the_embedding_of_the_whole_set(
        self, embedding_network, samples, key
    ):
        gold = jnp.array([0.5, -1.0])

        loss, _, _, _ = embedding_network.gold_loss(samples, None, gold, key=key)

        z, _ = embedding_network.embed(samples)
        log_prob, _, _ = embedding_network.log_prob(gold[jnp.newaxis], z)
        assert float(loss) == pytest.approx(-float(log_prob[0]), rel=1e-5)

    def test_loss_and_its_gradients_are_finite(self, embedding_network, samples, key):
        loss, grads = eqx.filter_value_and_grad(lambda m: m.loss(samples, key=key)[0])(
            embedding_network
        )

        assert jnp.isfinite(loss)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads))


class TestGenerativeModel:
    def test_generated_samples_are_independent_draws(self, embedding_network, samples, key):
        z, _ = embedding_network.embed(samples)

        generated = np.asarray(embedding_network.generate(z, 32, key=key))

        assert generated.shape == (32, 2)
        assert len(np.unique(generated, axis=0)) == 32

    def test_repeated_log_prob_passes_are_averaged_in_probability_space(self, make_model, key):
        # Dequantization noise makes each pass of a discrete normalizing flow different.
        model = make_model(flow_type='nf', dequantize=True, nf_dequant_type='uniform')
        xs = jax.random.randint(key, (8, 2), 0, 2).astype(jnp.float32)
        z, _ = model.embed(xs)

        passes = jnp.stack([model.log_prob(xs, z, key=k)[0] for k in jax.random.split(key, 4)])
        averaged, _, _ = model.log_prob(xs, z, reps=4, key=key)

        assert not np.allclose(passes[0], passes[1])
        np.testing.assert_allclose(
            averaged, jax.nn.logsumexp(passes, axis=0) - jnp.log(4.0), rtol=1e-5
        )


class TestLikelihoodFeatures:
    """Samples can carry per-sample likelihood features, which `embed` hands to the moment
    stream as extra columns and to the relational stream as a gate."""

    NUM_VARIABLES = 9

    @pytest.fixture
    def model(self, make_model):
        return make_model(
            flow_type='ar',
            input_dim=self.NUM_VARIABLES,
            likelihood_dim=1,
            relational_mode='learned',
        )

    @pytest.fixture
    def states(self, key) -> np.ndarray:
        return np.asarray(jax.random.bernoulli(key, 0.5, (32, self.NUM_VARIABLES)))

    @pytest.fixture
    def likelihood(self, key) -> jax.Array:
        return jnp.where(jax.random.bernoulli(key, 0.5, (32, 1)), 0.0, -2.5)

    def test_likelihood_features_change_the_embedding(self, model, states, likelihood):
        xs = jnp.asarray(states, dtype=jnp.float32)

        assert not np.allclose(
            model.embed(xs, likelihood)[0], model.embed(xs, jnp.zeros_like(likelihood))[0]
        )

    def test_boolean_states_do_not_truncate_the_likelihood(self, model, states, likelihood):
        np.testing.assert_allclose(
            model.embed(jnp.asarray(states), likelihood)[0],
            model.embed(jnp.asarray(states, dtype=jnp.float32), likelihood)[0],
            atol=1e-6,
        )

    @pytest.mark.parametrize('shape', [None, (32, 2), (31, 1)])
    def test_embed_requires_the_declared_likelihood_shape(self, model, states, shape):
        likelihood = None if shape is None else jnp.zeros(shape)

        with pytest.raises(ValueError, match='likelihood'):
            model.embed(jnp.asarray(states, dtype=jnp.float32), likelihood)
