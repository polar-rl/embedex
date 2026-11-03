"""The relational stream, which embeds the pairwise statistics of categorical samples, its two
ablation arms, and the likelihood gate that weighs the samples it reads."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.distributions.config import EmbeddingConfig
from embedex.distributions.relational import (
    LikelihoodGate,
    RawAgreeStream,
    RawEdgeStream,
    RelationalStream,
    pair_weights,
)

NUM_VARIABLES, NUM_CLASSES = 6, 3
MODES = ('learned', 'raw_agree', 'raw_edge')
STREAM_TYPES = {'learned': RelationalStream, 'raw_agree': RawAgreeStream, 'raw_edge': RawEdgeStream}


@pytest.fixture(params=MODES)
def mode(request) -> str:
    return request.param


@pytest.fixture
def make_stream(make_model):
    """The relational stream of a model built for `mode`. Its class embeddings are scaled up from
    their small initial values, so the pairwise statistics move the output measurably."""

    def _make(mode: str, num_variables: int = NUM_VARIABLES, num_classes: int = NUM_CLASSES):
        model = make_model(
            flow_type='ar',
            input_dim=num_variables,
            num_classes=num_classes,
            relational_mode=mode,
        )
        stream = model.relational
        if hasattr(stream, 'cell_embed'):
            stream = eqx.tree_at(lambda s: s.cell_embed, stream, 10.0 * stream.cell_embed)
        return stream

    return _make


@pytest.fixture
def states(key) -> jax.Array:
    return jax.random.randint(key, (40, NUM_VARIABLES), 0, NUM_CLASSES).astype(jnp.float32)


class TestStreams:
    def test_each_mode_builds_its_own_stream(self, make_stream, mode):
        assert type(make_stream(mode)) is STREAM_TYPES[mode]

    @pytest.mark.parametrize('num_variables, num_classes', [(18, 2), (9, 4)])
    def test_every_arm_emits_one_value_per_pair_of_variables(
        self, make_stream, mode, num_variables, num_classes, key
    ):
        # The ablation compares the arms at the same conditioning width, so a difference between
        # them cannot be a width effect.
        stream = make_stream(mode, num_variables, num_classes)
        xs = jax.random.randint(key, (16, num_variables), 0, num_classes).astype(jnp.float32)

        assert stream.out_dim == num_variables**2
        assert stream(xs).shape == (num_variables**2,)

    def test_arm_parameter_counts_order_as_designed(self, make_stream):
        # rawA is the statistic itself; rawE keeps the cell embedding and the projections but not
        # the attention, so it sits between rawA and the learned stream.
        def count(mode):
            params = eqx.filter(make_stream(mode), eqx.is_inexact_array)
            return sum(x.size for x in jax.tree.leaves(params))

        assert count('raw_agree') == 0
        assert 0 < count('raw_edge') < count('learned')

    def test_streams_read_the_samples_as_a_set(self, make_stream, mode, states, key):
        stream = make_stream(mode)
        shuffled = states[jax.random.permutation(key, len(states))]

        np.testing.assert_allclose(stream(shuffled), stream(states), atol=1e-5)
        np.testing.assert_allclose(stream(jnp.tile(states, (2, 1))), stream(states), atol=1e-5)

    def test_relabeling_the_variables_relabels_the_output(self, make_stream, mode, states):
        # Nothing in a stream tells the variables apart, so permuting them permutes the rows of
        # the output (and, for the pairwise arms, its columns too).
        stream = make_stream(mode)
        perm = jnp.array([4, 2, 0, 5, 1, 3])
        out = stream(states).reshape(NUM_VARIABLES, -1)

        permuted = stream(states[:, perm]).reshape(NUM_VARIABLES, -1)

        expected = out[perm] if mode == 'learned' else out[perm][:, perm]
        np.testing.assert_allclose(permuted, expected, atol=1e-5)

    def test_raw_agree_is_the_agreement_matrix(self, make_stream, states):
        xs = np.asarray(states)

        agreement = make_stream('raw_agree')(states).reshape(NUM_VARIABLES, NUM_VARIABLES)

        expected = np.mean(xs[:, :, None] == xs[:, None, :], axis=0)
        np.testing.assert_allclose(agreement, expected, atol=1e-6)

    def test_out_of_range_states_are_clipped_to_valid_classes(self, make_stream, mode, states):
        stream = make_stream(mode)
        out_of_range = jnp.where(states == NUM_CLASSES - 1, NUM_CLASSES + 3.0, states)

        np.testing.assert_allclose(stream(out_of_range), stream(states), atol=1e-6)


class TestLikelihoodGate:
    def test_starts_by_weighing_every_sample_the_same(self):
        weights = LikelihoodGate(NUM_VARIABLES)(16, jnp.linspace(-30.0, 0.0, 16)[:, None])

        np.testing.assert_allclose(weights, 1 / 16, atol=1e-6)

    def test_without_likelihood_every_sample_weighs_the_same(self):
        np.testing.assert_allclose(LikelihoodGate(NUM_VARIABLES)(8, None), 1 / 8)

    def test_a_closed_gate_drops_unlikely_samples(self):
        gate = closed(LikelihoodGate(NUM_VARIABLES))
        likelihood = jnp.array([0.0, 0.0, -30.0, 0.0])[:, None]

        weights = gate(4, likelihood)

        np.testing.assert_allclose(weights[:, 0], [1 / 3, 1 / 3, 0.0, 1 / 3], atol=1e-6)

    def test_an_index_routes_each_variable_to_its_factor(self):
        gate = closed(LikelihoodGate(NUM_VARIABLES, (0, 0, 1, 1, 2, 2)))
        likelihood = jnp.zeros((8, 3)).at[:4, 1].set(-30.0)  # factor 1 rules out samples 0-3

        weights = np.asarray(gate(8, likelihood))

        np.testing.assert_allclose(weights[:, [0, 1, 4, 5]], 1 / 8, atol=1e-6)
        np.testing.assert_allclose(weights[:4, 2:4], 0.0, atol=1e-6)
        np.testing.assert_allclose(weights[4:, 2:4], 1 / 4, atol=1e-6)

    @pytest.mark.parametrize('width', [1, NUM_VARIABLES])
    def test_one_feature_is_shared_and_one_per_variable_is_aligned(self, width, key):
        gate = closed(LikelihoodGate(NUM_VARIABLES))
        likelihood = jnp.where(jax.random.bernoulli(key, 0.5, (8, width)), 0.0, -30.0)

        weights = gate(8, likelihood)

        kept = jnp.broadcast_to(likelihood == 0.0, (8, NUM_VARIABLES))
        np.testing.assert_allclose(weights, kept / kept.sum(0), atol=1e-6)

    def test_rejects_a_feature_width_it_cannot_map(self):
        with pytest.raises(ValueError, match='likelihood_index'):
            LikelihoodGate(NUM_VARIABLES)(4, jnp.zeros((4, 3)))

    def test_rejects_an_index_of_the_wrong_length(self):
        with pytest.raises(ValueError, match='likelihood_index'):
            LikelihoodGate(NUM_VARIABLES, (0, 1))

    def test_pair_weights_reduce_to_sample_weights(self):
        sample_weights = jax.nn.softmax(jnp.linspace(-3.0, 0.0, 16))

        pairs = pair_weights(jnp.broadcast_to(sample_weights[:, None], (16, NUM_VARIABLES)))

        expected = jnp.broadcast_to(sample_weights[:, None, None], pairs.shape)
        np.testing.assert_allclose(pairs, expected, rtol=1e-6)

    def test_closed_gate_gives_the_statistics_of_the_kept_samples(self, make_stream, states, key):
        stream = make_stream('learned')
        stream = eqx.tree_at(lambda s: s.gate, stream, closed(stream.gate))
        likelihood = jnp.where(jax.random.bernoulli(key, 0.5, (len(states), 1)), 0.0, -30.0)
        kept = np.asarray(likelihood[:, 0] == 0.0)

        gated = stream(states, likelihood)

        expected = stream(states[kept], jnp.zeros((int(kept.sum()), 1)))
        np.testing.assert_allclose(gated, expected, atol=1e-4)


def closed(gate: LikelihoodGate) -> LikelihoodGate:
    """`gate` with sigmoid(l + 15): open (1) for log-likelihood 0, shut for -30."""

    n = gate.num_variables
    return eqx.tree_at(lambda g: (g.a, g.b), gate, (jnp.ones(n), jnp.full(n, 15.0)))


class TestModeAlias:
    """`relational: bool` predates `relational_mode` and every older checkpoint still sets it."""

    @pytest.mark.parametrize(
        'fields, mode',
        [
            ({}, 'none'),
            ({'relational': True}, 'learned'),
            ({'relational': False}, 'none'),
            ({'relational_mode': 'raw_edge'}, 'raw_edge'),
            ({'relational': True, 'relational_mode': 'learned'}, 'learned'),
        ],
    )
    def test_resolves_to_a_mode(self, fields, mode):
        assert EmbeddingConfig(**fields).rel_mode == mode

    def test_contradicting_the_alias_raises(self):
        # Otherwise a config setting both fields would build the wrong model in silence.
        with pytest.raises(ValueError, match='contradicts'):
            _ = EmbeddingConfig(relational=True, relational_mode='raw_agree').rel_mode
