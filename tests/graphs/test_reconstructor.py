"""Reconstruction heads, their losses, and the training-free reconstruction rules."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.graphs.builder import (
    create_embedding_network,
    create_medians_network,
    create_transition_network,
)
from embedex.graphs.families.gridworld import GridWorldGenerator
from embedex.graphs.families.random_trees import RandomTreesGenerator
from embedex.graphs.reconstructor import (
    medians_loss,
    recon_average_degree,
    recon_medians,
    recon_transitions,
    recon_vertex_degrees,
    transition_distances,
    transition_loss,
)

N = 8  # vertices per graph; matches the `embedding_network` fixture

# A small symmetric distance matrix whose nearest-neighbour structure is easy to reason about:
# vertex 0 is closest to 1 then 2, and 3 is the odd one out.
DIST = jnp.array(
    [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 4.0, 5.0],
        [2.0, 4.0, 0.0, 6.0],
        [3.0, 5.0, 6.0, 0.0],
    ]
)
# Edges 0-1 and 0-2: what each rule below recovers from DIST.
NEAREST = np.zeros((4, 4), bool)
NEAREST[[0, 1, 0, 2], [1, 0, 2, 0]] = True


def assert_adjacency(adj: jax.Array) -> np.ndarray:
    """Checks the output is a symmetric boolean adjacency without self-loops, and returns it."""

    adj = np.asarray(adj)
    assert adj.dtype == bool
    np.testing.assert_array_equal(np.diag(adj), False)
    np.testing.assert_array_equal(adj, adj.T)

    return adj


def constant_scores(network, value: float = 0.0):
    """A transition network that gives every pair the same score."""

    last = network.net.layers[-1]
    network = eqx.tree_at(lambda m: m.net.layers[-1].weight, network, jnp.zeros_like(last.weight))
    return eqx.tree_at(lambda m: m.net.layers[-1].bias, network, jnp.full_like(last.bias, value))


def symmetric_distances(n: int, key: jax.Array) -> jax.Array:
    d = jnp.abs(jax.random.normal(key, (n, n)))
    return (d + d.T) * (1.0 - jnp.eye(n))


@pytest.fixture
def tree_edges(key) -> jax.Array:
    edges, _ = RandomTreesGenerator(N).sample(4, 64, key)
    return edges


class TestTrainingFreeRules:
    def test_average_degree_keeps_the_target_number_of_edges(self):
        # Average degree 1 over 4 vertices -> the 2 closest pairs.
        adj = assert_adjacency(recon_average_degree(DIST, jnp.asarray(1.0)))

        np.testing.assert_array_equal(adj, NEAREST)

    def test_average_degree_keeps_at_least_one_edge(self):
        adj = assert_adjacency(recon_average_degree(DIST, jnp.asarray(0.0)))

        assert adj.sum() >= 2  # one undirected edge

    def test_vertex_degrees_connect_each_vertex_to_its_nearest_neighbours(self):
        adj = assert_adjacency(recon_vertex_degrees(DIST, jnp.array([1, 1, 1, 1])))

        np.testing.assert_array_equal(adj, NEAREST)

    def test_vertex_with_degree_zero_stays_isolated(self):
        adj = assert_adjacency(recon_vertex_degrees(DIST, jnp.array([0, 1, 1, 1])))

        assert not adj[0].any()

    def test_medians_connect_pairs_within_their_summed_radii(self):
        # Edge iff dist(u, v) <= med(u) + med(v): radii 1 give a cutoff of 2.
        adj = assert_adjacency(recon_medians(DIST, jnp.ones(4)))

        np.testing.assert_array_equal(adj, NEAREST)

    def test_larger_radii_only_add_edges(self):
        small = assert_adjacency(recon_medians(DIST, jnp.full(4, 1.0)))
        large = assert_adjacency(recon_medians(DIST, jnp.full(4, 10.0)))

        assert np.all(large | ~small)
        assert large.sum() > small.sum()

    def test_transitions_are_a_distribution_over_pairs_without_self_loops(self, key):
        weights = recon_transitions(symmetric_distances(N, key))

        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
        np.testing.assert_array_equal(jnp.diag(weights), 0.0)

    def test_closer_pairs_get_more_transition_weight(self):
        weights = recon_transitions(DIST)

        assert weights[0, 1] > weights[0, 2] > weights[0, 3]
        np.testing.assert_allclose(weights, weights.T)


class TestTransitionNetwork:
    def test_predicts_a_distribution_over_pairs_without_self_loops(self, make_config, key):
        network = create_transition_network(make_config(), key)
        z = jax.random.normal(key, (8,))

        weights = network(symmetric_distances(N, key), z)

        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)
        np.testing.assert_array_equal(jnp.diag(weights), 0.0)
        assert jnp.all(weights >= 0)

    def test_self_loops_get_minus_infinite_logits(self, make_config, key):
        network = create_transition_network(make_config(), key)
        logits = network.logits(symmetric_distances(N, key), jnp.ones(8))

        assert jnp.all(jnp.isneginf(jnp.diag(logits)))
        assert jnp.all(jnp.isfinite(logits[~jnp.eye(N, dtype=bool)]))

    def test_equal_scores_give_uniform_weights(self, make_config, key):
        network = constant_scores(create_transition_network(make_config(), key), 3.0)
        weights = network(symmetric_distances(N, key), jnp.ones(8))

        np.testing.assert_allclose(weights[~jnp.eye(N, dtype=bool)], 1.0 / (N * (N - 1)))


class TestLosses:
    def test_transition_distances_match_the_embedding_distances(
        self, embedding_network, tree_edges
    ):
        z = embedding_network.embed(tree_edges[0])

        np.testing.assert_allclose(
            transition_distances(embedding_network, z, N),
            embedding_network.distance_matrix(jnp.eye(N), z),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_transition_loss_of_uniform_weights_is_log_of_the_pair_count(
        self, make_config, embedding_network, tree_edges, key
    ):
        # Uniform weights put probability 1 / (N (N - 1)) on every held-out edge.
        network = constant_scores(create_transition_network(make_config(), key))

        loss = transition_loss(network, embedding_network, tree_edges, key)

        np.testing.assert_allclose(loss, np.log(N * (N - 1)), rtol=1e-5)

    @pytest.mark.parametrize(
        'loss_fn, create_fn',
        [
            (medians_loss, create_medians_network),
            (transition_loss, create_transition_network),
        ],
        ids=['medians', 'transition'],
    )
    def test_losses_and_gradients_are_finite(
        self, loss_fn, create_fn, make_config, embedding_network, tree_edges, key
    ):
        network = create_fn(make_config(), key)

        loss, grads = eqx.filter_value_and_grad(
            lambda m: loss_fn(m, embedding_network, tree_edges, key)
        )(network)

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads))

    @pytest.mark.parametrize(
        'loss_fn, create_fn',
        [
            (medians_loss, create_medians_network),
            (transition_loss, create_transition_network),
        ],
        ids=['medians', 'transition'],
    )
    def test_losses_stay_finite_when_eigenvalues_are_not_positive(
        self, loss_fn, create_fn, make_config, embedding_network, tree_edges, key
    ):
        # Degenerate sub-embeddings can push every ALLO eigenvalue to zero or below.
        embedding_network = eqx.tree_at(
            lambda m: m.betas.net.layers[-1].bias, embedding_network, jnp.full((1,), 10.0)
        )
        network = create_fn(make_config(), key)

        assert jnp.isfinite(loss_fn(network, embedding_network, tree_edges, key))

    def test_transition_loss_runs_on_weighted_graphs(self, make_config, key):
        cfg = make_config()
        cfg.model.input_size = 9
        edges, _ = GridWorldGenerator(3, num_walls=2).sample(4, 64, key)

        embedding_network = create_embedding_network(cfg, key)
        network = create_transition_network(cfg, key)

        assert jnp.isfinite(transition_loss(network, embedding_network, edges, key))
