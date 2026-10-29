"""The `Generator` contract: what every graph family must produce, and how edges are sampled."""

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from embedex.graphs.generator import Generator

BATCH, SAMPLES = 32, 64


class FixedGraph(Generator):
    """Always returns the same adjacency, so edge sampling can be checked in isolation."""

    def __init__(self, adj: jax.Array) -> None:
        self.adj = adj

    def _sample_one(self, num_samples: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self._sample_edges_from_adj(self.adj, num_samples, key), self.adj


def edge_indices(edges: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    """(sources, targets) of one-hot encoded edges with shape (..., 2, n)."""

    edges = np.asarray(edges)
    return edges[..., 0, :].argmax(-1), edges[..., 1, :].argmax(-1)


def test_edges_are_drawn_in_proportion_to_their_weight(key):
    # A source is drawn by weighted degree and a neighbour by edge weight, so an ordered pair
    # (u, v) appears with probability w(u, v) / sum(w).
    adj = jnp.array(
        [
            [0.0, 1.0, 3.0, 0.0],
            [1.0, 0.0, 0.0, 2.0],
            [3.0, 0.0, 0.0, 0.5],
            [0.0, 2.0, 0.5, 0.0],
        ]
    )
    edges, _ = FixedGraph(adj).sample(1, 40_000, key)
    sources, targets = edge_indices(edges[0])

    frequencies = np.zeros((4, 4))
    np.add.at(frequencies, (sources, targets), 1.0)
    frequencies /= frequencies.sum()

    np.testing.assert_allclose(frequencies, np.asarray(adj / adj.sum()), atol=0.01)


def test_samples_have_the_documented_shapes(family, key):
    edges, adjs = family.generator().sample(BATCH, SAMPLES, key)
    n = family.num_vertices

    assert edges.shape == (BATCH, SAMPLES, 2, n)
    assert adjs.shape == (BATCH, n, n)


def test_edges_are_pairs_of_one_hot_vertices(family, key):
    edges, _ = family.generator().sample(BATCH, SAMPLES, key)
    edges = np.asarray(edges)

    assert set(np.unique(edges)) <= {0.0, 1.0}
    np.testing.assert_array_equal(edges.sum(-1), 1.0)


def test_every_sampled_edge_belongs_to_its_graph(family, key):
    edges, adjs = family.generator().sample(BATCH, SAMPLES, key)
    sources, targets = edge_indices(edges)
    graphs = np.arange(BATCH)[:, np.newaxis]

    assert np.all(np.asarray(adjs)[graphs, sources, targets] > 0)


def test_graphs_are_undirected_without_self_loops(family, key):
    _, adjs = family.generator().sample(BATCH, SAMPLES, key)
    adjs = np.asarray(adjs)

    np.testing.assert_array_equal(adjs, np.transpose(adjs, (0, 2, 1)))
    np.testing.assert_array_equal(np.diagonal(adjs, axis1=1, axis2=2), 0.0)
    assert np.all(adjs >= 0.0)


def test_edge_weights_match_the_family_kind(family, key):
    _, adjs = family.generator().sample(BATCH, SAMPLES, key)
    weights = np.asarray(adjs)[np.asarray(adjs) > 0]

    if family.weighted:
        assert np.all(weights <= 1.0)
        assert len(np.unique(weights)) > 1
    else:
        np.testing.assert_array_equal(weights, 1.0)


def test_graphs_are_connected(family, key):
    _, adjs = family.generator().sample(BATCH, SAMPLES, key)

    for adj in np.asarray(adjs):
        assert nx.is_connected(nx.from_numpy_array(adj > 0))


def test_sampling_is_reproducible_and_key_dependent(family, key):
    generator = family.generator()
    edges, adjs = generator.sample(BATCH, SAMPLES, key)
    edges_again, adjs_again = generator.sample(BATCH, SAMPLES, key)
    _, other_adjs = generator.sample(BATCH, SAMPLES, jax.random.key(1))

    np.testing.assert_array_equal(edges, edges_again)
    np.testing.assert_array_equal(adjs, adjs_again)
    assert not np.array_equal(adjs, other_adjs)


def test_a_batch_holds_different_graphs(family, key):
    _, adjs = family.generator().sample(BATCH, SAMPLES, key)

    assert len({np.asarray(adj).tobytes() for adj in adjs}) > 1
