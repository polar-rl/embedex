"""Barabási-Albert scale-free graphs, grown by preferential attachment."""

import networkx as nx
import numpy as np
import pytest

from embedex.graphs.families.random_trees import RandomTreesGenerator
from embedex.graphs.families.scale_free import ScaleFreeGenerator


def max_degrees(adjs) -> np.ndarray:
    return np.asarray(adjs).sum(-1).max(-1)


def test_each_new_vertex_adds_between_one_and_m_edges(key):
    n, m = 12, 3
    _, adjs = ScaleFreeGenerator(n, m).sample(64, 4, key)

    num_edges = np.asarray(adjs).sum((1, 2)) / 2
    assert np.all((num_edges >= n - 1) & (num_edges <= m * (n - 1)))


def test_one_edge_per_new_vertex_grows_a_tree(key):
    _, adjs = ScaleFreeGenerator(10, 1).sample(64, 4, key)

    assert all(nx.is_tree(nx.from_numpy_array(adj)) for adj in np.asarray(adjs))


def test_preferential_attachment_grows_hubs(key):
    # Trees of the same size: preferential attachment gives far larger hubs than uniform ones.
    _, scale_free = ScaleFreeGenerator(40, 1).sample(256, 4, key)
    _, uniform = RandomTreesGenerator(40).sample(256, 4, key)

    assert max_degrees(scale_free).mean() > 1.5 * max_degrees(uniform).mean()


def test_vertex_labels_are_shuffled(key):
    # Without the shuffle, the oldest vertex would always be the biggest hub.
    _, adjs = ScaleFreeGenerator(10, 2).sample(64, 4, key)

    hubs = np.asarray(adjs).sum(-1).argmax(-1)
    assert len(np.unique(hubs)) > 1


def test_m_must_be_smaller_than_the_number_of_vertices():
    with pytest.raises(AssertionError):
        ScaleFreeGenerator(5, 5)
