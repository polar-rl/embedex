"""Random labelled trees, decoded from uniformly random Prüfer sequences."""

from collections import Counter

import networkx as nx
import numpy as np

from embedex.graphs.families.random_trees import RandomTreesGenerator


def test_every_graph_is_a_spanning_tree(key):
    _, adjs = RandomTreesGenerator(10).sample(64, 4, key)

    for adj in np.asarray(adjs):
        graph = nx.from_numpy_array(adj)
        assert graph.number_of_nodes() == 10
        assert nx.is_tree(graph)


def test_every_labelled_tree_is_equally_likely(key):
    # Prüfer decoding is a bijection, so all 4^2 = 16 labelled trees on 4 vertices (Cayley's
    # formula) appear with probability 1/16 each.
    num_graphs = 8000
    _, adjs = RandomTreesGenerator(4).sample(num_graphs, 2, key)

    counts = Counter(adj.tobytes() for adj in np.asarray(adjs))

    assert len(counts) == 16
    expected = num_graphs / 16
    assert all(abs(count - expected) < 0.2 * expected for count in counts.values())
