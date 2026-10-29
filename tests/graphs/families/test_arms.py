"""'Arms': a hub with equally long paths ('arms') hanging off it."""

import networkx as nx
import numpy as np

from embedex.graphs.families.arms import ArmsGenerator

N = 7  # arm lengths must divide N - 1 = 6, so a graph has 6, 3 or 2 arms (of length 1, 2 or 3)


def num_arms(adj: np.ndarray) -> int:
    """Checks that the graph is a hub with equally long arms, and returns how many arms it has."""

    graph = nx.from_numpy_array(adj)
    assert nx.is_tree(graph)

    leaves = [v for v, degree in graph.degree if degree == 1]
    arms = len(leaves)
    assert (N - 1) % arms == 0
    length = (N - 1) // arms

    hubs = [
        v
        for v in graph
        if graph.degree[v] == arms
        and all(nx.shortest_path_length(graph, v, leaf) == length for leaf in leaves)
    ]
    assert hubs, 'no vertex has every arm ending at the same distance'
    assert all(graph.degree[v] <= 2 for v in graph if v not in hubs)

    return arms


def test_every_graph_is_a_hub_with_equally_long_arms(key):
    _, adjs = ArmsGenerator(N).sample(64, 4, key)

    for adj in np.asarray(adjs):
        num_arms(adj)


def test_every_arm_length_occurs(key):
    _, adjs = ArmsGenerator(N).sample(64, 4, key)

    assert {num_arms(adj) for adj in np.asarray(adjs)} == {6, 3, 2}


def test_hub_is_not_always_the_same_vertex(key):
    _, adjs = ArmsGenerator(N).sample(64, 4, key)

    hubs = np.asarray(adjs).sum(-1).argmax(-1)
    assert len(np.unique(hubs)) > 1
