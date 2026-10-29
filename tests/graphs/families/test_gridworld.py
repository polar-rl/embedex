"""Weighted gridworlds: a square grid with random edge weights and some edges removed ('walls')."""

import networkx as nx
import numpy as np

from embedex.graphs.families.gridworld import GridWorldGenerator

GRID_EDGES = 24  # undirected edges of a full 4x4 grid


def num_edges(adjs) -> np.ndarray:
    return (np.asarray(adjs) > 0).sum((1, 2)) / 2


def test_without_walls_every_graph_is_the_full_grid(key):
    generator = GridWorldGenerator(4, num_walls=0)
    _, adjs = generator.sample(64, 16, key)

    support = np.asarray(adjs) > 0
    np.testing.assert_array_equal(support, np.broadcast_to(generator.grid_mask > 0, support.shape))


def test_edges_only_join_grid_neighbours(key):
    generator = GridWorldGenerator(4, num_walls=3)
    _, adjs = generator.sample(64, 16, key)

    assert np.all(np.asarray(adjs)[:, np.asarray(generator.grid_mask) == 0] == 0.0)


def test_walls_remove_at_most_num_walls_edges(key):
    _, adjs = GridWorldGenerator(4, num_walls=2).sample(256, 16, key)

    assert np.all((num_edges(adjs) >= GRID_EDGES - 2) & (num_edges(adjs) <= GRID_EDGES))


def test_the_number_of_walls_varies_up_to_num_walls(key):
    # Each graph draws its number of walls uniformly from {0, ..., num_walls}.
    _, adjs = GridWorldGenerator(4, num_walls=2).sample(256, 16, key)

    assert set(GRID_EDGES - num_edges(adjs)) == {0, 1, 2}


def test_walls_never_disconnect_the_grid(key):
    # With up to 4 of a 3x3 grid's 12 edges removed, unchecked walls would often cut off a
    # vertex: a corner has only two edges.
    _, adjs = GridWorldGenerator(3, num_walls=4).sample(256, 16, key)

    for adj in np.asarray(adjs):
        assert nx.is_connected(nx.from_numpy_array(adj > 0))


def test_walls_vary_across_graphs(key):
    _, adjs = GridWorldGenerator(4, num_walls=2).sample(64, 16, key)

    assert len({(adj > 0).tobytes() for adj in np.asarray(adjs)}) > 3


def test_edge_weights_are_raw_not_normalised(key):
    # Evaluation normalises the target to S / sum(S) itself.
    _, adjs = GridWorldGenerator(4, num_walls=0).sample(64, 16, key)

    assert np.all(np.asarray(adjs).sum((1, 2)) > 1.0)


def test_connectivity_check_detects_an_isolated_vertex():
    generator = GridWorldGenerator(4, num_walls=0)
    assert bool(generator._connected(generator.grid_mask))

    # Cut both edges of corner 0 (to its right neighbour 1 and its lower neighbour 4).
    cut = generator.grid_mask.at[0, 1].set(0).at[1, 0].set(0).at[0, 4].set(0).at[4, 0].set(0)
    assert not bool(generator._connected(cut))
