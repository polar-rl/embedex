import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from embedex.graphs.embedding import GraphEmbedding
from embedex.graphs.generator import Generator
from embedex.graphs.reconstructor import (
    MediansNetwork,
    recon_average_degree,
    recon_medians,
    recon_vertex_degrees,
)
from embedex.plotting import save_figure

# A single graph's layout: node index → (x, y) position.
Layout = dict[int, tuple[float, float]]


def plot_geometry(
    embedding_network: GraphEmbedding,
    generator: Generator,
    num_graphs: int,
    num_vertices: int,
    num_samples: int,
    save_path: str,
    key: jax.Array,
) -> None:
    vertices = jnp.eye(num_vertices)
    edges, adjs = generator.sample(num_graphs, num_samples, key)
    adjacencies = list(np.asarray(adjs))

    distance_matrices = [
        np.asarray(embedding_network.distance_matrix(vertices, embedding_network.embed(e)))
        for e in edges
    ]

    fig, _, positions = create_graph_plot(adjacencies, title='Ground Truth')
    save_figure(fig, os.path.join(save_path, 'ground_truth_geometry.pdf'))

    fig, _ = create_geometry_plot(distance_matrices, positions)
    save_figure(fig, os.path.join(save_path, 'geometry.pdf'))


def plot_reconstructions(
    embedding_network: GraphEmbedding,
    medians_network: MediansNetwork,
    generator: Generator,
    num_graphs: int,
    num_vertices: int,
    num_samples: int,
    save_path: str,
    key: jax.Array,
) -> None:
    vertices = jnp.eye(num_vertices)
    edges, adjs = generator.sample(num_graphs, num_samples, key)
    adjacencies = list(np.asarray(adjs))

    embeddings = [embedding_network.embed(e) for e in edges]
    distances = [embedding_network.distance_matrix(vertices, z) for z in embeddings]
    medians = [jax.vmap(medians_network, in_axes=(0, None))(vertices, z) for z in embeddings]

    # Ground truth, saved on its own; reuse its layout for every reconstruction.
    fig, _, positions = create_graph_plot(adjacencies, title='Ground Truth')
    save_figure(fig, os.path.join(save_path, 'ground_truth_reconstruction.pdf'))

    strategies = {
        'vertex_degrees': [
            np.asarray(recon_vertex_degrees(d, jnp.sum(a, axis=0, dtype=jnp.int32)))
            for d, a in zip(distances, adjacencies, strict=True)
        ],
        'average_degree': [
            np.asarray(recon_average_degree(d, jnp.sum(a) / num_vertices))
            for d, a in zip(distances, adjacencies, strict=True)
        ],
        'medians': [
            np.asarray(recon_medians(d, m)) for d, m in zip(distances, medians, strict=True)
        ],
    }

    for name, preds in strategies.items():
        fig, _, _ = create_graph_plot(preds, positions, f'Reconstruction ({name})')
        save_figure(fig, os.path.join(save_path, f'reconstruction_{name}.pdf'))


def create_geometry_plot(
    distance_matrices: list[np.ndarray],
    positions: list[Layout],
) -> tuple[Figure, Axes]:
    num_graphs = len(distance_matrices)
    num_cols = int(np.ceil(np.sqrt(num_graphs)))
    num_rows = int(np.ceil(num_graphs / num_cols))

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(num_cols * 3.5, num_rows * 3.5))

    for i, dist_matrix in enumerate(distance_matrices):
        num_vertices = dist_matrix.shape[0]
        colour = cycle[i % len(cycle)]
        pos = positions[i]
        pair_indices = np.triu_indices(num_vertices, k=1)

        # Closeness with a sharp Gaussian falloff (bandwidth = a low distance
        # quantile) so near edges stay bold and far ones drop off fast.
        scale = np.quantile(dist_matrix[pair_indices], 0.3) + 1e-12
        weight = np.exp(-((dist_matrix[pair_indices] / scale) ** 2))

        segments = [[pos[int(a)], pos[int(b)]] for a, b in zip(*pair_indices, strict=True)]
        rgba = np.tile((*to_rgb(colour), 0.0), (len(segments), 1))
        rgba[:, 3] = np.clip(weight * 0.9, 0.0, 1.0)  # 0.9 = max edge opacity
        visible = rgba[:, 3] > 0.03  # cull edges too faint to see (display only)

        ax.add_collection(
            LineCollection(
                [s for s, v in zip(segments, visible, strict=True) if v],
                colors=rgba[visible],
                linewidths=(0.2 + 3.5 * weight)[visible],
            )
        )

        xs = [pos[k][0] for k in range(num_vertices)]
        ys = [pos[k][1] for k in range(num_vertices)]
        ax.scatter(xs, ys, s=300.0, facecolors='white', edgecolors=colour, linewidths=1.2, zorder=3)

        for k in range(num_vertices):
            ax.text(xs[k], ys[k], str(k), ha='center', va='center', fontsize=8, zorder=4)

    pad = 0.2
    all_x = [x for p in positions for x, _ in p.values()]
    all_y = [y for p in positions for _, y in p.values()]

    ax.set_title('Predicted Geometry')
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax


def create_graph_plot(
    adjacencies: list[np.ndarray],
    positions: list[Layout] | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes, list[Layout]]:
    num_graphs = len(adjacencies)
    num_cols = int(np.ceil(np.sqrt(num_graphs)))
    num_rows = int(np.ceil(num_graphs / num_cols))

    masks = [a.astype(bool) for a in adjacencies]
    if positions is None:
        positions = _grid_positions(masks, num_cols)

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(num_cols * 3.5, num_rows * 3.5))

    for i, mask in enumerate(masks):
        colour = cycle[i % len(cycle)]
        graph = nx.from_numpy_array(mask.astype(np.int32))
        nx.draw(
            graph,
            pos=positions[i],
            ax=ax,
            with_labels=True,
            node_color='white',
            edgecolors=colour,
            edge_color=colour,
            node_size=300.0,
            font_size=8,
        )

    pad = 0.2
    all_x = [x for p in positions for x, _ in p.values()]
    all_y = [y for p in positions for _, y in p.values()]

    if title is not None:
        ax.set_title(title)

    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax, positions


def _grid_positions(masks: list[np.ndarray], num_cols: int) -> list[Layout]:
    positions, step = [], 1.0 + 0.3  # 0.3 = inter-graph gap, as a fraction of a graph's extent

    for i, mask in enumerate(masks):
        base = _normalize_positions(_layout_positions(mask))
        row, col = divmod(i, num_cols)
        positions.append({n: (x + col * step, y - row * step) for n, (x, y) in base.items()})

    return positions


def _normalize_positions(positions: dict[int, np.ndarray]) -> Layout:
    xs = np.array([p[0] for p in positions.values()])
    ys = np.array([p[1] for p in positions.values()])
    scale = 1.0 / max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-9)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0

    return {n: ((x - cx) * scale + 0.5, (y - cy) * scale + 0.5) for n, (x, y) in positions.items()}


def _layout_positions(mask: np.ndarray) -> dict[int, np.ndarray]:
    return nx.spring_layout(nx.from_numpy_array(mask.astype(np.int32)), seed=0)
