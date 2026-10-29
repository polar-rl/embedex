import os
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from embedex.graphs.config import Config
from embedex.graphs.embedding import GraphEmbedding
from embedex.graphs.generator import Generator
from embedex.graphs.metrics import (
    average_precision,
    confusion_counts,
    precision_recall_f1,
    roc_auc,
    spearman,
)
from embedex.graphs.reconstructor import (
    MediansNetwork,
    TransitionNetwork,
    recon_average_degree,
    recon_medians,
    recon_transitions,
    recon_vertex_degrees,
    transition_distances,
)
from embedex.graphs.visualize import plot_geometry
from embedex.logger import safe_log


def evaluate_embeddings(
    cfg: Config,
    embedding_network: GraphEmbedding,
    generator: Generator,
    key: jax.Array,
) -> None:
    key, sample_key, plot_key = jax.random.split(key, 3)

    num_graphs = cfg.evaluation.num_eval_graphs
    num_samples = cfg.evaluation.num_eval_samples

    edges, adjs = generator.sample(num_graphs, num_samples, sample_key)
    geometry, base_rate = score_geometry(embedding_network, edges, adjs)
    report_geometry(cfg, geometry, base_rate)

    plot_geometry(
        embedding_network,
        generator,
        cfg.evaluation.num_plot_graphs,
        cfg.model.input_size,
        num_samples,
        os.path.join(cfg.figure_dir, cfg.run_name),
        plot_key,
    )


def evaluate_reconstruction(
    cfg: Config,
    embedding_network: GraphEmbedding,
    medians_network: MediansNetwork,
    generator: Generator,
    plot_fn: Callable,
    key: jax.Array,
) -> None:
    key, sample_key, plot_key = jax.random.split(key, 3)

    num_graphs = cfg.evaluation.num_eval_graphs
    num_samples = cfg.evaluation.num_eval_samples

    edges, adjs = generator.sample(num_graphs, num_samples, sample_key)
    results, base_rate = score_reconstructions(embedding_network, medians_network, edges, adjs)
    report_reconstructions(cfg, results, base_rate, num_graphs)

    plot_fn(
        embedding_network,
        medians_network,
        generator,
        cfg.evaluation.num_plot_graphs,
        cfg.model.input_size,
        num_samples,
        os.path.join(cfg.figure_dir, cfg.run_name),
        plot_key,
    )


def evaluate_transitions(
    cfg: Config,
    embedding_network: GraphEmbedding,
    transition_network: TransitionNetwork,
    generator: Generator,
    plot_fn: Callable,
    key: jax.Array,
) -> None:
    key, sample_key, plot_key = jax.random.split(key, 3)

    num_graphs = cfg.evaluation.num_eval_graphs
    num_samples = cfg.evaluation.num_eval_samples

    edges, adjs = generator.sample(num_graphs, num_samples, sample_key)
    results = score_transitions(embedding_network, transition_network, edges, adjs)
    report_transitions(cfg, results)

    plot_fn(
        embedding_network,
        transition_network,
        generator,
        cfg.evaluation.num_plot_graphs,
        cfg.model.input_size,
        num_samples,
        os.path.join(cfg.figure_dir, cfg.run_name),
        plot_key,
    )


@eqx.filter_jit
def score_geometry(
    embedding_network: GraphEmbedding, edges: jax.Array, adjs: jax.Array
) -> tuple[dict[str, jax.Array], jax.Array]:
    def _per_graph(edges, adj):
        n = adj.shape[0]
        z = embedding_network.embed(edges)
        distances = embedding_network.distance_matrix(jnp.eye(n), z)

        indices = jnp.triu_indices(n, k=1)
        proximity = (-distances)[indices]
        weights = adj[indices]
        is_edge = weights > 0.0

        num_pairs = n * (n - 1) // 2
        num_zeros = num_pairs - jnp.sum(is_edge)
        ceiling = jnp.sqrt(1.0 - (num_zeros**3 - num_zeros) / (num_pairs**3 - num_pairs))

        return (
            roc_auc(proximity, is_edge),
            average_precision(proximity, is_edge),
            spearman(proximity, weights) / ceiling,
            spearman(proximity, weights, is_edge),
            jnp.mean(is_edge),
        )

    aucs, aps, weight_rhos, weight_rhos_edges, edge_fracs = jax.vmap(_per_graph)(edges, adjs)

    geometry = {
        'roc_auc': jnp.nanmean(aucs),
        'avg_precision': jnp.nanmean(aps),
        'weight_spearman': jnp.nanmean(weight_rhos),
        'weight_spearman_edges': jnp.nanmean(weight_rhos_edges),
        'num_graphs_scored': jnp.sum(~jnp.isnan(aucs)),
    }

    return geometry, jnp.mean(edge_fracs)


@eqx.filter_jit
def score_reconstructions(
    embedding_network: GraphEmbedding,
    medians_network: MediansNetwork,
    edges: jax.Array,
    adjs: jax.Array,
) -> tuple[dict[str, dict[str, jax.Array]], jax.Array]:
    def _per_graph(edges, adj):
        num_vertices = adj.shape[0]
        vertices = jnp.eye(num_vertices)

        z = embedding_network.embed(edges)
        distances = embedding_network.distance_matrix(vertices, z)
        medians = jax.vmap(medians_network, in_axes=(0, None))(vertices, z)
        ground_truth = adj > 0.5
        off_diag = jnp.triu(jnp.ones((num_vertices, num_vertices), jnp.bool_), k=1)

        preds = {
            'vertex_degrees': recon_vertex_degrees(
                distances, jnp.sum(adj, axis=1).astype(jnp.int32)
            ),
            'average_degree': recon_average_degree(distances, jnp.sum(adj) / num_vertices),
            'medians': recon_medians(distances, medians),
        }
        counts_by_strategy = {
            name: confusion_counts(pred, ground_truth, off_diag) for name, pred in preds.items()
        }
        edge_frac = jnp.mean(ground_truth[jnp.triu_indices(num_vertices, k=1)])

        return counts_by_strategy, edge_frac

    counts_by_strategy, edge_fracs = jax.vmap(_per_graph)(edges, adjs)

    totals = jax.tree.map(lambda c: jnp.sum(c, axis=0), counts_by_strategy)
    results = {name: precision_recall_f1(*c) for name, c in totals.items()}

    return results, jnp.mean(edge_fracs)


@eqx.filter_jit
def score_transitions(
    embedding_network: GraphEmbedding,
    transition_network: TransitionNetwork,
    edges: jax.Array,
    adjs: jax.Array,
) -> dict[str, jax.Array]:
    def _per_graph(edges, adj):
        n = adj.shape[0]
        z = embedding_network.embed(edges)
        distances = transition_distances(embedding_network, z, n)

        target = adj / jnp.sum(adj)
        true = target.reshape(-1)
        edge_mask = true > 0
        off_diag = (~jnp.eye(n, dtype=jnp.bool_)).reshape(-1)

        num_pairs = n * n - n
        num_zeros = num_pairs - jnp.sum(edge_mask)
        ceiling = jnp.sqrt(1.0 - (num_zeros**3 - num_zeros) / (num_pairs**3 - num_pairs))

        def _errors(predicted):
            flat = predicted.reshape(-1)
            rho_edges = spearman(flat, true, edge_mask)
            rho_all = spearman(flat, true, off_diag) / ceiling

            diff = (predicted - target) * off_diag.reshape(n, n)
            mae = jnp.sum(jnp.abs(diff)) / num_pairs
            rmse = jnp.sqrt(jnp.sum(diff**2) / num_pairs)

            return rho_edges, rho_all, mae, rmse

        return _errors(transition_network(distances, z)), _errors(recon_transitions(distances))

    (re_l, ra_l, mae_l, rmse_l), (re_o, ra_o, mae_o, rmse_o) = jax.vmap(_per_graph)(edges, adjs)

    return {
        'learned_spearman': jnp.nanmean(re_l),
        'learned_spearman_all': jnp.nanmean(ra_l),
        'learned_mae': jnp.mean(mae_l),
        'learned_rmse': jnp.mean(rmse_l),
        'oracle_spearman': jnp.nanmean(re_o),
        'oracle_spearman_all': jnp.nanmean(ra_o),
        'oracle_mae': jnp.mean(mae_o),
        'oracle_rmse': jnp.mean(rmse_o),
    }


def report_geometry(cfg: Config, geometry: dict[str, jax.Array], base_rate: jax.Array) -> None:
    print(
        f'Geometry   '
        f'ROC-AUC {geometry["roc_auc"]:.3f}   AP {geometry["avg_precision"]:.3f}   '
        f'(Chance AP {base_rate:.3f}, macro-averaged over {geometry["num_graphs_scored"]} graphs)'
    )

    # The weight metrics are meaningful only for weighted graphs
    weighted = not bool(jnp.isnan(geometry['weight_spearman_edges']))
    if weighted:
        print(
            f'Weights    '
            f'Spearman(prox, w) {geometry["weight_spearman"]:.3f} (all, rescaled)   '
            f'edges-only {geometry["weight_spearman_edges"]:.3f}   '
            f'(rank corr of proximity vs edge weight)'
        )

    if hasattr(cfg, 'run'):
        safe_log(cfg.run, 'eval/geometry/roc_auc', geometry['roc_auc'])
        safe_log(cfg.run, 'eval/geometry/avg_precision', geometry['avg_precision'])
        safe_log(cfg.run, 'eval/geometry/base_rate', base_rate)

        if weighted:
            safe_log(cfg.run, 'eval/geometry/weight_spearman', geometry['weight_spearman'])
            safe_log(
                cfg.run, 'eval/geometry/weight_spearman_edges', geometry['weight_spearman_edges']
            )


def report_reconstructions(
    cfg: Config, results: dict[str, jax.Array], base_rate: jax.Array, num_graphs: int
) -> None:
    notes = {
        'vertex_degrees': 'Full-graph per-vertex degrees',
        'average_degree': 'Full-graph average degree',
        'medians': 'Learned medians net',
    }

    print(f'Reconstruction   ({num_graphs} graphs, edge base rate {base_rate:.3f})')
    for name, note in notes.items():
        m = results[name]
        print(
            f'    {name:<15} P {results[name]["precision"]:.3f}   '
            f'R {results[name]["recall"]:.3f}   F1 {results[name]["f1"]:.3f}   ({note})'
        )

    if hasattr(cfg, 'run'):
        for name, m in results.items():
            safe_log(cfg.run, f'eval/reconstruction/{name}/precision', m['precision'])
            safe_log(cfg.run, f'eval/reconstruction/{name}/recall', m['recall'])
            safe_log(cfg.run, f'eval/reconstruction/{name}/f1', m['f1'])

        safe_log(cfg.run, 'eval/reconstruction/base_rate', base_rate)


def report_transitions(cfg: Config, results: dict[str, jax.Array]) -> None:
    print('Reconstruction')
    print(
        f'    Spearman (edges)          learned {results["learned_spearman"]:.3f}   '
        f'oracle {results["oracle_spearman"]:.3f}'
    )
    print(
        f'    Spearman (all, rescaled)  learned {results["learned_spearman_all"]:.3f}   '
        f'oracle {results["oracle_spearman_all"]:.3f}'
    )
    print(
        f'    MAE  (off-diag)           learned {results["learned_mae"]:.4f}   '
        f'oracle {results["oracle_mae"]:.4f}'
    )
    print(
        f'    RMSE (off-diag)           learned {results["learned_rmse"]:.4f}   '
        f'oracle {results["oracle_rmse"]:.4f}'
    )

    if hasattr(cfg, 'run'):
        for key, value in results.items():
            safe_log(cfg.run, f'eval/transition/{key}', value)
