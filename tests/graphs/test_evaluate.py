"""Evaluation of graph embeddings: scoring geometry, reconstructions and spectra, and reporting."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.graphs.builder import create_embedding_network
from embedex.graphs.evaluate import (
    report_geometry,
    report_reconstructions,
    report_spectrum,
    report_transitions,
    score_geometry,
    score_reconstructions,
    score_spectrum,
    score_transitions,
)

NUM_GRAPHS, NUM_SAMPLES, NUM_DIMS = 8, 512, 4
GEOMETRY_KEYS = {'roc_auc', 'avg_precision', 'weight_spearman', 'weight_spearman_edges'}
SPECTRUM_KEYS = {'eigvec_cosine', 'eigval_rel_error', 'eigvec_subspace'}
TRANSITION_KEYS = {
    f'{source}_{metric}'
    for source in ('learned', 'oracle')
    for metric in ('spearman', 'spearman_all', 'mae', 'rmse')
}


class ExactEmbedding(eqx.Module):
    """Stands in for a perfectly trained `GraphEmbedding` on unweighted graphs.

    It counts how often each pair was sampled, which recovers the graph exactly once every edge
    has been seen. Pairs sampled more often are closer, and the eigensystem is that of the true
    Laplacian `I - (P + P^T) / 2`, as `score_spectrum` defines it.
    """

    num_vertices: int = eqx.field(static=True)

    def embed(self, edges: jax.Array) -> jax.Array:
        counts = jnp.transpose(edges[:, 0]) @ edges[:, 1]
        return (counts + jnp.transpose(counts)).reshape(-1)

    def distance_matrix(self, vertices: jax.Array, z: jax.Array) -> jax.Array:
        return -z.reshape(self.num_vertices, self.num_vertices)

    def _eigh(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        adj = (z.reshape(self.num_vertices, self.num_vertices) > 0).astype(jnp.float32)
        transition = adj / jnp.sum(adj, axis=1, keepdims=True)
        return jnp.linalg.eigh(jnp.eye(self.num_vertices) - 0.5 * (transition + transition.T))

    def eigenvalues(self, z: jax.Array) -> jax.Array:
        return self._eigh(z)[0][:NUM_DIMS]

    def eigenfunctions(self, vertex: jax.Array, z: jax.Array) -> jax.Array:
        return vertex @ self._eigh(z)[1][:, :NUM_DIMS]


@pytest.fixture
def graphs(family, key):
    return family.generator().sample(NUM_GRAPHS, NUM_SAMPLES, key)


@pytest.fixture
def untrained(family, make_config, key):
    """Freshly initialised (embedding, reconstruction head) for the family."""

    cfg = make_config(family)
    embedding_key, head_key = jax.random.split(key)
    return create_embedding_network(cfg, embedding_key), family.create_reconstructor(cfg, head_key)


def edge_fraction(adjs: jax.Array) -> float:
    """Fraction of vertex pairs joined by an edge, averaged over graphs."""

    rows, cols = np.triu_indices(adjs.shape[-1], k=1)
    return float(np.mean(np.asarray(adjs)[:, rows, cols] > 0))


def in_unit_interval(value) -> bool:
    return 0.0 <= float(value) <= 1.0


class TestGeometry:
    def test_scores_are_probabilities(self, family, graphs, untrained):
        geometry, base_rate = score_geometry(untrained[0], *graphs)

        assert set(geometry) >= GEOMETRY_KEYS
        assert in_unit_interval(geometry['roc_auc'])
        assert in_unit_interval(geometry['avg_precision'])
        assert int(geometry['num_graphs_scored']) == NUM_GRAPHS
        assert float(base_rate) == pytest.approx(edge_fraction(graphs[1]))

    def test_weight_correlations_are_defined_only_for_weighted_graphs(
        self, family, graphs, untrained
    ):
        geometry, _ = score_geometry(untrained[0], *graphs)

        assert np.isnan(geometry['weight_spearman_edges']) != family.weighted

    def test_exact_geometry_separates_edges_perfectly(self, binary_family, key):
        edges, adjs = binary_family.generator().sample(NUM_GRAPHS, NUM_SAMPLES, key)

        geometry, _ = score_geometry(ExactEmbedding(binary_family.num_vertices), edges, adjs)

        assert float(geometry['roc_auc']) == pytest.approx(1.0)
        assert float(geometry['avg_precision']) == pytest.approx(1.0)


class TestReconstructions:
    def test_scores_are_probabilities(self, binary_family, make_config, key):
        cfg = make_config(binary_family)
        edges, adjs = binary_family.generator().sample(NUM_GRAPHS, NUM_SAMPLES, key)
        embedding = create_embedding_network(cfg, key)
        medians = binary_family.create_reconstructor(cfg, key)

        results, base_rate = score_reconstructions(embedding, medians, edges, adjs)

        assert set(results) == {'vertex_degrees', 'average_degree', 'medians'}
        for metrics in results.values():
            assert all(in_unit_interval(metrics[name]) for name in ('precision', 'recall', 'f1'))
        assert float(base_rate) == pytest.approx(edge_fraction(adjs))


class TestTransitions:
    def test_scores_are_well_formed(self, weighted_family, make_config, key):
        cfg = make_config(weighted_family)
        edges, adjs = weighted_family.generator().sample(NUM_GRAPHS, NUM_SAMPLES, key)
        embedding = create_embedding_network(cfg, key)
        transitions = weighted_family.create_reconstructor(cfg, key)

        results = score_transitions(embedding, transitions, edges, adjs)

        assert set(results) == TRANSITION_KEYS
        for source in ('learned', 'oracle'):
            assert -1.0 <= float(results[f'{source}_spearman']) <= 1.0
            assert 0.0 <= float(results[f'{source}_mae']) <= float(results[f'{source}_rmse'])


class TestSpectrum:
    def test_scores_are_in_range(self, family, graphs, untrained):
        spectrum = score_spectrum(untrained[0], *graphs)

        assert set(spectrum) == SPECTRUM_KEYS
        assert in_unit_interval(spectrum['eigvec_cosine'])
        assert in_unit_interval(spectrum['eigvec_subspace'])
        assert float(spectrum['eigval_rel_error']) >= 0.0

    def test_exact_eigensystem_is_recovered_perfectly(self, binary_family, key):
        edges, adjs = binary_family.generator().sample(NUM_GRAPHS, NUM_SAMPLES, key)

        spectrum = score_spectrum(ExactEmbedding(binary_family.num_vertices), edges, adjs)

        assert float(spectrum['eigvec_cosine']) == pytest.approx(1.0, abs=1e-4)
        assert float(spectrum['eigvec_subspace']) == pytest.approx(1.0, abs=1e-4)
        assert float(spectrum['eigval_rel_error']) == pytest.approx(0.0, abs=1e-4)


class TestReports:
    GEOMETRY = {
        'roc_auc': jnp.asarray(0.9),
        'avg_precision': jnp.asarray(0.8),
        'weight_spearman': jnp.asarray(0.5),
        'num_graphs_scored': jnp.asarray(8),
    }

    @pytest.mark.parametrize('weighted', [False, True])
    def test_geometry_reports_weights_only_for_weighted_graphs(
        self, weighted, make_config, fake_run, capsys
    ):
        cfg = make_config()
        cfg.run = fake_run
        spearman_edges = jnp.asarray(0.4 if weighted else jnp.nan)

        report_geometry(cfg, self.GEOMETRY | {'weight_spearman_edges': spearman_edges}, 0.2)

        printed = capsys.readouterr().out
        logged = set(fake_run.series)
        assert 'Geometry' in printed
        assert {'eval/geometry/roc_auc', 'eval/geometry/avg_precision'} <= logged
        assert ('Weights' in printed) == weighted
        assert ('eval/geometry/weight_spearman_edges' in logged) == weighted

    def test_reconstructions_log_precision_recall_and_f1_per_rule(self, make_config, fake_run):
        cfg = make_config()
        cfg.run = fake_run
        metrics = {'precision': 0.5, 'recall': 0.25, 'f1': 1 / 3}
        results = {name: metrics for name in ('vertex_degrees', 'average_degree', 'medians')}

        report_reconstructions(cfg, results, 0.2, num_graphs=8)

        assert fake_run.series['eval/reconstruction/medians/recall'] == [0.25]
        assert len(fake_run.series) == 3 * 3 + 1  # three metrics per rule, plus the base rate

    @pytest.mark.parametrize(
        'report, keys, prefix',
        [
            (report_spectrum, SPECTRUM_KEYS, 'spectrum'),
            (report_transitions, TRANSITION_KEYS, 'transition'),
        ],
        ids=['spectrum', 'transitions'],
    )
    def test_every_score_is_logged(self, report, keys, prefix, make_config, fake_run):
        cfg = make_config()
        cfg.run = fake_run

        report(cfg, {k: jnp.asarray(0.5) for k in keys})

        assert set(fake_run.series) == {f'eval/{prefix}/{k}' for k in keys}

    def test_reports_work_without_a_run(self, make_config, capsys):
        report_spectrum(make_config(), {k: jnp.asarray(0.5) for k in SPECTRUM_KEYS})

        assert 'Spectrum' in capsys.readouterr().out
