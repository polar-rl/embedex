"""Final evaluation of trained networks: scoring, reporting and plotting, for ODE families and
per polynomial degree."""

from pathlib import Path

import jax
import numpy as np
import pytest

from embedex.functions.builder import create_embedding_network, create_operator_network
from embedex.functions.evaluation_ode import (
    evaluate_embeddings,
    evaluate_solution_operator,
    report_metrics,
)
from embedex.functions.families import antiderivatives, derivatives
from embedex.functions.families.antiderivatives import PolynomialAntiderivativeGenerator
from embedex.functions.families.derivatives import PolynomialDerivativeGenerator

METRICS = ('rmse', 'rel_l2', 'rel_max')


@pytest.fixture
def networks(make_config, fake_run, key):
    """A config whose run is `fake_run`, plus untrained networks for the given generator."""

    def _make(family, generator):
        cfg = make_config(family)
        cfg.run = fake_run
        embedding_key, operator_key = jax.random.split(key)
        embedding = create_embedding_network(cfg, embedding_key)
        return cfg, embedding, create_operator_network(cfg, generator, operator_key)

    return _make


def figure_dir(cfg) -> Path:
    return Path(cfg.figure_dir) / cfg.run_name


def logged_degrees(run, stage: str) -> set[str]:
    prefix = f'eval/{stage}/rmse/'
    return {name.removeprefix(prefix) for name in run.series if name.startswith(prefix)}


class TestOdeEvaluation:
    @staticmethod
    def check_plot(cfg, generator, plot, stage: str) -> None:
        _, ts, ys, grid, preds, rmse = plot
        num_plots, num_samples = cfg.evaluation.num_plot_functions, cfg.evaluation.num_eval_samples

        assert ts.shape == (num_plots, num_samples, 1)
        assert ys.shape == (num_plots, num_samples, cfg.model.output_size)
        assert grid.shape == (256, 1)
        np.testing.assert_allclose(grid[[0, -1], 0], generator.t_range)
        assert preds.shape == (num_plots, 256, cfg.model.output_size)
        assert float(rmse) == pytest.approx(cfg.run.series[f'eval/{stage}/rmse'][0])

    def test_embedding_is_scored_and_its_reconstructions_plotted(
        self, ode_family, networks, fake_run, key
    ):
        generator = ode_family.generator()
        cfg, embedding, _ = networks(ode_family, generator)
        plots = []

        evaluate_embeddings(cfg, embedding, generator, lambda *a: plots.append(a), key)

        assert set(fake_run.series) == {f'eval/embedding/{m}' for m in METRICS}
        (plot,) = plots
        self.check_plot(cfg, generator, plot, 'embedding')
        # Each curve decodes the embedding of the first half of the samples plotted with it.
        _, ts, ys, grid, preds, _ = plot
        half = cfg.evaluation.num_eval_samples // 2
        expected = embedding.eval(grid, embedding.embed(ts[0, :half], ys[0, :half]))
        np.testing.assert_allclose(preds[0], expected, rtol=1e-5, atol=1e-6)

    def test_operator_is_scored_and_its_predictions_plotted(
        self, ode_family, networks, fake_run, key
    ):
        generator = ode_family.generator()
        cfg, embedding, operator = networks(ode_family, generator)
        plots = []

        evaluate_solution_operator(
            cfg, embedding, operator, generator, lambda *a: plots.append(a), key
        )

        assert set(fake_run.series) == {f'eval/operator/{m}' for m in METRICS}
        (plot,) = plots
        self.check_plot(cfg, generator, plot, 'operator')


def test_report_prints_and_logs_every_metric(make_config, fake_run, capsys):
    cfg = make_config()
    cfg.run = fake_run

    report_metrics(cfg, 'operator', 0.5, 0.25, 0.75)

    assert 'Operator: RMSE: 0.5; Rel L2: 0.25; Rel Max: 0.75' in capsys.readouterr().out
    assert fake_run.series == {
        'eval/operator/rmse': [0.5],
        'eval/operator/rel_l2': [0.25],
        'eval/operator/rel_max': [0.75],
    }


def test_report_works_without_a_run(make_config, capsys):
    report_metrics(make_config(), 'embedding', 0.5, 0.25, 0.75)

    assert 'Embedding: RMSE' in capsys.readouterr().out


class TestPolynomialEvaluation:
    def test_embedding_is_scored_per_degree_and_overall(self, networks, fake_run, key):
        generator = PolynomialDerivativeGenerator((-4.0, 4.0), (-5.0, 5.0), 0, 3)
        cfg, embedding, _ = networks(None, generator)

        derivatives.evaluate_embeddings(cfg, embedding, generator, key=key)

        degrees = [f'degree_{d}' for d in range(4)]
        assert logged_degrees(fake_run, 'embedding') == {*degrees, 'overall'}
        for metric in METRICS:
            per_degree = [fake_run.series[f'eval/embedding/{metric}/{d}'][0] for d in degrees]
            overall = fake_run.series[f'eval/embedding/{metric}/overall'][0]
            assert overall == pytest.approx(np.mean(per_degree), rel=1e-5)
        assert (figure_dir(cfg) / 'polynomial_embeddings_by_degree.png').exists()

    @pytest.mark.parametrize(
        'generator_cls, evaluate',
        [
            (
                PolynomialDerivativeGenerator,
                lambda cfg, e, o, g, key: derivatives.evaluate_embeddings(cfg, e, g, key=key),
            ),
            (PolynomialDerivativeGenerator, derivatives.evaluate_differential_operator),
            (PolynomialAntiderivativeGenerator, antiderivatives.evaluate_antiderivative_operator),
        ],
        ids=['embedding', 'differential_operator', 'antidifferential_operator'],
    )
    def test_functions_are_sampled_as_densely_as_the_evaluation_asks(
        self, generator_cls, evaluate, networks, monkeypatch, key
    ):
        generator = generator_cls((-4.0, 4.0), (-5.0, 5.0), 0, 3)
        cfg, embedding, operator = networks(None, generator)
        cfg.evaluation.num_eval_samples = 2 * cfg.training.num_samples
        sample_degree, sample_counts = generator.sample_degree, set()

        def record(degree, batch_size, num_samples, key):
            sample_counts.add(num_samples)
            return sample_degree(degree, batch_size, num_samples, key)

        monkeypatch.setattr(generator, 'sample_degree', record)
        evaluate(cfg, embedding, operator, generator, key=key)

        assert sample_counts == {cfg.evaluation.num_eval_samples}

    @pytest.mark.parametrize('min_degree', [0, 1])
    def test_differential_operator_is_scored_on_the_degrees_it_trains_on(
        self, min_degree, networks, fake_run, key
    ):
        # A constant's derivative is zero everywhere, which the relative errors cannot score.
        generator = PolynomialDerivativeGenerator((-4.0, 4.0), (-5.0, 5.0), min_degree, 3)
        cfg, embedding, operator = networks(None, generator)

        derivatives.evaluate_differential_operator(cfg, embedding, operator, generator, key=key)

        expected = {f'degree_{d}' for d in range(min_degree, 4)} | {'overall'}
        assert logged_degrees(fake_run, 'operator') == expected
        assert (figure_dir(cfg) / 'polynomial_differential_operator_by_degree.png').exists()

    def test_antiderivative_operator_is_scored_on_every_degree(self, networks, fake_run, key):
        generator = PolynomialAntiderivativeGenerator((-4.0, 4.0), (-5.0, 5.0), 0, 3)
        cfg, embedding, operator = networks(None, generator)

        antiderivatives.evaluate_antiderivative_operator(
            cfg, embedding, operator, generator, key=key
        )

        expected = {f'degree_{d}' for d in range(4)} | {'overall'}
        assert logged_degrees(fake_run, 'operator') == expected
        assert (figure_dir(cfg) / 'polynomial_antidifferential_operator_by_degree.png').exists()
