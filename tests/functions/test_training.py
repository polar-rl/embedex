"""Training function embeddings and the operators that act on them."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from embedex.checkpointing import load_ckpt, save_ckpt
from embedex.config import build_dataclass
from embedex.functions import training
from embedex.functions.builder import (
    build_from_config,
    create_embedding_network,
    create_operator_network,
)
from embedex.functions.embedding import FunctionEmbedding
from embedex.functions.families.logistic import LogisticGenerator
from embedex.functions.operator import OperatorMapping
from embedex.functions.training import (
    run_training_pipeline,
    train_function_embeddings,
    train_operator_jointly,
    train_operator_network,
)

METRICS = ('rmse', 'rel_l2', 'rel_max')


def is_finite(tree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(tree))


def params(network) -> list[jax.Array]:
    return jax.tree.leaves(eqx.filter(network, eqx.is_inexact_array))


def changed(before, after) -> bool:
    """Whether training moved any parameter. Training donates its input network, so `before`
    must be a separately built copy, not the network that was trained."""

    return not all(
        jnp.array_equal(a, b) for a, b in zip(params(before), params(after), strict=True)
    )


class TestTrainingStages:
    def test_embedding_training_updates_the_network(self, family, make_config, key):
        cfg = make_config(family)
        init_key, train_key = jax.random.split(key)

        trained = train_function_embeddings(
            cfg, create_embedding_network(cfg, init_key), family.generator(), train_key
        )

        assert isinstance(trained, FunctionEmbedding)
        assert changed(create_embedding_network(cfg, init_key), trained)
        assert is_finite(params(trained))

    def test_operator_training_updates_only_the_operator(self, family, make_config, key):
        cfg = make_config(family)
        generator = family.generator()
        init_key, train_key = jax.random.split(key)
        embedding = create_embedding_network(cfg, init_key)

        trained = train_operator_network(
            cfg, embedding, create_operator_network(cfg, generator, init_key), generator, train_key
        )

        assert isinstance(trained, OperatorMapping)
        assert changed(create_operator_network(cfg, generator, init_key), trained)
        assert not changed(create_embedding_network(cfg, init_key), embedding)  # stays frozen
        assert is_finite(params(trained))

    def test_joint_training_updates_both_networks(self, make_config, key):
        cfg = make_config()
        generator = LogisticGenerator((0.0, 5.0), (0.1, 2.0), (0.5, 3.0), (1.0, 5.0))
        init_key, train_key = jax.random.split(key)

        embedding, operator = train_operator_jointly(
            cfg,
            create_embedding_network(cfg, init_key),
            create_operator_network(cfg, generator, init_key),
            generator,
            train_key,
        )

        assert changed(create_embedding_network(cfg, init_key), embedding)
        assert changed(create_operator_network(cfg, generator, init_key), operator)

    @pytest.mark.parametrize('stage', ['embedding', 'operator', 'joint'])
    def test_training_logs_every_loss_and_periodic_progress(
        self, stage, make_config, fake_run, key
    ):
        cfg = make_config()
        cfg.run = fake_run
        cfg.training.eval_every = 2
        generator = LogisticGenerator((0.0, 5.0), (0.1, 2.0), (0.5, 3.0), (1.0, 5.0))
        embedding = create_embedding_network(cfg, key)
        operator = create_operator_network(cfg, generator, key)

        if stage == 'embedding':
            train_function_embeddings(cfg, embedding, generator, key)
        elif stage == 'operator':
            train_operator_network(cfg, embedding, operator, generator, key)
        else:
            train_operator_jointly(cfg, embedding, operator, generator, key)

        logged = 'embedding' if stage == 'embedding' else 'operator'
        assert len(fake_run.series[f'train/{logged}/loss']) == 4
        for metric in METRICS:
            assert len(fake_run.series[f'progress/{logged}/{metric}']) == 2


def test_embedding_training_lowers_the_held_out_error(make_config, key):
    # A sign error in the update would still change the parameters, so check the direction.
    cfg = make_config()
    cfg.training.num_iterations, cfg.training.learning_rate = 50, 3e-3
    generator = LogisticGenerator((0.0, 5.0), (0.1, 2.0), (0.5, 3.0), (1.0, 5.0))
    init_key, train_key, eval_key = jax.random.split(key, 3)
    ts, ys, _ = generator.sample(64, 32, eval_key)
    before = training.evaluate_rel_l2(create_embedding_network(cfg, init_key), ts, ys)

    trained = train_function_embeddings(
        cfg, create_embedding_network(cfg, init_key), generator, train_key
    )

    assert training.evaluate_rel_l2(trained, ts, ys) < 0.8 * before


class TestPipeline:
    @pytest.fixture
    def recorder(self, monkeypatch):
        """Replaces the training stages and evaluations with fakes that record what ran, and on
        which generator."""

        calls = []

        def train_embedding(cfg, network, generator, key):
            calls.append(('train embedding', generator))
            return network

        def train_operator(cfg, embedding, operator, generator, key):
            calls.append(('train operator', generator))
            return operator

        def train_jointly(cfg, embedding, operator, generator, key):
            calls.append(('train jointly', generator))
            return embedding, operator

        def evaluate_embedding(cfg, embedding, generator, key):
            calls.append(('evaluate embedding', generator))

        def evaluate_operator(cfg, embedding, operator, generator, key):
            calls.append(('evaluate operator', generator))

        monkeypatch.setattr(training, 'train_function_embeddings', train_embedding)
        monkeypatch.setattr(training, 'train_operator_network', train_operator)
        monkeypatch.setattr(training, 'train_operator_jointly', train_jointly)

        return calls, evaluate_embedding, evaluate_operator

    @pytest.fixture
    def generators(self):
        # Distinct objects, so the calls show which stage got which generator.
        return (
            LogisticGenerator((0.0, 5.0), (0.1, 2.0), (0.5, 3.0), (1.0, 5.0)),
            LogisticGenerator((0.0, 5.0), (0.1, 2.0), (0.5, 3.0), (1.0, 5.0)),
        )

    def test_trains_and_evaluates_both_stages_in_order(self, recorder, generators, make_config):
        calls, *evaluations = recorder
        generator, operator_generator = generators

        embedding, operator = run_training_pipeline(
            make_config(), generator, operator_generator, *evaluations
        )

        assert isinstance(embedding, FunctionEmbedding)
        assert isinstance(operator, OperatorMapping)
        assert calls == [
            ('train embedding', generator),
            ('evaluate embedding', generator),
            ('train operator', operator_generator),
            ('evaluate operator', operator_generator),
        ]

    def test_can_skip_the_operator(self, recorder, generators, make_config):
        calls, *evaluations = recorder
        cfg = make_config()
        cfg.training.train_operator = False

        _, operator = run_training_pipeline(cfg, *generators, *evaluations)

        assert operator is None
        assert [name for name, _ in calls] == ['train embedding', 'evaluate embedding']

    def test_can_train_the_operator_jointly_with_the_embedding(
        self, recorder, generators, make_config
    ):
        calls, *evaluations = recorder
        generator, operator_generator = generators
        cfg = make_config()
        cfg.training.train_operator_jointly = True

        run_training_pipeline(cfg, generator, operator_generator, *evaluations)

        assert calls == [
            ('train jointly', operator_generator),
            ('evaluate embedding', generator),
            ('evaluate operator', operator_generator),
        ]

    def test_reuses_a_pretrained_embedding(self, recorder, generators, make_config):
        calls, *evaluations = recorder
        cfg = make_config()
        pretrained = create_embedding_network(cfg, jax.random.key(7))
        cfg.training.embedding_ckpt_path = save_ckpt(pretrained, cfg, step=1)

        embedding, _ = run_training_pipeline(cfg, *generators, *evaluations)

        assert not changed(pretrained, embedding)
        assert [name for name, _ in calls] == ['train operator', 'evaluate operator']


@pytest.mark.slow
def test_family_trains_end_to_end(family, make_config, tmp_path, monkeypatch, key):
    """Each family's entry point trains, evaluates, plots and checkpoints a tiny model."""

    monkeypatch.chdir(tmp_path)  # the experiment tracker writes its offline files here
    cfg = make_config(family)

    family.main(cfg)

    run_dir = Path(cfg.model_dir) / cfg.run_name
    assert [p.name for p in run_dir.iterdir()] == [f'{cfg.training.num_iterations}.eqx']

    # The checkpoint rebuilds from its own saved config.
    def build(saved: dict):
        saved_cfg = build_dataclass(family.config_cls, saved)
        return (
            build_from_config(saved, key),
            create_operator_network(saved_cfg, family.generator(), key),
        )

    embedding, operator = load_ckpt(str(run_dir / f'{cfg.training.num_iterations}.eqx'), build)
    assert is_finite(params((embedding, operator)))

    figures = {p.name for p in (Path(cfg.figure_dir) / cfg.run_name).iterdir()}
    assert set(family.figures) <= figures
