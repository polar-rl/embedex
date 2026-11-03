"""Training graph embeddings and their reconstruction heads."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.checkpointing import load_ckpt, save_ckpt
from embedex.config import build_dataclass
from embedex.graphs import training
from embedex.graphs.builder import (
    build_from_config,
    create_embedding_network,
    create_medians_network,
    create_transition_network,
)
from embedex.graphs.embedding import GraphEmbedding
from embedex.graphs.families.random_trees import RandomTreesGenerator
from embedex.graphs.training import (
    compute_training_diagnostics,
    run_chunked_training,
    run_training_pipeline,
    train_graph_embeddings,
    train_medians_network,
    train_transition_network,
)


def is_finite(tree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(tree))


def changed(before, after) -> bool:
    """Whether training moved any parameter. Training donates its input network and key, so
    `before` must be a separately built copy, not the network that was trained."""

    params = (eqx.filter(m, eqx.is_inexact_array) for m in (before, after))
    return not eqx.tree_equal(*params)


class TestChunkedTraining:
    @staticmethod
    def count_steps(steps: jax.Array, total: jax.Array, key: jax.Array):
        """A training step that counts how often it ran; its 'loss' is the step index."""

        return (steps + 1, total + steps, key), steps

    @pytest.mark.parametrize(
        'num_iterations, chunk_size, chunks',
        [(10, 4, [4, 4, 2]), (8, 4, [4, 4]), (3, 8, [3])],
    )
    def test_runs_every_iteration_in_chunks(self, num_iterations, chunk_size, chunks, key):
        seen = []
        state = (jnp.asarray(0), jnp.asarray(0), key)

        steps, total, _ = run_chunked_training(
            state,
            num_iterations,
            chunk_size,
            self.count_steps,
            lambda losses, _: seen.append(losses),
        )

        assert int(steps) == num_iterations
        assert int(total) == sum(range(num_iterations))
        assert [len(losses) for losses in seen] == chunks
        np.testing.assert_array_equal(np.concatenate(seen), np.arange(num_iterations))


class TestTrainingStages:
    def test_embedding_training_updates_the_network(self, family, make_config, key):
        cfg = make_config(family)
        init_key, train_key = jax.random.split(key)

        trained = train_graph_embeddings(
            cfg, create_embedding_network(cfg, init_key), family.generator(), train_key
        )

        assert isinstance(trained, GraphEmbedding)
        assert changed(create_embedding_network(cfg, init_key), trained)
        assert is_finite(eqx.filter(trained, eqx.is_inexact_array))

    def test_embedding_training_logs_losses_and_diagnostics(self, make_config, fake_run, key):
        cfg = make_config()
        cfg.run = fake_run
        generator = RandomTreesGenerator(cfg.model.input_size)

        train_graph_embeddings(cfg, create_embedding_network(cfg, key), generator, key)

        assert len(fake_run.series['train/embedding/loss']) == cfg.training.num_iterations
        num_chunks = cfg.training.num_iterations // cfg.training.chunk_size
        for name in ('eig_vals_sum', 'true_eig_vals_sum', 'ortho_violation', 'betas_max'):
            assert len(fake_run.series[f'progress/{name}']) == num_chunks

    @pytest.mark.parametrize(
        'create, train',
        [
            (create_medians_network, train_medians_network),
            (create_transition_network, train_transition_network),
        ],
        ids=['medians', 'transition'],
    )
    def test_head_training_updates_only_the_head(self, create, train, family, make_config, key):
        cfg = make_config(family)
        init_key, train_key = jax.random.split(key)
        embedding = create_embedding_network(cfg, init_key)

        trained = train(cfg, embedding, create(cfg, init_key), family.generator(), train_key)

        assert type(trained) is type(create(cfg, init_key))
        assert changed(create(cfg, init_key), trained)
        assert eqx.tree_equal(embedding, create_embedding_network(cfg, init_key))  # stays frozen
        assert is_finite(eqx.filter(trained, eqx.is_inexact_array))

    def test_diagnostics_are_finite(self, family, make_config, key):
        cfg = make_config(family)
        edges, adjs = family.generator().sample(4, 32, key)

        diagnostics = compute_training_diagnostics(create_embedding_network(cfg, key), edges, adjs)

        assert len(diagnostics) == 4
        assert is_finite(diagnostics)


class TestPipeline:
    @pytest.fixture
    def recorder(self):
        """Fake pipeline stages that record the order they ran in."""

        calls = []

        def embedding_eval(cfg, embedding, generator, key):
            calls.append('evaluate embedding')

        def create_head(cfg, key):
            calls.append('create head')
            return 'head'

        def train_head(cfg, embedding, head, generator, key):
            calls.append(f'train {head}')
            return 'trained head'

        def head_eval(cfg, embedding, head, generator, key):
            calls.append(f'evaluate {head}')

        return calls, (embedding_eval, create_head, train_head, head_eval)

    def test_trains_and_evaluates_both_stages_in_order(self, recorder, make_config, monkeypatch):
        calls, stages = recorder
        cfg = make_config()

        def train_embedding(cfg, network, generator, key):
            calls.append('train embedding')
            return network

        monkeypatch.setattr(training, 'train_graph_embeddings', train_embedding)
        embedding, head = run_training_pipeline(cfg, RandomTreesGenerator(8), *stages)

        assert isinstance(embedding, GraphEmbedding)
        assert head == 'trained head'
        assert calls == [
            'train embedding',
            'evaluate embedding',
            'create head',
            'train head',
            'evaluate trained head',
        ]

    def test_reuses_a_pretrained_embedding(self, recorder, make_config, monkeypatch, key):
        calls, stages = recorder
        cfg = make_config()
        pretrained = create_embedding_network(cfg, jax.random.key(7))
        cfg.training.embedding_ckpt_path = save_ckpt(pretrained, cfg, step=1)

        def train_embedding(*args):
            raise AssertionError('the embedding should have been loaded, not trained')

        monkeypatch.setattr(training, 'train_graph_embeddings', train_embedding)
        embedding, _ = run_training_pipeline(cfg, RandomTreesGenerator(8), *stages)

        assert eqx.tree_equal(embedding, pretrained)
        assert calls == ['create head', 'train head', 'evaluate trained head']


@pytest.mark.slow
def test_family_trains_end_to_end(family, make_config, tmp_path, monkeypatch, key):
    """Each family's entry point trains, evaluates, plots and checkpoints a tiny model."""

    monkeypatch.chdir(tmp_path)  # the experiment tracker writes its offline files here
    cfg = make_config(family)

    family.main(cfg)

    run_dir = tmp_path / 'models' / cfg.run_name
    assert [p.name for p in run_dir.iterdir()] == [f'{cfg.training.num_iterations}.eqx']

    # The checkpoint rebuilds from its own saved config.
    embedding, head = load_ckpt(
        str(run_dir / f'{cfg.training.num_iterations}.eqx'),
        lambda saved: (
            build_from_config(saved, key),
            family.create_reconstructor(build_dataclass(family.config_cls, saved), key),
        ),
    )
    assert isinstance(embedding, GraphEmbedding)
    assert is_finite(eqx.filter((embedding, head), eqx.is_inexact_array))

    figures = {p.name for p in (tmp_path / 'figures' / cfg.run_name).iterdir()}
    assert {'ground_truth_geometry.pdf', 'geometry.pdf', *family.figures} <= figures
