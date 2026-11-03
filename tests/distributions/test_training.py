"""Training distribution embeddings through their generators."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from embedex.checkpointing import load_ckpt, read_ckpt_config
from embedex.config import build_dataclass
from embedex.distributions.builder import build_from_config, create_model
from embedex.distributions.config import TrainingConfig
from embedex.distributions.training import init_optimizer, train


def params(model) -> list[jax.Array]:
    return jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))


def changed(before, after) -> bool:
    return not all(
        jnp.array_equal(a, b) for a, b in zip(params(before), params(after), strict=True)
    )


def is_finite(model) -> bool:
    return all(bool(jnp.all(jnp.isfinite(p))) for p in params(model))


def sampler(domain, cfg):
    return lambda batch_size, num_samples, key: domain.sample(
        cfg.data, batch_size, num_samples, key
    )


class TestTrain:
    def test_training_updates_the_model(self, domain, make_config, key):
        cfg = make_config(domain)
        init_key, train_key = jax.random.split(key)
        model = create_model(cfg.model, dequantize=False, key=init_key)

        trained, losses, norms = train(model, cfg.training, sampler(domain, cfg), key=train_key)

        assert changed(model, trained)
        assert is_finite(trained)
        assert len(losses) == len(norms) == cfg.training.num_iterations
        assert np.all(np.isfinite(losses)) and np.all(np.asarray(norms) > 0)

    @pytest.mark.parametrize('num_iterations, logged', [(5, 3), (6, 4)])
    def test_logs_every_interval_and_the_last_iteration(
        self, num_iterations, logged, domains, make_config, key
    ):
        cfg = make_config(domains['donuts'])
        cfg.training.num_iterations, cfg.training.log_interval = num_iterations, 2
        model = create_model(cfg.model, dequantize=False, key=key)

        _, losses, norms = train(model, cfg.training, sampler(domains['donuts'], cfg), key=key)

        assert len(losses) == len(norms) == logged

    @pytest.mark.parametrize('interval, steps', [(2, [2, 4]), (4, [4]), (6, []), (0, [])])
    def test_checkpoints_every_interval_but_the_last(
        self, interval, steps, domains, make_config, key
    ):
        cfg = make_config(domains['donuts'])
        cfg.training.num_iterations, cfg.training.checkpoint_interval = 6, interval
        model = create_model(cfg.model, dequantize=False, key=key)
        saved = []

        train(
            model,
            cfg.training,
            sampler(domains['donuts'], cfg),
            checkpoint_fn=lambda step, m: saved.append(step),
            key=key,
        )

        assert saved == steps

    def test_batches_may_come_with_weights(self, domains, make_config, key):
        cfg = make_config(domains['donuts'])
        model = create_model(cfg.model, dequantize=False, key=key)
        sample = sampler(domains['donuts'], cfg)

        def weighted(batch_size, num_samples, key):
            return sample(batch_size, num_samples, key), jnp.ones((batch_size, num_samples))

        trained, _, _ = train(model, cfg.training, weighted, key=key)
        unweighted, _, _ = train(model, cfg.training, sample, key=key)

        for a, b in zip(params(trained), params(unweighted), strict=True):
            np.testing.assert_array_equal(a, b)

    def test_training_lowers_the_held_out_loss(self, domains, make_model, make_config, key):
        # A sign error in the update would still change the parameters, so check the direction.
        donuts = domains['donuts']
        cfg = make_config(donuts)
        cfg.training.num_iterations, cfg.training.learning_rate = 100, 3e-3
        init_key, train_key, eval_key = jax.random.split(key, 3)
        model = make_model(flow_type='nf', key=init_key)
        held_out = donuts.sample(cfg.data, 32, 64, eval_key)

        def loss(m):
            return float(
                jnp.mean(jax.vmap(lambda xs: m.loss(xs, shuffle=False, key=key)[0])(held_out))
            )

        trained, _, _ = train(model, cfg.training, sampler(donuts, cfg), key=train_key)

        assert loss(trained) < loss(model) - 0.5


def update_sizes(config: TrainingConfig, grads: list[float]) -> list[float]:
    """The size of each Adam update of one parameter given these gradients. Adam's first steps
    move a parameter by about the learning rate whatever the gradient's size."""

    optimizer, state = init_optimizer({'w': jnp.zeros(1)}, config)
    sizes = []
    for g in grads:
        updates, state = optimizer.update({'w': jnp.array([g])}, state)
        sizes.append(float(jnp.abs(updates['w'][0])))

    return sizes


class TestOptimizer:
    def test_without_warmup_the_learning_rate_is_constant(self):
        config = TrainingConfig(learning_rate=1e-2, num_iterations=10)

        np.testing.assert_allclose(update_sizes(config, [1.0] * 10), 1e-2, rtol=1e-4)

    def test_warmup_rises_to_the_peak_then_decays_to_the_end_rate(self):
        config = TrainingConfig(
            learning_rate=1e-2, end_learning_rate=1e-4, warmup_steps=4, num_iterations=12
        )

        sizes = update_sizes(config, [1.0] * 13)

        assert sizes[0] == pytest.approx(1e-4, rel=1e-3)
        assert sizes[4] == pytest.approx(1e-2, rel=1e-3)
        assert sizes[12] == pytest.approx(1e-4, rel=1e-3)
        assert all(a < b for a, b in zip(sizes[:4], sizes[1:5], strict=True))
        assert all(a > b for a, b in zip(sizes[4:12], sizes[5:13], strict=True))

    def test_clipping_tempers_a_gradient_spike(self):
        # Adam ignores a constant rescaling of the gradients, but not a single spike among them.
        clipped = TrainingConfig(learning_rate=1e-2, clip_grad_norm=1.0)
        unclipped = TrainingConfig(learning_rate=1e-2)

        assert update_sizes(clipped, [1.0, 100.0])[1] == pytest.approx(1e-2, rel=1e-3)
        assert update_sizes(unclipped, [1.0, 100.0])[1] < 0.8e-2

    def test_optimizer_state_covers_only_the_parameters(self, make_model):
        model = make_model()

        optimizer, state = init_optimizer(model, TrainingConfig())

        adam_state = state[0]
        assert isinstance(adam_state, optax.ScaleByAdamState)
        assert len(jax.tree.leaves(adam_state.mu)) == len(params(model))


@pytest.mark.slow
def test_domain_trains_and_evaluates_end_to_end(domain, make_config, tmp_path, monkeypatch, key):
    """Each domain's entry point trains and checkpoints a tiny model, then evaluates and plots it
    from the checkpoint, as the paper's runs do."""

    monkeypatch.chdir(tmp_path)
    cfg = make_config(domain)

    domain.main(cfg)

    ckpt = Path(cfg.model_dir) / cfg.run_name / f'{cfg.training.num_iterations}.eqx'
    assert [p.name for p in ckpt.parent.iterdir()] == [ckpt.name]
    model = load_ckpt(str(ckpt), lambda saved: build_from_config(saved, dequantize=False, key=key))
    assert is_finite(model)

    evaluation = make_config(domain)
    evaluation.model_ckpt = str(ckpt)
    domain.main(evaluation)

    figures = {p.name for p in (Path(cfg.figure_dir) / cfg.run_name).iterdir()}
    assert {'loss_curve.pdf', 'embedding_norm_curve.pdf', *domain.figures} == figures

    # The saved config matches the domain's config class (an extra key would raise here).
    build_dataclass(domain.config_cls, read_ckpt_config(str(ckpt)))
