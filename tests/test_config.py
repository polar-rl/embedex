"""Building configs from YAML files, saved configs and command-line overrides."""

import sys
import warnings
from dataclasses import asdict, dataclass, field

import pytest

from embedex.config import apply_overrides, build_dataclass, load_config, parse_args


@dataclass
class Optimizer:
    learning_rate: float = 1e-3
    schedule: str = 'cosine'


@dataclass
class Experiment:
    optimizer: Optimizer = field(default_factory=Optimizer)
    seed: int = 0
    checkpoint: str | None = None

    @property
    def tag(self) -> str:
        return f'{self.optimizer.schedule}-{self.seed}'


@pytest.fixture
def write_yaml(tmp_path):
    def _write(text: str) -> str:
        path = tmp_path / 'config.yaml'
        path.write_text(text)
        return str(path)

    return _write


class TestConfigFiles:
    def test_values_fill_nested_fields_and_take_the_fields_types(self, write_yaml):
        cfg = load_config(Experiment, write_yaml('optimizer:\n  learning_rate: 1\nseed: "3"\n'))

        assert cfg == Experiment(Optimizer(learning_rate=1.0), seed=3)
        assert isinstance(cfg.optimizer.learning_rate, float)

    def test_an_empty_file_gives_the_defaults(self, write_yaml):
        assert load_config(Experiment, write_yaml('')) == Experiment()

    def test_unknown_keys_are_rejected_and_all_named(self, write_yaml):
        path = write_yaml('optimizer:\n  lr: 0.1\n  schedule: constant\nseeds: 3\n')

        with pytest.raises(ValueError, match='Experiment has no fields optimizer.lr, seeds'):
            load_config(Experiment, path)


class TestSavedConfigs:
    def test_unknown_keys_are_skipped_with_a_warning(self):
        # Checkpoint headers keep fields that later code renamed or removed.
        saved = {'optimizer': {'momentum': 0.9, 'schedule': 'constant'}, 'seed': 1, 'eval_only': 1}

        with pytest.warns(UserWarning, match='has no fields optimizer.momentum, eval_only'):
            cfg = build_dataclass(Experiment, saved)

        assert cfg == Experiment(Optimizer(schedule='constant'), seed=1)

    def test_a_current_config_round_trips_without_warnings(self):
        cfg = Experiment(Optimizer(0.5, 'constant'), seed=2, checkpoint='run/10.eqx')

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert build_dataclass(Experiment, asdict(cfg)) == cfg


class TestOverrides:
    def test_values_are_parsed_to_the_fields_types(self):
        cfg = Experiment()

        overridden = apply_overrides(cfg, ['optimizer.learning_rate=0.5', 'seed=7'])

        assert overridden == Experiment(Optimizer(learning_rate=0.5), seed=7)
        assert cfg == Experiment()  # the original is left untouched

    @pytest.mark.parametrize(
        'override',
        ['sed=1', 'optimizer.lr=0.1', 'optimiser.learning_rate=0.1', 'seed.value=1', 'tag=x'],
    )
    def test_unknown_keys_are_rejected(self, override):
        with pytest.raises(ValueError, match='Experiment has no field'):
            apply_overrides(Experiment(), [override])

    def test_malformed_overrides_are_skipped_with_a_warning(self):
        with pytest.warns(UserWarning, match="malformed override 'seed'"):
            assert apply_overrides(Experiment(), ['seed']) == Experiment()


def test_the_command_line_applies_overrides_on_top_of_the_file(write_yaml, monkeypatch):
    path = write_yaml('seed: 3\noptimizer:\n  schedule: constant\n')
    monkeypatch.setattr(sys, 'argv', ['main.py', '--config', path, '--set', 'seed=5'])

    assert parse_args(Experiment) == Experiment(Optimizer(schedule='constant'), seed=5)
