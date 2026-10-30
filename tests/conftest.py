"""Fixtures shared by every test suite."""

from collections import defaultdict
from typing import Any

import jax
import pytest


@pytest.fixture
def key() -> jax.Array:
    return jax.random.key(0)


class FakeRun:
    """Stands in for an experiment-tracker run and records everything logged to it."""

    def __init__(self) -> None:
        self.series: dict[str, list] = defaultdict(list)
        self.fields: dict[str, Any] = {}

    def __getitem__(self, key: str) -> '_Handle':
        return _Handle(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.fields[key] = value


class _Handle:
    def __init__(self, run: FakeRun, key: str) -> None:
        self.run, self.key = run, key

    def append(self, value: Any) -> None:
        self.run.series[self.key].append(value)

    def __setitem__(self, key: str, value: Any) -> None:
        self.run.fields[f'{self.key}/{key}'] = value


@pytest.fixture
def fake_run() -> FakeRun:
    return FakeRun()
