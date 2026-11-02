import argparse
import ast
import copy
import types
import warnings
from dataclasses import fields, is_dataclass
from typing import Any

import yaml


def load_config[T](cfg_cls: type[T], path: str) -> T:
    """Load a YAML file into an instance of `cfg_cls`, rejecting keys it has no field for."""

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if unknown := _unknown_keys(cfg_cls, data):
        raise ValueError(f'{path}: {cfg_cls.__name__} has no fields {", ".join(unknown)}')

    return _build_dataclass(cfg_cls, data)


def build_dataclass[T](cls: type[T], data: dict[str, Any]) -> T:
    """Construct `cls` from a saved config, recursing into nested dataclass fields.

    Saved configs, such as checkpoint headers, keep fields that newer code may have renamed or
    removed, so keys that `cls` has no field for are skipped with a warning rather than rejected.
    Scalar values are coerced to the field's annotated type where possible.
    """

    if unknown := _unknown_keys(cls, data):
        warnings.warn(
            f'{cls.__name__} has no fields {", ".join(unknown)}; ignoring them', stacklevel=2
        )

    return _build_dataclass(cls, data)


def _unknown_keys(cls: type, data: dict[str, Any]) -> list[str]:
    """Report the keys in `data` that are not fields of `cls`, as dotted paths.

    Nested dictionaries are checked against the dataclass of their field, so a misspelled
    `{'training': {'lr': 0.1}}` is reported as `training.lr`.
    """

    annotations = {f.name: f.type for f in fields(cls)}
    unknown = []

    for key, value in data.items():
        if key not in annotations:
            unknown.append(key)
        elif isinstance(value, dict) and _is_dataclass_type(annotations[key]):
            unknown += [f'{key}.{path}' for path in _unknown_keys(annotations[key], value)]

    return unknown


def _build_dataclass[T](cls: type[T], data: dict[str, Any]) -> T:
    def _type_allows_none(anno):
        """Check if a type annotation includes None (e.g. `str | None`)."""

        return isinstance(anno, types.UnionType) and type(None) in anno.__args__

    valid = {f.name for f in fields(cls)}
    kwargs = {}

    for key, value in data.items():
        if key not in valid:
            continue

        field = next(f for f in fields(cls) if f.name == key)
        anno = field.type

        if (value is None or value == 'None') and _type_allows_none(anno):
            kwargs[key] = None
        elif isinstance(value, dict) and _is_dataclass_type(anno):
            kwargs[key] = _build_dataclass(anno, value)
        elif anno in (int, float, str, bool) and not isinstance(value, anno):
            kwargs[key] = anno(value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def _is_dataclass_type(anno: Any) -> bool:
    return isinstance(anno, type) and hasattr(anno, '__dataclass_fields__')


def apply_overrides[T](config: T, overrides: list[str]) -> T:
    """Return a deep copy of `config` with `key.path=value` strings applied.

    Each value is parsed to match the type of the field it overrides.
    """

    config = copy.deepcopy(config)
    for override in overrides:
        if '=' not in override:
            warnings.warn(
                f'Ignoring malformed override {override!r}; expected key.path=value', stacklevel=2
            )
            continue

        key_path, value_str = override.split('=', 1)
        parts = key_path.split('.')

        obj = config
        for i, part in enumerate(parts):
            if not is_dataclass(obj) or part not in {f.name for f in fields(obj)}:
                raise ValueError(f'{type(config).__name__} has no field {".".join(parts[: i + 1])}')
            if i < len(parts) - 1:
                obj = getattr(obj, part)

        field_name = parts[-1]
        current = getattr(obj, field_name)

        if isinstance(current, bool):
            parsed = value_str.lower() in ('true', '1', 'yes')
        elif isinstance(current, int):
            parsed = int(value_str)
        elif isinstance(current, float):
            parsed = float(value_str)
        elif isinstance(current, list):
            try:
                parsed = ast.literal_eval(value_str)
                if not isinstance(parsed, list):
                    parsed = [parsed]
            except (ValueError, SyntaxError):
                parsed = [value_str]
        elif current is None:
            parsed = value_str if value_str.lower() != 'null' else None
        else:
            parsed = value_str

        setattr(obj, field_name, parsed)

    return config


def parse_args[T](cfg_cls: type[T]) -> T:
    """Parse `--config` / `--set` CLI arguments into a `cfg_cls` instance."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to a YAML config file')
    parser.add_argument('--set', nargs='*', default=[], help='Overrides in the form key.path=value')
    args = parser.parse_args()

    config = load_config(cfg_cls, args.config) if args.config is not None else cfg_cls()

    if args.set:
        config = apply_overrides(config, args.set)

    return config
