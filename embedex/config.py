import argparse
import ast
import copy
import os
import sys
import types
from dataclasses import fields
from typing import Any

import yaml


def load_config[T](cfg_cls: type[T], save_path: str) -> T:
    """Load a YAML file into an instance of `cfg_cls`.

    A relative `save_path` attribute, if present, is resolved against the config file's directory.
    """

    with open(save_path) as f:
        data = yaml.safe_load(f) or {}

    config = build_dataclass(cfg_cls, data)

    if hasattr(config, 'save_path') and not os.path.isabs(config.save_path):
        config.save_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(save_path)), config.save_path)
        )

    return config


def build_dataclass[T](cls: type[T], data: dict[str, Any]) -> T:
    """Construct `cls` from `data`, recursing into nested dataclass fields.

    Unknown keys are ignored; scalar values are coerced to the field's annotated type where possible.
    """

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
        elif (
            isinstance(value, dict)
            and isinstance(anno, type)
            and hasattr(anno, '__dataclass_fields__')
        ):
            kwargs[key] = build_dataclass(anno, value)
        elif anno in (int, float, str, bool) and not isinstance(value, anno):
            kwargs[key] = anno(value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def apply_overrides[T](config: T, overrides: list[str]) -> T:
    """Return a deep copy of `config` with `key.path=value` strings applied.

    Each value is parsed to match the type of the field it overrides.
    """

    config = copy.deepcopy(config)
    for override in overrides:
        if '=' not in override:
            print(f'Warning: ignoring malformed override: {override}', file=sys.stderr)
            continue

        key_path, value_str = override.split('=', 1)
        parts = key_path.split('.')

        obj = config
        for part in parts[:-1]:
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
