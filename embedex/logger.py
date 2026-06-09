import math
from dataclasses import asdict
from typing import Any

import equinox as eqx
import jax
import numpy as np


def log_cfg(run: Any, cfg: Any) -> None:
    config_dict = asdict(cfg)

    for section in config_dict.values():
        if not isinstance(section, dict):
            continue

        for key, value in section.items():
            if value is None:
                continue

            run['cfg'][key] = str(value) if isinstance(value, list | tuple) else value


def safe_log(run: Any, key: str, value: Any) -> None:
    if hasattr(value, 'item'):
        value = value.item()

    if value is None or not np.isfinite(value):
        print(f'Warning: Not logging {key} due to non-finite value: {value}!')
        return None

    if value == 0.0 and isinstance(value, float) and math.copysign(1.0, value) < 0:
        value = 0.0

    run[key].append(value)


def count_params(module: eqx.Module) -> int:
    leaves = jax.tree.leaves(eqx.filter(module, eqx.is_inexact_array))
    return sum(leaf.size for leaf in leaves)


def log_param_count(cfg: Any, name: str, module: eqx.Module) -> None:
    num_params = count_params(module)
    print(f'{name} parameters: {num_params:,}')

    if hasattr(cfg, 'run'):
        cfg.run[f'cfg/{name}_num_params'] = num_params
