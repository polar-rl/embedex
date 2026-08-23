import json
import os
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import equinox as eqx


def save_ckpt(model: eqx.Module, cfg: Any, step: int) -> str:
    ckpt_dir = os.path.join(cfg.model_dir, cfg.run_name)
    num_steps = cfg.training.num_iterations
    ckpt_path = os.path.join(ckpt_dir, f'{step:0{len(str(num_steps))}}.eqx')
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(ckpt_path, 'wb') as f:
        f.write((json.dumps(asdict(cfg)) + '\n').encode())
        eqx.tree_serialise_leaves(f, model)

    return ckpt_path


def load_ckpt(ckpt_path: str, build_model: Callable[[dict], eqx.Module]) -> eqx.Module:
    with open(ckpt_path, 'rb') as f:
        cfg = json.loads(f.readline().decode())

        return eqx.tree_deserialise_leaves(f, build_model(cfg))


def read_ckpt_config(ckpt_path: str) -> dict:
    with open(ckpt_path, 'rb') as f:
        return json.loads(f.readline().decode())


def select_last_ckpt(directory: str) -> str:
    files = [f for f in os.listdir(directory) if f.endswith('.eqx')]

    # Numeric sort: checkpoint names are zero-padded to the run's iteration
    # width, so resumed runs mix widths ('60000.eqx' vs '120000.eqx') and a
    # string sort would pick the shorter-width earlier checkpoint as "last".
    def _key(f):
        stem = f.rsplit('.', 1)[0]
        return (0, int(stem), '') if stem.isdigit() else (1, 0, f)

    return os.path.join(directory, sorted(files, key=_key)[-1])
