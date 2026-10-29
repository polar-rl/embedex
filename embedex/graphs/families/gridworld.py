from functools import partial

import jax
import jax.numpy as jnp
import minfx.neptune_v2 as minfx

from embedex.checkpointing import save_ckpt
from embedex.graphs.builder import create_transition_network
from embedex.graphs.evaluate import evaluate_embeddings, evaluate_transitions
from embedex.graphs.families.config import GridWorldConfig, parse_args
from embedex.graphs.generator import Generator
from embedex.graphs.training import run_training_pipeline, train_transition_network
from embedex.graphs.visualize import plot_weighted_reconstructions
from embedex.logger import log_cfg


class GridWorldGenerator(Generator):
    def __init__(self, grid_size: int, num_walls: int) -> None:
        self.num_vertices = grid_size * grid_size
        self.num_walls = num_walls

        idx = jnp.arange(self.num_vertices)
        r, c = idx // grid_size, idx % grid_size
        manhattan = jnp.abs(r[:, jnp.newaxis] - r[jnp.newaxis, :]) + jnp.abs(
            c[:, jnp.newaxis] - c[jnp.newaxis, :]
        )
        self.grid_mask = (manhattan == 1).astype(jnp.float32)

        rows, cols = jnp.triu_indices(self.num_vertices, k=1)
        is_edge = self.grid_mask[rows, cols] > 0
        self.edges = jnp.stack([rows[is_edge], cols[is_edge]], axis=1)

    def _connected(self, present: jax.Array) -> jax.Array:
        n = self.num_vertices
        reach = (present > 0) | jnp.eye(n, dtype=jnp.bool_)

        for _ in range((n - 1).bit_length()):
            reach = (reach.astype(jnp.float32) @ reach.astype(jnp.float32)) > 0

        return jnp.all(reach)

    def _wall_mask(self, key: jax.Array) -> jax.Array:
        n = self.num_vertices

        if self.num_walls == 0:
            return jnp.ones((n, n))

        count_key, cand_key = jax.random.split(key, 2)
        count = jax.random.randint(count_key, (), 0, self.num_walls + 1)
        active = jnp.arange(self.num_walls) < count

        def _candidate(k):
            walls = self.edges[jax.random.choice(k, self.edges.shape[0], (self.num_walls,), False)]
            set_vals = jnp.where(active, 0.0, 1.0)
            keep = jnp.ones((n, n)).at[walls[:, 0], walls[:, 1]].set(set_vals)
            keep = keep.at[walls[:, 1], walls[:, 0]].set(set_vals)

            return keep, self._connected(self.grid_mask * keep)

        keeps, connected = jax.vmap(_candidate)(jax.random.split(cand_key, 8))
        first = jnp.argmax(connected)

        return jnp.where(connected[first], keeps[first], jnp.ones((n, n)))

    def _sample_one(self, num_samples: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        weight_key, wall_key, edge_key = jax.random.split(key, 3)

        n = self.num_vertices
        w = jnp.triu(jax.random.uniform(weight_key, (n, n)), k=1)
        adj = (w + jnp.transpose(w)) * self.grid_mask * self._wall_mask(wall_key)

        return self._sample_edges_from_adj(adj, num_samples, edge_key), adj


def main(cfg: GridWorldConfig) -> None:
    cfg.model.input_size = cfg.data.grid_size**2

    cfg.run = minfx.init_run(
        project=cfg.project,
        name=cfg.run_name,
        tags=['graph_embeddings', 'gridworld'],
        mode='async' if not cfg.debug_mode else 'debug',
    )
    log_cfg(cfg.run, cfg)

    generator = GridWorldGenerator(cfg.data.grid_size, cfg.data.num_walls)
    embedding_network, transition_network = run_training_pipeline(
        cfg,
        generator,
        evaluate_embeddings,
        create_transition_network,
        train_transition_network,
        partial(evaluate_transitions, plot_fn=plot_weighted_reconstructions),
    )

    save_ckpt((embedding_network, transition_network), cfg, cfg.training.num_iterations)
    cfg.run.stop()


if __name__ == '__main__':
    main(parse_args(GridWorldConfig))
