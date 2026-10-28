from functools import partial

import jax
import jax.numpy as jnp
import minfx.neptune_v2 as minfx

from embedex.checkpointing import save_ckpt
from embedex.graphs.evaluate import evaluate_embeddings, evaluate_reconstruction
from embedex.graphs.families.config import ArmsConfig, parse_args
from embedex.graphs.generator import Generator
from embedex.graphs.training import run_training_pipeline
from embedex.graphs.visualize import plot_reconstructions
from embedex.logger import log_cfg


class ArmsGenerator(Generator):
    def __init__(self, num_vertices: int) -> None:
        self.num_vertices = num_vertices
        self._arm_lengths = jnp.asarray(
            [al for al in range(1, num_vertices - 1) if (num_vertices - 1) % al == 0]
        )

    def _sample_one(self, num_samples: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        idx_key, perm_key, edge_key = jax.random.split(key, 3)
        arm_length = self._arm_lengths[jax.random.randint(idx_key, (), 0, len(self._arm_lengths))]

        p = jnp.arange(self.num_vertices - 1)
        j = p + 1
        depth = p % arm_length

        adj = jnp.zeros((self.num_vertices, self.num_vertices))
        spoke = (depth == 0).astype(jnp.float32)
        adj = adj.at[0, j].add(spoke).at[j, 0].add(spoke)

        interior = (depth < arm_length - 1).astype(jnp.float32)
        nxt = jnp.minimum(j + 1, self.num_vertices - 1)
        adj = adj.at[j, nxt].add(interior).at[nxt, j].add(interior)

        perm = jax.random.permutation(perm_key, self.num_vertices)
        adj = adj[perm][:, perm]

        return self._sample_edges_from_adj(adj, num_samples, edge_key), adj


def main(cfg: ArmsConfig) -> None:
    cfg.model.input_size = cfg.data.num_vertices

    cfg.run = minfx.init_run(
        project=cfg.project,
        name=cfg.run_name,
        tags=['graph_embeddings', 'arms'],
        mode='async' if not cfg.debug_mode else 'debug',
    )
    log_cfg(cfg.run, cfg)

    generator = ArmsGenerator(cfg.data.num_vertices)
    embedding_network, medians_network = run_training_pipeline(
        cfg,
        generator,
        evaluate_embeddings,
        partial(evaluate_reconstruction, plot_fn=plot_reconstructions),
    )

    save_ckpt((embedding_network, medians_network), cfg, cfg.training.num_iterations)
    cfg.run.stop()


if __name__ == '__main__':
    main(parse_args(ArmsConfig))
