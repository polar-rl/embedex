from functools import partial

import jax
import jax.numpy as jnp
import minfx.neptune_v2 as minfx

from embedex.checkpointing import save_ckpt
from embedex.graphs.evaluate import evaluate_embeddings, evaluate_reconstruction
from embedex.graphs.families.config import RandomTreesConfig, parse_args
from embedex.graphs.generator import Generator
from embedex.graphs.training import run_training_pipeline
from embedex.graphs.visualize import plot_reconstructions
from embedex.logger import log_cfg


class RandomTreesGenerator(Generator):
    def __init__(self, num_vertices: int) -> None:
        self.num_vertices = num_vertices

    def _sample_one(self, num_samples: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        adj_key, edge_key = jax.random.split(key, 2)

        n = self.num_vertices
        prufer = jax.random.randint(adj_key, (n - 2,), 0, n)
        degrees = jnp.ones(n, jnp.int32).at[prufer].add(1)
        idx = jnp.arange(n)

        def _attach(carry, v):
            degrees, adj = carry

            leaf = jnp.argmin(jnp.where(degrees == 1, idx, n))
            degrees = degrees.at[leaf].add(-1).at[v].add(-1)
            adj = adj.at[leaf, v].set(1.0).at[v, leaf].set(1.0)

            return (degrees, adj), None

        (degrees, adj), _ = jax.lax.scan(_attach, (degrees, jnp.zeros((n, n))), prufer)

        # The two vertices left with degree 1 form the final edge.
        u = jnp.argmin(jnp.where(degrees == 1, idx, n))
        w = jnp.argmin(jnp.where(degrees.at[u].add(-1) == 1, idx, n))
        adj = adj.at[u, w].set(1.0).at[w, u].set(1.0)

        return self._sample_edges_from_adj(adj, num_samples, edge_key), adj


def main(cfg: RandomTreesConfig) -> None:
    cfg.model.input_size = cfg.data.num_vertices

    cfg.run = minfx.init_run(
        project=cfg.project,
        name=cfg.run_name,
        tags=['graph_embeddings', 'random_tree'],
        mode='async' if not cfg.debug_mode else 'debug',
    )
    log_cfg(cfg.run, cfg)

    generator = RandomTreesGenerator(cfg.data.num_vertices)
    embedding_network, medians_network = run_training_pipeline(
        cfg,
        generator,
        evaluate_embeddings,
        partial(evaluate_reconstruction, plot_fn=plot_reconstructions),
    )

    save_ckpt((embedding_network, medians_network), cfg, cfg.training.num_iterations)
    cfg.run.stop()


if __name__ == '__main__':
    main(parse_args(RandomTreesConfig))
