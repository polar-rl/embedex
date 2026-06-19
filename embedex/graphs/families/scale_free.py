import jax
import jax.numpy as jnp
import minfx.neptune_v2 as minfx

from embedex.checkpointing import save_ckpt
from embedex.graphs.evaluate import evaluate_embeddings, evaluate_reconstruction
from embedex.graphs.families.config import ScaleFreeConfig, parse_args
from embedex.graphs.generator import Generator
from embedex.graphs.training import run_training_pipeline
from embedex.logger import log_cfg


class ScaleFreeGenerator(Generator):
    def __init__(self, num_vertices: int, m: int) -> None:
        assert 1 <= m < num_vertices

        self.num_vertices = num_vertices
        self.m = m

    def _sample_one(self, num_samples: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Generate a single Barabási–Albert scale-free graph.

        Vertices are added in index order; vertex `t` attaches to `m` earlier vertices
        drawn by preferential attachment — probability proportional to `degree + 1` so
        the first edges are well-defined when all degrees are zero.
        """

        attach_key, perm_key, edge_key = jax.random.split(key, 3)

        n = self.num_vertices
        idx = jnp.arange(n)

        def _attach(adj, inputs):
            t, node_key = inputs

            weights = jnp.where(idx < t, jnp.sum(adj, axis=1) + 1.0, 0.0)  # only earlier vertices
            targets = jax.random.choice(
                node_key, n, (self.m,), replace=True, p=weights / jnp.sum(weights)
            )
            adj = adj.at[t, targets].set(1.0).at[targets, t].set(1.0)

            return adj, None

        node_keys = jax.random.split(attach_key, n - 1)
        adj, _ = jax.lax.scan(_attach, jnp.zeros((n, n)), (jnp.arange(1, n), node_keys))

        perm = jax.random.permutation(perm_key, n)
        adj = adj[perm][:, perm]

        return self._sample_edges_from_adj(adj, num_samples, edge_key), adj


def main(cfg: ScaleFreeConfig) -> None:
    cfg.model.input_size = cfg.data.num_vertices

    cfg.run = minfx.init_run(
        project=cfg.project,
        name=cfg.run_name,
        tags=['graph_embeddings', 'scale_free'],
        mode='async' if not cfg.debug_mode else 'debug',
    )
    log_cfg(cfg.run, cfg)

    generator = ScaleFreeGenerator(cfg.data.num_vertices, cfg.data.m)
    embedding_network, _ = run_training_pipeline(
        cfg,
        generator,
        evaluate_embeddings,
        evaluate_reconstruction,
    )

    save_ckpt(embedding_network, cfg, step=cfg.training.num_iterations)
    cfg.run.stop()


if __name__ == '__main__':
    main(parse_args(ScaleFreeConfig))
