from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from embedex.checkpointing import load_ckpt
from embedex.graphs.builder import (
    build_from_config,
    create_embedding_network,
    create_medians_network,
    create_optimizer,
)
from embedex.graphs.config import Config
from embedex.graphs.embedding import GraphEmbedding
from embedex.graphs.generator import Generator
from embedex.graphs.metrics import ortho_violation
from embedex.graphs.reconstructor import MediansNetwork, medians_loss
from embedex.logger import log_param_count, safe_log


def run_training_pipeline(
    cfg: Config,
    generator: Generator,
    embedding_eval_fn: Callable,
    medians_eval_fn: Callable,
) -> tuple[GraphEmbedding, MediansNetwork]:
    key = jax.random.key(cfg.seed)

    key, network_key = jax.random.split(key, 2)
    embedding_network = create_embedding_network(cfg, network_key)
    log_param_count(cfg, 'embedding', embedding_network)

    ckpt_path = cfg.training.embedding_ckpt_path
    if ckpt_path:
        embedding_network = load_ckpt(ckpt_path, lambda c: build_from_config(c, network_key))
        print(f'Loaded embedding from {ckpt_path}; skipping embedding training.')
    else:
        key, train_key, eval_key = jax.random.split(key, 3)
        embedding_network = train_graph_embeddings(cfg, embedding_network, generator, train_key)
        embedding_eval_fn(cfg, embedding_network, generator, key=eval_key)

    key, network_key, train_key, eval_key = jax.random.split(key, 4)
    medians_network = create_medians_network(cfg, network_key)
    log_param_count(cfg, 'medians', medians_network)

    medians_network = train_medians_network(
        cfg, embedding_network, medians_network, generator, train_key
    )
    medians_eval_fn(cfg, embedding_network, medians_network, generator, key=eval_key)

    return embedding_network, medians_network


def train_graph_embeddings(
    cfg: Config,
    embedding_network: GraphEmbedding,
    generator: Generator,
    key: jax.Array,
) -> GraphEmbedding:
    @eqx.filter_jit(donate='all')
    def train_step(embedding_network, opt, opt_state, edges, key):
        @eqx.filter_value_and_grad
        def compute_loss(embedding_network, edges, keys):
            return jnp.mean(jax.vmap(embedding_network.loss)(edges, keys))

        keys = jax.random.split(key, edges.shape[0])
        loss_value, grads = compute_loss(embedding_network, edges, keys)

        # Sign-flip on betas weights (saddle-point ascent on dual variables)
        grads = eqx.tree_at(lambda m: m.betas, grads, jax.tree.map(lambda x: -x, grads.betas))

        updates, opt_state = opt.update(grads, opt_state, embedding_network)
        embedding_network = eqx.apply_updates(embedding_network, updates)

        return embedding_network, opt_state, loss_value

    opt = create_optimizer(cfg, cfg.training.num_iterations)
    opt_state = opt.init(eqx.filter(embedding_network, eqx.is_inexact_array))

    eval_every = cfg.training.eval_every

    for step in tqdm(range(cfg.training.num_iterations), desc='Training graph embeddings'):
        key, data_key, train_key = jax.random.split(key, 3)
        edges, _ = generator.sample(cfg.training.batch_size, cfg.training.num_samples, data_key)

        embedding_network, opt_state, loss_value = train_step(
            embedding_network, opt, opt_state, edges, train_key
        )

        if hasattr(cfg, 'run'):
            safe_log(cfg.run, 'train/embedding/loss', loss_value)

        if eval_every > 0 and (step + 1) % eval_every == 0:
            key, eval_key = jax.random.split(key, 2)
            eval_edges, eval_adjs = generator.sample(
                cfg.training.batch_size, cfg.training.num_samples, eval_key
            )
            eig_vals_sum, true_eig_vals_sum, ortho_viol, betas_max = compute_training_diagnostics(
                embedding_network, eval_edges, eval_adjs
            )

            if hasattr(cfg, 'run'):
                safe_log(cfg.run, 'progress/eig_vals_sum', eig_vals_sum)
                safe_log(cfg.run, 'progress/true_eig_vals_sum', true_eig_vals_sum)
                safe_log(cfg.run, 'progress/ortho_violation', ortho_viol)
                safe_log(cfg.run, 'progress/betas_max', betas_max)

    return embedding_network


def train_medians_network(
    cfg: Config,
    embedding_network: GraphEmbedding,
    medians_network: MediansNetwork,
    generator: Generator,
    key: jax.Array,
) -> MediansNetwork:
    @eqx.filter_jit(donate='all')
    def train_step(medians_network, opt, opt_state, edges, key):
        @eqx.filter_value_and_grad
        def compute_loss(medians_network, edges, key):
            return medians_loss(medians_network, embedding_network, edges, key)

        loss_value, grads = compute_loss(medians_network, edges, key)
        updates, opt_state = opt.update(grads, opt_state, medians_network)
        medians_network = eqx.apply_updates(medians_network, updates)

        return medians_network, opt_state, loss_value

    opt = optax.adam(cfg.training.medians_learning_rate)
    opt_state = opt.init(eqx.filter(medians_network, eqx.is_inexact_array))

    for _ in tqdm(range(cfg.training.medians_num_iterations), desc='Training medians network'):
        key, data_key, loss_key = jax.random.split(key, 3)
        edges, _ = generator.sample(cfg.training.batch_size, cfg.training.num_samples, data_key)

        medians_network, opt_state, loss_value = train_step(
            medians_network, opt, opt_state, edges, loss_key
        )

        if hasattr(cfg, 'run'):
            safe_log(cfg.run, 'train/medians/loss', loss_value)

    return medians_network


@eqx.filter_jit
def compute_training_diagnostics(
    embedding_network: GraphEmbedding, edges: jax.Array, adjs: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    def _per_graph(edges, adj):
        vertices = jnp.eye(adj.shape[-1])
        z = embedding_network.embed(edges)

        eig_vals = embedding_network.eigenvalues(z)
        eig_vals_sum = jnp.sum(eig_vals)

        laplacian = jnp.diag(jnp.sum(adj, axis=0)) - adj
        true_eig_vals = jax.numpy.linalg.eigh(laplacian)[0]
        true_eig_vals_sum = jnp.sum(true_eig_vals[: eig_vals.shape[0]])

        uvs = jax.vmap(lambda v: embedding_network.eigenfunctions(v, z))(vertices)
        ortho_viol = ortho_violation(uvs, jnp.sum(adj, axis=1))

        betas_max = jnp.max(jnp.abs(embedding_network.betas(z)))

        return eig_vals_sum, true_eig_vals_sum, ortho_viol, betas_max

    diagnostics = jax.vmap(_per_graph)(edges, adjs)
    diagnostics = jax.tree.map(lambda x: jnp.mean(x), diagnostics)

    return diagnostics
