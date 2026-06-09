from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from tqdm import tqdm

from embedex.checkpointing import load_ckpt
from embedex.functions.builder import (
    build_from_config,
    create_embedding_network,
    create_operator_network,
    create_optimizer,
)
from embedex.functions.config import Config
from embedex.functions.embedding import FunctionEmbedding
from embedex.functions.generator import Generator
from embedex.functions.metrics import (
    evaluate_mse,
    evaluate_operator_mse,
    evaluate_operator_rel_l2,
    evaluate_rel_l2,
)
from embedex.functions.operator import OperatorMapping
from embedex.logger import log_param_count, safe_log


def run_training_pipeline(
    cfg: Config,
    generator: Generator,
    embedding_eval_fn: Callable,
    operator_eval_fn: Callable,
) -> tuple[FunctionEmbedding, OperatorMapping | None]:
    key = jax.random.key(cfg.seed)

    if cfg.training.train_operator_jointly:
        key, embedding_key, operator_key, train_key, eval_key = jax.random.split(key, 5)
        embedding_network = create_embedding_network(cfg, embedding_key)
        operator_network = create_operator_network(cfg, generator, operator_key)
        log_param_count(cfg, 'embedding', embedding_network)
        log_param_count(cfg, 'operator', operator_network)

        embedding_network, operator_network = train_operator_jointly(
            cfg,
            embedding_network,
            operator_network,
            generator,
            train_key,
        )

        embedding_eval_fn(cfg, embedding_network, generator, eval_key)
        operator_eval_fn(cfg, embedding_network, operator_network, generator, eval_key)

        return embedding_network, operator_network

    key, network_key = jax.random.split(key, 2)
    embedding_network = create_embedding_network(cfg, network_key)
    log_param_count(cfg, 'embedding', embedding_network)

    ckpt_path = cfg.training.embedding_ckpt_path
    if ckpt_path:
        embedding_network = load_ckpt(ckpt_path, lambda c: build_from_config(c, network_key))
        print(f'Loaded embedding from {ckpt_path}; skipping embedding training.')
    else:
        key, train_key, eval_key = jax.random.split(key, 3)
        embedding_network = train_function_embeddings(cfg, embedding_network, generator, train_key)
        embedding_eval_fn(cfg, embedding_network, generator, eval_key)

    if not cfg.training.train_operator:
        return embedding_network, None

    key, network_key, train_key, eval_key = jax.random.split(key, 4)
    operator_network = create_operator_network(cfg, generator, network_key)
    log_param_count(cfg, 'operator', operator_network)

    operator_network = train_operator_network(
        cfg,
        embedding_network,
        operator_network,
        generator,
        train_key,
    )
    operator_eval_fn(cfg, embedding_network, operator_network, generator, eval_key)

    return embedding_network, operator_network


def train_function_embeddings(
    cfg: Config,
    embedding_network: FunctionEmbedding,
    generator: Generator,
    key: jax.Array,
) -> FunctionEmbedding:
    @eqx.filter_jit(donate='all')
    def train_step(embedding_network, opt, opt_state, xs, ys, key):
        @eqx.filter_value_and_grad
        def compute_loss(embedding_network, xs, ys, keys):
            return jnp.mean(jax.vmap(embedding_network.loss)(xs, ys, keys))

        keys = jax.random.split(key, xs.shape[0])
        loss_value, grads = compute_loss(embedding_network, xs, ys, keys)
        updates, opt_state = opt.update(grads, opt_state, embedding_network)
        embedding_network = eqx.apply_updates(embedding_network, updates)

        return embedding_network, opt_state, loss_value

    opt = create_optimizer(cfg, cfg.training.weight_decay, cfg.training.num_iterations)
    opt_state = opt.init(eqx.filter(embedding_network, eqx.is_inexact_array))

    eval_every = cfg.training.eval_every

    for step in tqdm(range(cfg.training.num_iterations), desc='Training function embeddings'):
        key, data_key, train_key = jax.random.split(key, 3)
        xs, ys, *_ = generator.sample(cfg.training.batch_size, cfg.training.num_samples, data_key)

        embedding_network, opt_state, loss_value = train_step(
            embedding_network, opt, opt_state, xs, ys, train_key
        )

        if hasattr(cfg, 'run'):
            safe_log(cfg.run, 'train/embedding/loss', loss_value)

        if eval_every > 0 and (step + 1) % eval_every == 0:
            key, eval_key = jax.random.split(key, 2)
            eval_xs, eval_ys, *_ = generator.sample(
                cfg.training.batch_size, cfg.training.num_samples, eval_key
            )
            mse = evaluate_mse(embedding_network, eval_xs, eval_ys)
            rel_l2 = evaluate_rel_l2(embedding_network, eval_xs, eval_ys)

            if hasattr(cfg, 'run'):
                safe_log(cfg.run, 'progress/embedding/mse', mse)
                safe_log(cfg.run, 'progress/embedding/rel_l2', rel_l2)

    return embedding_network


def train_operator_network(
    cfg: Config,
    embedding_network: FunctionEmbedding,
    operator_network: OperatorMapping,
    generator: Generator,
    key: jax.Array,
) -> OperatorMapping:
    @eqx.filter_jit(donate='all')
    def train_step(operator_network, opt, opt_state, xs, ys, extra):
        @eqx.filter_value_and_grad
        def compute_loss(operator_network, xs, ys, extra):
            def _single_loss(x, y, e):
                source = generator.operator_input(embedding_network, x, y, e)
                pred_embedding = operator_network(source)
                y_pred = embedding_network.eval(x, pred_embedding)
                y_target = generator.operator_ground_truth(x, y, e)

                return embedding_network.loss_fn(y_pred, y_target)

            return jnp.mean(jax.vmap(_single_loss)(xs, ys, extra))

        loss_value, grads = compute_loss(operator_network, xs, ys, extra)
        updates, opt_state = opt.update(grads, opt_state, operator_network)
        operator_network = eqx.apply_updates(operator_network, updates)

        return operator_network, opt_state, loss_value

    num_steps = cfg.training.operator_num_iterations
    opt = create_optimizer(cfg, cfg.training.operator_weight_decay, num_steps)
    opt_state = opt.init(eqx.filter(operator_network, eqx.is_inexact_array))

    eval_every = cfg.training.eval_every

    for step in tqdm(range(num_steps), desc='Training operator mapping'):
        key, data_key = jax.random.split(key, 2)
        xs, ys, extra = generator.sample(
            cfg.training.batch_size, cfg.training.num_samples, data_key
        )

        operator_network, opt_state, loss_value = train_step(
            operator_network, opt, opt_state, xs, ys, extra
        )

        if hasattr(cfg, 'run'):
            safe_log(cfg.run, 'train/operator/loss', loss_value)

        if eval_every > 0 and (step + 1) % eval_every == 0:
            key, eval_key = jax.random.split(key, 2)
            eval_xs, eval_ys, eval_extra = generator.sample(
                cfg.training.batch_size, cfg.training.num_samples, eval_key
            )
            mse = evaluate_operator_mse(
                embedding_network, operator_network, generator, eval_xs, eval_ys, eval_extra
            )
            rel_l2 = evaluate_operator_rel_l2(
                embedding_network, operator_network, generator, eval_xs, eval_ys, eval_extra
            )

            if hasattr(cfg, 'run'):
                safe_log(cfg.run, 'progress/operator/mse', mse)
                safe_log(cfg.run, 'progress/operator/rel_l2', rel_l2)

    return operator_network


def train_operator_jointly(
    cfg: Config,
    embedding_network: FunctionEmbedding,
    operator_network: OperatorMapping,
    generator: Generator,
    key: jax.Array,
) -> tuple[FunctionEmbedding, OperatorMapping]:
    @eqx.filter_jit(donate='all')
    def train_step(networks, opt_state, xs, ys, extra):
        @eqx.filter_value_and_grad
        def compute_loss(networks, xs, ys, extra):
            embedding, operator = networks

            def _single_loss(x, y, e):
                source = generator.operator_input(embedding, x, y, e)
                pred_embedding = operator(source)
                y_pred = embedding.eval(x, pred_embedding)
                y_target = generator.operator_ground_truth(x, y, e)
                return embedding.loss_fn(y_pred, y_target)

            return jnp.mean(jax.vmap(_single_loss)(xs, ys, extra))

        loss_value, grads = compute_loss(networks, xs, ys, extra)
        updates, opt_state = opt.update(grads, opt_state, networks)
        networks = eqx.apply_updates(networks, updates)

        return networks, opt_state, loss_value

    num_steps = cfg.training.operator_num_iterations
    opt = create_optimizer(cfg, cfg.training.operator_weight_decay, num_steps)

    networks = (embedding_network, operator_network)
    opt_state = opt.init(eqx.filter(networks, eqx.is_inexact_array))

    eval_every = cfg.training.eval_every

    for step in tqdm(range(num_steps), desc='Training operator (joint)'):
        key, data_key = jax.random.split(key, 2)
        xs, ys, extra = generator.sample(
            cfg.training.batch_size, cfg.training.num_samples, data_key
        )

        networks, opt_state, loss_value = train_step(networks, opt_state, xs, ys, extra)

        if hasattr(cfg, 'run'):
            safe_log(cfg.run, 'train/operator/loss', loss_value)

        if eval_every > 0 and (step + 1) % eval_every == 0:
            embedding, operator = networks

            key, eval_key = jax.random.split(key, 2)
            eval_xs, eval_ys, eval_extra = generator.sample(
                cfg.training.batch_size, cfg.training.num_samples, eval_key
            )
            mse = evaluate_operator_mse(
                embedding, operator, generator, eval_xs, eval_ys, eval_extra
            )
            rel_l2 = evaluate_operator_rel_l2(
                embedding, operator, generator, eval_xs, eval_ys, eval_extra
            )
            if hasattr(cfg, 'run'):
                safe_log(cfg.run, 'progress/operator/mse', mse)
                safe_log(cfg.run, 'progress/operator/rel_l2', rel_l2)

    return networks
