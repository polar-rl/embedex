from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from embedex.distributions.config import TrainingConfig
from embedex.distributions.embedding import DistributionEmbedding

# Type for the data generation callback:
#   (batch_size, num_samples, key) -> either:
#     - Array of shape (batch_size, num_samples, input_dim), or
#     - tuple (xs, weights) with xs as above and weights of shape (batch_size, num_samples).
DataGenerator = Callable[[int, int, jax.Array], jax.Array | tuple[jax.Array, jax.Array]]


def train(
    model: DistributionEmbedding,
    config: TrainingConfig,
    generate_batch: DataGenerator,
    *,
    checkpoint_fn: Callable[[int, DistributionEmbedding], None] | None = None,
    key: jax.Array,
) -> tuple[DistributionEmbedding, list[float], list[float]]:
    """Generic training loop.

    Args:
        model: The DistributionEmbedding model to train.
        config: Training hyperparameters.
        generate_batch: Callback that generates a batch of data.
            Signature: (batch_size, num_samples, key) -> Array (batch_size, num_samples, input_dim).
        checkpoint_fn: Optional callback invoked as `checkpoint_fn(step, model)` every
            `config.checkpoint_interval` steps (except at the final iteration).
        key: PRNG key.

    Returns:
        Tuple of (trained_model, losses, norms).
    """

    optimizer, opt_state = init_optimizer(model, config)

    losses: list[float] = []
    norms: list[float] = []

    progress_bar = tqdm(range(config.num_iterations))
    for i in progress_bar:
        key, data_key, step_key = jax.random.split(key, 3)

        batch = generate_batch(config.batch_size, config.num_samples, data_key)
        if isinstance(batch, tuple):
            x, weights = batch
        else:
            x = batch
            weights = jnp.ones(x.shape[:2])

        model, opt_state, loss_value, mean_norm = train_step(
            model, optimizer, opt_state, x, weights, key=step_key
        )

        if i % config.log_interval == 0 or i == config.num_iterations - 1:
            losses.append(float(loss_value))
            norms.append(float(mean_norm))
            progress_bar.set_postfix({'loss': f'{float(loss_value):.4f}'})

        if (
            checkpoint_fn is not None
            and config.checkpoint_interval > 0
            and (i + 1) % config.checkpoint_interval == 0
            and i + 1 < config.num_iterations
        ):
            checkpoint_fn(i + 1, model)

    return model, losses, norms


def init_optimizer(
    model: DistributionEmbedding,
    config: TrainingConfig,
) -> tuple[optax.GradientTransformation, optax.OptState]:
    if config.warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.end_learning_rate,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.num_iterations,
            end_value=config.end_learning_rate,
        )
        optimizer = optax.adam(schedule)
    else:
        optimizer = optax.adam(config.learning_rate)

    if config.clip_grad_norm is not None and config.clip_grad_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(config.clip_grad_norm)),
            optimizer,
        )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    return optimizer, opt_state


@eqx.filter_jit
def train_step(
    model: DistributionEmbedding,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    x: jax.Array,
    weights: jax.Array,
    key: jax.Array,
) -> tuple[DistributionEmbedding, optax.OptState, jax.Array, jax.Array]:
    @eqx.filter_value_and_grad(has_aux=True)
    def compute_loss(model, x, weights, keys):
        results = jax.vmap(model.loss)(x, weights, key=keys)
        return jnp.mean(results[0]), jnp.mean(results[3])

    keys = jax.random.split(key, x.shape[0])
    (loss_value, mean_norm), grads = compute_loss(model, x, weights, keys)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss_value, mean_norm
