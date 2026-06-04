from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from embedex.distributions.domains.donuts.config import DonutDistributionConfig, EvaluationConfig
from embedex.distributions.domains.donuts.donut import random_donuts, sample_batch
from embedex.distributions.embedding import DistributionEmbedding, FlowEmbedding
from embedex.distributions.metrics import sliced_wasserstein_distance


class EvaluationMetrics(NamedTuple):
    mean_error: jax.Array
    radius_error: jax.Array
    width_error: jax.Array
    width_radius_ratio_error: jax.Array
    swd: jax.Array
    nll: jax.Array = jnp.array(jnp.nan)


@eqx.filter_jit
def rmse(x: jax.Array, y: jax.Array, axis: int = -1) -> jax.Array:
    return jnp.mean(jnp.sqrt(jnp.sum((x - y) ** 2, axis=axis)))


@eqx.filter_jit
def compute_nll(
    model: FlowEmbedding,
    data: jax.Array,
    num_embedding_samples: int,
    num_logprob_samples: int,
) -> jax.Array:
    """Mean held-out NLL via the exact log-density path (`key=None`).

    Exact for both generators in 2D: closed-form change of variables for the
    NF, integrated exact Jacobian trace for FM. The stochastic (Hutchinson)
    path must not be used here: its noise enters the density through an
    exponent, so E[e^X] > e^(E[X]) inflates densities and reports NLL below
    the entropy floor of the family.
    """

    def per_dist(samples: jax.Array) -> jax.Array:
        embed_samples = samples[:num_embedding_samples]
        holdout = samples[-num_logprob_samples:]
        z, _ = model.embed(embed_samples, jnp.ones(embed_samples.shape[0]))
        log_probs, _, _ = model.log_prob(holdout, z, key=None)
        return -jnp.mean(log_probs)

    return jnp.mean(jax.vmap(per_dist)(data))


@eqx.filter_jit
def generate_from_samples(
    model: DistributionEmbedding,
    data: jax.Array,
    num_embedding_samples: int,
    num_generated_samples: int,
    key: jax.Array,
) -> jax.Array:
    keys = jax.random.split(key, data.shape[0])

    def generate(data, key):
        embed_samples = data[:num_embedding_samples]
        embedding, _ = model.embed(embed_samples, jnp.ones(embed_samples.shape[0]))
        return model.generate(embedding, num_generated_samples, key=key)

    return jax.vmap(generate)(data, keys)


@eqx.filter_jit
def compute_metrics(
    data: jax.Array,
    samples: jax.Array,
    key: jax.Array,
) -> EvaluationMetrics:
    data_mean = jnp.mean(data, keepdims=True, axis=1)
    samples_mean = jnp.mean(samples, keepdims=True, axis=1)
    mean_error = rmse(data_mean, samples_mean)

    data_radius = jnp.linalg.norm(data - data_mean, axis=2)
    samples_radius = jnp.linalg.norm(samples - samples_mean, axis=2)

    data_radius_mean = jnp.mean(data_radius, keepdims=True, axis=1)
    samples_radius_mean = jnp.mean(samples_radius, keepdims=True, axis=1)
    radius_error = rmse(data_radius_mean, samples_radius_mean)

    data_radius_diff = data_radius - data_radius_mean
    samples_radius_diff = samples_radius - samples_radius_mean

    data_radius_std = jnp.std(data_radius_diff, keepdims=True, axis=1)
    samples_radius_std = jnp.std(samples_radius_diff, keepdims=True, axis=1)
    width_error = rmse(data_radius_std, samples_radius_std)

    width_radius_ratio_error = rmse(
        data_radius_std / (data_radius_mean + 1e-8),
        samples_radius_std / (samples_radius_mean + 1e-8),
    )

    num_dist, num_gen = samples.shape[:2]
    swd_keys = jax.random.split(key, num_dist)
    swd = jnp.mean(
        jax.vmap(sliced_wasserstein_distance, in_axes=(0, 0, None, 0))(
            data[:, :num_gen], samples, 1000, swd_keys
        )
    )

    return EvaluationMetrics(
        mean_error=mean_error,
        radius_error=radius_error,
        width_error=width_error,
        width_radius_ratio_error=width_radius_ratio_error,
        swd=swd,
    )


def evaluate(
    model: DistributionEmbedding,
    config: EvaluationConfig,
    distribution: DonutDistributionConfig,
    key: jax.Array,
) -> EvaluationMetrics:
    total_samples = (
        config.num_embedding_samples + config.num_generated_samples + config.num_logprob_samples
    )

    key, donut_key, sample_key = jax.random.split(key, 3)
    donuts = random_donuts(
        config.num_distributions,
        distribution.mean_std,
        distribution.radius_range,
        distribution.width_range,
        donut_key,
    )
    data = sample_batch(donuts, total_samples, sample_key)

    key, gen_key = jax.random.split(key, 2)
    samples = generate_from_samples(
        model,
        data,
        config.num_embedding_samples,
        config.num_generated_samples,
        gen_key,
    )

    key, swd_key = jax.random.split(key, 2)
    metrics = compute_metrics(data, samples, swd_key)
    nll = compute_nll(model, data, config.num_embedding_samples, config.num_logprob_samples)
    metrics = metrics._replace(nll=nll)

    print(f'Mean Error: {metrics.mean_error:.5f}')
    print(f'Radius Error: {metrics.radius_error:.5f}')
    print(f'Width Error: {metrics.width_error:.5f}')
    print(f'Width / Radius Error: {metrics.width_radius_ratio_error:.5f}')
    print(f'SWD: {metrics.swd:.5f}')
    print(f'NLL: {metrics.nll:.5f}')

    return metrics


def evaluate_sample_efficiency(
    model: DistributionEmbedding,
    config: EvaluationConfig,
    distribution: DonutDistributionConfig,
    key: jax.Array,
) -> dict[int, EvaluationMetrics]:
    sample_counts = config.sample_efficiency_counts
    total_samples = max(sample_counts) + config.num_generated_samples + config.num_logprob_samples

    key, donut_key, sample_key = jax.random.split(key, 3)
    donuts = random_donuts(
        config.num_distributions,
        distribution.mean_std,
        distribution.radius_range,
        distribution.width_range,
        donut_key,
    )
    data = sample_batch(donuts, total_samples, sample_key)

    print(f'{"n":>6}  {"Mean":>8}  {"Radius":>8}  {"Width":>8}  {"W/R":>8}  {"SWD":>8}  {"NLL":>8}')
    print(f'{"-" * 6}  {"-" * 8}  {"-" * 8}  {"-" * 8}  {"-" * 8}  {"-" * 8}  {"-" * 8}')

    results = {}
    for n in sample_counts:
        key, gen_key = jax.random.split(key, 2)
        samples = generate_from_samples(
            model,
            data,
            n,
            config.num_generated_samples,
            gen_key,
        )

        key, swd_key = jax.random.split(key, 2)
        metrics = compute_metrics(data, samples, swd_key)
        nll = compute_nll(model, data, n, config.num_logprob_samples)
        metrics = metrics._replace(nll=nll)

        results[n] = metrics
        print(
            f'{n:>6d}  {metrics.mean_error:>8.4f}  {metrics.radius_error:>8.4f}  '
            f'{metrics.width_error:>8.4f}  {metrics.width_radius_ratio_error:>8.4f}  '
            f'{metrics.swd:>8.4f}  {metrics.nll:>8.4f}'
        )

    return results
