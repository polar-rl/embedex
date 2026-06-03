from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from embedex.distributions.domains.markov_random_fields.config import (
    EvaluationConfig,
    MRFDistributionConfig,
)
from embedex.distributions.domains.markov_random_fields.mrf import (
    MRF,
    all_states,
    random_mrfs,
    sample_batch,
)
from embedex.distributions.embedding import FlowEmbedding
from embedex.distributions.metrics import js_divergence


class EvaluationMetrics(NamedTuple):
    mean_jsd: jax.Array
    std_jsd: jax.Array


def empirical_pmf(samples: jax.Array, num_spins: int) -> jax.Array:
    """Empirical PMF over all 2**N states from generated binary samples.

    Used for quick visualization labels; the headline metric scores the model's
    log_prob directly (see `model_log_pmf`) to avoid finite-sample noise.
    """

    powers = 2 ** jnp.arange(num_spins)
    bits = jnp.clip(jnp.round(samples), 0, 1).astype(jnp.int32)
    indices = bits @ powers
    counts = jnp.zeros(2**num_spins).at[indices].add(1.0)
    total = counts.sum()
    return jnp.where(total > 0, counts / total, jnp.ones_like(counts) / counts.shape[0])


def model_log_pmf(
    model: FlowEmbedding,
    embedding: jax.Array,
    states: jax.Array,
    num_evals: int,
    key: jax.Array,
) -> jax.Array:
    """Normalized model log-PMF over all enumerated states.

    `FlowEmbedding.log_prob` handles the variance reduction: with `reps > 1` it
    averages that many stochastic passes in probability space. Here we just
    renormalize the result across the enumerated support so it's a proper PMF,
    with no dependence on generated-sample count.
    """

    log_probs, _, _ = model.log_prob(states, embedding, key, reps=num_evals)
    return log_probs - jax.nn.logsumexp(log_probs)


@eqx.filter_jit
def compute_metrics(
    model: FlowEmbedding,
    mrfs: MRF,
    data: jax.Array,
    states: jax.Array,
    num_embedding_samples: int,
    num_evals: int,
    key: jax.Array,
) -> EvaluationMetrics:
    true_pmfs = jax.vmap(lambda m: m.pmf())(mrfs)

    keys = jax.random.split(key, data.shape[0])

    def model_pmf(data_i: jax.Array, eval_key: jax.Array) -> jax.Array:
        embed_samples = data_i[:num_embedding_samples]
        embedding, _ = model.embed(embed_samples, jnp.ones(embed_samples.shape[0]))
        return jnp.exp(model_log_pmf(model, embedding, states, num_evals, eval_key))

    model_pmfs = jax.vmap(model_pmf)(data, keys)
    jsds = jax.vmap(js_divergence)(true_pmfs, model_pmfs)

    return EvaluationMetrics(mean_jsd=jnp.mean(jsds), std_jsd=jnp.std(jsds))


def evaluate(
    model: FlowEmbedding,
    config: EvaluationConfig,
    distribution: MRFDistributionConfig,
    key: jax.Array,
) -> EvaluationMetrics:
    key, mrf_key, sample_key, eval_key = jax.random.split(key, 4)
    mrfs = random_mrfs(
        config.num_distributions,
        distribution.num_spins,
        distribution.coupling_scale,
        distribution.bias_scale,
        mrf_key,
    )
    data = sample_batch(mrfs, config.num_embedding_samples, sample_key)
    states = all_states(distribution.num_spins)

    metrics = compute_metrics(
        model,
        mrfs,
        data,
        states,
        config.num_embedding_samples,
        config.num_logprob_evals,
        eval_key,
    )

    print(f'JSD: {metrics.mean_jsd:.5f} +/- {metrics.std_jsd:.5f}')

    return metrics
