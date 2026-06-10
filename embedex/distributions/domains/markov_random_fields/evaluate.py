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


def model_log_pmf(
    model: FlowEmbedding,
    embedding: jax.Array,
    states: jax.Array,
    num_evals: int,
    key: jax.Array,
) -> jax.Array:
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
        distribution.num_states,
        distribution.coupling_scale,
        distribution.bias_scale,
        distribution.edge_prob,
        mrf_key,
        graph_type=distribution.graph_type,
        edge_prob_min=distribution.edge_prob_min,
        edge_prob_max=distribution.edge_prob_max,
    )
    data = sample_batch(mrfs, config.num_embedding_samples, sample_key)
    states = all_states(distribution.num_spins, distribution.num_states)

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
