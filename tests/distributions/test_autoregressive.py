"""KV-cached AR sampling must be exactly equivalent to naive full-prefix decoding."""

import jax
import jax.numpy as jnp
import pytest

from embedex.distributions.builder import create_generative_network
from embedex.distributions.config import GeneratorConfig, ModelConfig


def _build_ar(num_variables: int, conditioning_dim: int, relational: bool, key):
    gen = GeneratorConfig(
        flow_type='ar',
        ar_num_classes=2,
        ar_token_dim=16,
        ar_num_heads=2,
        ar_num_layers=3,
        ar_ff_multiplier=2,
    )
    cfg = ModelConfig(input_dim=num_variables, generator=gen)
    cfg.embedding.relational = relational
    return create_generative_network(cfg, conditioning_dim, False, key)


def _naive_sample(model, sideinfo, key):
    keys = jax.random.split(key, model.num_variables)

    def body(i, x_built):
        logits = model.net(x_built, sideinfo)
        xi = jax.random.categorical(keys[i], logits[i])
        return x_built.at[i].set(xi.astype(x_built.dtype))

    x0 = jnp.zeros(model.num_variables, dtype=jnp.int32)
    return jax.lax.fori_loop(0, model.num_variables, body, x0).astype(jnp.float32)


@pytest.mark.parametrize('relational', [False, True])
def test_cached_sample_matches_naive(relational):
    num_variables = 7
    build_key, key = jax.random.split(jax.random.key(0))
    moment_dim = 2 * num_variables
    cond_dim = (moment_dim + num_variables * num_variables) if relational else 24
    model = _build_ar(num_variables, cond_dim, relational, build_key)

    for trial in range(20):
        key, side_key, sample_key = jax.random.split(key, 3)
        sideinfo = jax.random.normal(side_key, (cond_dim,))
        cached = model.sample(sideinfo, None, sample_key)
        naive = _naive_sample(model, sideinfo, sample_key)
        assert jnp.array_equal(cached, naive), f'trial {trial}: {cached} != {naive}'


@pytest.mark.parametrize('relational', [False, True])
def test_cached_logits_match_teacher_forced(relational):
    num_variables = 7
    build_key, key = jax.random.split(jax.random.key(1))
    moment_dim = 2 * num_variables
    cond_dim = (moment_dim + num_variables * num_variables) if relational else 24
    model = _build_ar(num_variables, cond_dim, relational, build_key)

    key, side_key, sample_key = jax.random.split(key, 3)
    sideinfo = jax.random.normal(side_key, (cond_dim,))
    x = model.sample(sideinfo, None, sample_key).astype(jnp.int32)

    logits = model.net(x, sideinfo)
    keys = jax.random.split(sample_key, num_variables)
    redrawn = jnp.array([jax.random.categorical(keys[i], logits[i]) for i in range(num_variables)])
    assert jnp.array_equal(x, redrawn)

    lp, _, _ = model.log_prob(x.astype(jnp.float32), sideinfo)
    assert jnp.isfinite(lp)
