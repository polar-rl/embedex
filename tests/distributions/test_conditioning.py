"""The ways a generator's networks read the embedding: concatenation, FiLM, a hypernetwork, and
cross-attention to conditioning tokens."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.distributions.conditioning import (
    ConcatMLP,
    CrossAttentionBlock,
    FiLMMLP,
    HyperMLP,
    batch_layer_norm,
    batch_linear,
)

IN, COND, OUT, WIDTH, DEPTH = 5, 3, 4, 8, 2


@pytest.fixture
def x(key) -> jax.Array:
    return jax.random.normal(key, (IN,))


@pytest.fixture
def sideinfo(key) -> jax.Array:
    return jax.random.normal(jax.random.fold_in(key, 1), (COND,))


def test_concatenation_appends_the_embedding_to_the_input(x, sideinfo, key):
    net = ConcatMLP(IN + COND, OUT, WIDTH, DEPTH, key=key)

    np.testing.assert_allclose(net(x, sideinfo), net.mlp(jnp.concatenate([x, sideinfo])))


def test_film_without_modulation_is_the_plain_network(x, sideinfo, key):
    net = FiLMMLP(IN, COND, OUT, WIDTH, DEPTH, key=key)
    unmodulated = eqx.tree_at(
        lambda n: n.film_layers, net, jax.tree.map(jnp.zeros_like, net.film_layers)
    )

    np.testing.assert_allclose(unmodulated(x, sideinfo), net(x), atol=1e-6)
    assert not np.allclose(net(x, sideinfo), net(x))


def test_film_scales_and_shifts_each_hidden_layer(x, sideinfo, key):
    net = FiLMMLP(IN, COND, OUT, WIDTH, 1, key=key)

    gamma, beta = jnp.split(net.film_layers[0](sideinfo), 2)
    hidden = jax.nn.relu((1.0 + gamma) * net.layers[0](x) + beta)

    np.testing.assert_allclose(net(x, sideinfo), net.final_layer(hidden), rtol=1e-5, atol=1e-6)


def test_hypernetwork_generates_the_first_layer(x, sideinfo, key):
    # A hypernetwork that outputs the default first layer's weights, flattened row by row, and
    # its bias must reproduce the network without conditioning.
    net = HyperMLP(IN, COND, OUT, WIDTH, DEPTH, key=key)
    first = net.default_first_layer
    last = net.hyper_net.layers[-1]
    generating_default = eqx.tree_at(
        lambda n: (n.hyper_net.layers[-1].weight, n.hyper_net.layers[-1].bias),
        net,
        (jnp.zeros_like(last.weight), jnp.concatenate([first.weight.ravel(), first.bias])),
    )

    np.testing.assert_allclose(generating_default(x, sideinfo), net(x), atol=1e-6)
    assert not np.allclose(net(x, sideinfo), net(x))


def test_batched_layers_match_the_per_token_layers(key):
    tokens = jax.random.normal(key, (6, 8))
    norm = eqx.nn.LayerNorm(8)
    norm = eqx.tree_at(lambda n: (n.weight, n.bias), norm, (jnp.arange(8.0), jnp.ones(8)))
    linear = eqx.nn.Linear(8, 3, key=key)

    np.testing.assert_allclose(batch_layer_norm(tokens, norm), jax.vmap(norm)(tokens), atol=1e-5)
    np.testing.assert_allclose(batch_linear(tokens, linear), jax.vmap(linear)(tokens), atol=1e-6)


class TestCrossAttention:
    @pytest.fixture
    def block(self, key) -> CrossAttentionBlock:
        return CrossAttentionBlock(8, 8, num_heads=2, ff_multiplier=2, key=key)

    @pytest.fixture
    def tokens(self, key) -> tuple[jax.Array, jax.Array]:
        token_key, cond_key = jax.random.split(key)
        return jax.random.normal(token_key, (5, 8)), jax.random.normal(cond_key, (3, 8))

    def test_permuting_the_tokens_permutes_the_output(self, block, tokens):
        x, cond = tokens
        perm = jnp.array([3, 0, 4, 1, 2])

        np.testing.assert_allclose(block(x[perm], cond), block(x, cond)[perm], atol=1e-5)

    def test_conditioning_tokens_are_read_as_a_set(self, block, tokens):
        x, cond = tokens

        np.testing.assert_allclose(block(x, cond[::-1]), block(x, cond), atol=1e-5)
        assert not np.allclose(block(x, -cond), block(x, cond))

    def test_heads_must_divide_the_token_width(self, key):
        with pytest.raises(ValueError, match='divisible'):
            CrossAttentionBlock(8, 8, num_heads=3, ff_multiplier=2, key=key)
