"""Operator networks, which map an operator's input (an embedding or ODE parameters) to an
embedding of its output."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.functions.builder import create_operator
from embedex.functions.config import OperatorConfig
from embedex.functions.operator import ResidualBlock


@pytest.mark.parametrize('kind', ['mlp', 'residual'])
def test_operators_map_inputs_to_embeddings(kind, key):
    operator = create_operator(OperatorConfig(kind=kind, hidden_size=16, num_blocks=2), 5, 8, key)

    assert operator(jnp.ones(5)).shape == (8,)


def test_residual_block_adds_its_feed_forward_output_to_the_input(key):
    block = ResidualBlock(4, 8, jax.nn.relu, key)
    x = jax.random.normal(jax.random.key(1), (4,))
    silenced = eqx.tree_at(lambda b: b.ff.layers[-1].bias, block, jnp.zeros(4))
    silenced = eqx.tree_at(lambda b: b.ff.layers[-1].weight, silenced, jnp.zeros((4, 8)))

    np.testing.assert_allclose(silenced(x), x)
    np.testing.assert_allclose(block(x), x + block.ff(block.ln(x)), rtol=1e-6)
