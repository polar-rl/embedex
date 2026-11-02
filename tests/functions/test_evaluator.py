"""Decoders, which read a function's values at new inputs back out of its embedding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.functions.builder import create_embedding_network
from embedex.functions.evaluator import AttentionDecoder


@pytest.mark.parametrize('output_size', [1, 3])
def test_decoders_predict_every_output_dimension(decoder, output_size, make_config, key):
    cfg = make_config(decoder=decoder)
    cfg.model.output_size = output_size
    evaluator = create_embedding_network(cfg, key).evaluator

    prediction = evaluator(jnp.ones(1), jnp.ones(cfg.model.embedding.embedding_size))

    assert prediction.shape == (output_size,)


def test_attention_decoder_averages_rows_of_values(key):
    # Softmax weights are non-negative and sum to one, so every prediction lies within the
    # range of the value rows that the embedding is projected into.
    decoder = AttentionDecoder(1, 8, 4, 6, 3, 16, 1, jax.nn.relu, key)
    z = jax.random.normal(jax.random.key(1), (8,))
    values = np.asarray(decoder.v_proj(z).reshape(6, 3))

    predictions = np.asarray(
        jax.vmap(decoder, in_axes=(0, None))(jnp.linspace(-5, 5, 64)[:, None], z)
    )

    assert np.all(predictions >= values.min(axis=0) - 1e-5)
    assert np.all(predictions <= values.max(axis=0) + 1e-5)
    assert len(np.unique(predictions.round(5), axis=0)) > 1  # the weights depend on the input
