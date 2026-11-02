"""The relational stream's three arms: widths matched, rawA is the statistic, alias resolves.

The ablation of EXPERIMENTS_MRF_V2 section 1 compares the learned stream against two arms
that emit the same conditioning width, so a difference between them cannot be a width
effect. These tests pin that, and pin the deprecated `relational: bool` alias every fusion3
checkpoint was written with.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.distributions.builder import create_model, create_relational
from embedex.distributions.config import EmbeddingConfig, GeneratorConfig, ModelConfig
from embedex.distributions.relational import (
    RawAgreeStream,
    RawEdgeStream,
    RelationalStream,
)

MOMENT_DIM = 32


def _config(mode, num_variables, num_classes):
    return ModelConfig(
        input_dim=num_variables,
        embedding=EmbeddingConfig(
            block_type='set_transformer',
            embedding_dim=MOMENT_DIM,
            num_classes=num_classes,
            relational_mode=mode,
        ),
        generator=GeneratorConfig(flow_type='ar', ar_num_classes=num_classes),
    )


def _stream(mode, num_variables, num_classes):
    cfg = _config(mode, num_variables, num_classes)
    return create_model(cfg, dequantize=False, key=jax.random.key(0)).relational


# (num_variables, num_classes, conditioning_dim) for the two evaluated configurations.
SHAPES = [(18, 2, MOMENT_DIM + 18 * 18), (9, 4, MOMENT_DIM + 9 * 9)]


@pytest.mark.parametrize(('num_variables', 'num_classes', 'conditioning_dim'), SHAPES)
def test_arms_share_a_conditioning_width(num_variables, num_classes, conditioning_dim):
    widths = {
        mode: MOMENT_DIM + _stream(mode, num_variables, num_classes).out_dim
        for mode in ('learned', 'raw_agree', 'raw_edge')
    }
    assert set(widths.values()) == {conditioning_dim}, widths


def test_modes_build_their_own_stream():
    assert isinstance(_stream('learned', 9, 4), RelationalStream)
    assert isinstance(_stream('raw_agree', 9, 4), RawAgreeStream)
    assert isinstance(_stream('raw_edge', 9, 4), RawEdgeStream)
    assert create_relational(_config('none', 9, 4), jax.random.key(0)) is None


def test_arm_parameter_counts_order_as_designed():
    def count(mode):
        stream = _stream(mode, 18, 2)
        return sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(stream, eqx.is_inexact_array))
        )

    # rawA is the statistic itself; rawE keeps the embedding and projection but not the
    # attention, so the ordering is a structural property rather than a tuned one.
    assert count('raw_agree') == 0
    assert 0 < count('raw_edge') < count('learned')


def test_raw_agree_is_the_agreement_matrix():
    rng = np.random.default_rng(1)
    xs = rng.integers(0, 4, size=(64, 9))
    stream = _stream('raw_agree', 9, 4)

    got = np.asarray(stream(jnp.asarray(xs, dtype=jnp.float32)))
    want = np.mean((xs[:, :, None] == xs[:, None, :]).astype(float), axis=0)

    assert np.allclose(got.reshape(9, 9), want, atol=1e-6)
    assert np.allclose(np.diag(got.reshape(9, 9)), 1.0, atol=1e-6)


def test_deprecated_alias_resolves_to_learned():
    emb = EmbeddingConfig(relational=True, num_classes=2)
    assert emb.rel_mode == 'learned'
    assert EmbeddingConfig(relational=False).rel_mode == 'none'
    assert EmbeddingConfig(relational_mode='raw_edge').rel_mode == 'raw_edge'


def test_contradicting_the_alias_raises():
    # Otherwise a config setting both fields would ignore one of them and build the wrong model
    # in silence.
    with pytest.raises(ValueError, match='contradicts'):
        _ = EmbeddingConfig(relational=True, relational_mode='raw_agree').rel_mode
