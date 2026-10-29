import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embedex.graphs.metrics import (
    average_precision,
    compute_average_ranks,
    confusion_counts,
    ortho_violation,
    precision_recall_f1,
    roc_auc,
    spearman,
)


def test_average_ranks_match_scipy():
    from scipy.stats import rankdata

    scores = np.random.default_rng(0).normal(size=20)
    np.testing.assert_allclose(compute_average_ranks(jnp.asarray(scores)), rankdata(scores))


def test_average_ranks_share_ties():
    from scipy.stats import rankdata

    # Tied scores share the mean of the ranks they span.
    scores = jnp.array([2.0, 0.0, 2.0, 1.0, 0.0, 2.0])
    np.testing.assert_allclose(compute_average_ranks(scores), [5.0, 1.5, 5.0, 3.0, 1.5, 5.0])
    np.testing.assert_allclose(compute_average_ranks(scores), rankdata(np.asarray(scores)))


def test_ortho_violation_zero_for_orthonormal_basis():
    # Columns are orthogonal with squared norm V, so the uniform-weighted Gram
    # matrix ((1/V) u^T u) is exactly the identity.
    u = jnp.array([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]])
    assert float(ortho_violation(u, jnp.ones(4))) == pytest.approx(0.0, abs=1e-6)


def test_ortho_violation_detects_collinear_basis():
    # Both eigenfunctions identical -> off-diagonal Gram entry is 1.
    u = jnp.ones((4, 2))
    assert float(ortho_violation(u, jnp.ones(4))) == pytest.approx(1.0)


def test_ortho_violation_uses_weights():
    # Uniform basis but non-uniform weights: weighted Gram is diag(0.25, 0.75),
    # so the largest deviation from the identity is 1 - 0.25 = 0.75.
    u = jnp.eye(2)
    assert float(ortho_violation(u, weights=jnp.array([1.0, 3.0]))) == pytest.approx(0.75)
    # Equal weights give the uniform-weighted Gram diag(0.5, 0.5), so deviation 0.5.
    assert float(ortho_violation(u, weights=jnp.ones(2))) == pytest.approx(0.5)


def test_confusion_counts_respects_mask():
    pred = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    true = jnp.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)
    mask = jnp.triu(jnp.ones((3, 3), bool), k=1)

    # Upper triangle only: TP={0-1}, FP={1-2}, FN={0-2}.
    np.testing.assert_array_equal(np.array(confusion_counts(pred, true, mask)), [1, 1, 1])
    # Whole matrix double-counts each undirected pair.
    np.testing.assert_array_equal(np.array(confusion_counts(pred, true)), [2, 2, 2])


def test_precision_recall_f1_known_value():
    m = precision_recall_f1(3, 1, 0)
    assert float(m['precision']) == pytest.approx(0.75)
    assert float(m['recall']) == pytest.approx(1.0)
    assert float(m['f1']) == pytest.approx(2 * 0.75 / 1.75)


def test_precision_recall_f1_empty_counts_are_zero():
    m = precision_recall_f1(0, 0, 0)
    assert (float(m['precision']), float(m['recall']), float(m['f1'])) == (0.0, 0.0, 0.0)


def test_precision_recall_f1_accepts_confusion_counts_output():
    pred = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    true = jnp.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)
    mask = jnp.triu(jnp.ones((3, 3), bool), k=1)

    m = precision_recall_f1(*confusion_counts(pred, true, mask))
    assert (float(m['precision']), float(m['recall']), float(m['f1'])) == (0.5, 0.5, 0.5)


def test_roc_auc_known_value():
    scores = jnp.array([0.1, 0.4, 0.35, 0.8])
    labels = jnp.array([0, 0, 1, 1])
    assert float(roc_auc(scores, labels)) == pytest.approx(0.75)


def test_roc_auc_perfect_and_reversed_ranking():
    labels = jnp.array([0, 0, 1, 1])
    assert float(roc_auc(jnp.array([0.0, 1.0, 2.0, 3.0]), labels)) == pytest.approx(1.0)
    assert float(roc_auc(jnp.array([3.0, 2.0, 1.0, 0.0]), labels)) == pytest.approx(0.0)


def test_roc_auc_all_ties_is_chance():
    # Every score tied -> average ranks give exactly 0.5.
    scores = jnp.ones(4)
    assert float(roc_auc(scores, jnp.array([0, 1, 0, 1]))) == pytest.approx(0.5)


def test_roc_auc_nan_when_class_absent():
    scores = jnp.array([0.2, 0.7, 0.5])
    assert np.isnan(roc_auc(scores, jnp.array([0, 0, 0])))
    assert np.isnan(roc_auc(scores, jnp.array([1, 1, 1])))


def test_average_precision_known_value():
    scores = jnp.array([0.1, 0.4, 0.35, 0.8])
    labels = jnp.array([0, 0, 1, 1])
    assert float(average_precision(scores, labels)) == pytest.approx(0.8333333, abs=1e-6)


def test_average_precision_perfect_ranking_is_one():
    scores = jnp.array([3.0, 2.0, 1.0, 0.0])
    labels = jnp.array([1, 1, 0, 0])
    assert float(average_precision(scores, labels)) == pytest.approx(1.0)


def test_average_precision_nan_without_positives():
    assert np.isnan(average_precision(jnp.array([0.2, 0.7, 0.5]), jnp.array([0, 0, 0])))


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = 0.7 * x + rng.normal(size=50)  # correlated, with ties unlikely
    assert float(spearman(jnp.asarray(x), jnp.asarray(y))) == pytest.approx(
        spearmanr(x, y).statistic, abs=1e-6
    )


def test_spearman_handles_ties_like_scipy():
    from scipy.stats import spearmanr

    # Many tied values on both sides (mimics zero-weight non-edges).
    x = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0])
    y = jnp.array([0.0, 0.0, 1.0, 1.0, 3.0, 3.0])
    assert float(spearman(x, y)) == pytest.approx(spearmanr(np.array(x), np.array(y)).statistic)


def test_spearman_perfect_monotone():
    x = jnp.array([0.1, 0.2, 0.3, 0.9])
    assert float(spearman(x, 2 * x + 1)) == pytest.approx(1.0)
    assert float(spearman(x, -x)) == pytest.approx(-1.0)


def test_spearman_mask_restricts_to_subset():
    from scipy.stats import spearmanr

    # Ranking is computed within the masked subset only; out-of-mask entries are
    # adversarial (anti-correlated) yet must not affect the result.
    scores = jnp.array([5.0, 1.0, 2.0, 9.0, 3.0])
    values = jnp.array([0.0, 0.6, 0.7, 0.0, 0.9])
    mask = values > 0.0
    expected = spearmanr(np.array(scores)[np.array(mask)], np.array(values)[np.array(mask)])
    assert float(spearman(scores, values, mask)) == pytest.approx(expected.statistic)


def test_spearman_nan_without_variation():
    # All weights equal (an unweighted graph) -> undefined, returns NaN.
    assert np.isnan(spearman(jnp.array([3.0, 1.0, 2.0]), jnp.ones(3)))
    # Fewer than two kept entries -> NaN.
    assert np.isnan(
        spearman(jnp.array([1.0, 2.0]), jnp.array([0.0, 5.0]), jnp.array([False, True]))
    )


def test_spearman_jit_and_vmap():
    scores = jnp.array([[0.1, 0.4, 0.35, 0.8], [3.0, 2.0, 1.0, 0.0]])
    values = jnp.array([[0.1, 0.4, 0.35, 0.8], [0.0, 1.0, 2.0, 3.0]])

    rhos = jax.jit(jax.vmap(spearman))(scores, values)
    np.testing.assert_allclose(np.array(rhos), [1.0, -1.0], atol=1e-6)


def test_metrics_are_jit_compatible():
    scores = jnp.array([0.1, 0.4, 0.35, 0.8])
    labels = jnp.array([0, 0, 1, 1])

    assert float(jax.jit(roc_auc)(scores, labels)) == pytest.approx(0.75)
    assert float(jax.jit(average_precision)(scores, labels)) == pytest.approx(0.8333333, abs=1e-6)
    assert float(jax.jit(ortho_violation)(jnp.ones((4, 2)), jnp.ones(4))) == pytest.approx(1.0)

    prf = jax.jit(precision_recall_f1)(3, 1, 0)
    assert float(prf['f1']) == pytest.approx(2 * 0.75 / 1.75)


def test_metrics_vmap_over_batch():
    # Two independent score/label sets graded in one vmapped call.
    scores = jnp.array([[0.1, 0.4, 0.35, 0.8], [3.0, 2.0, 1.0, 0.0]])
    labels = jnp.array([[0, 0, 1, 1], [0, 0, 1, 1]])

    aucs = jax.vmap(roc_auc)(scores, labels)
    np.testing.assert_allclose(np.array(aucs), [0.75, 0.0])

    aps = jax.vmap(average_precision)(scores, labels)
    np.testing.assert_allclose(np.array(aps), [0.8333333, 0.4166667], atol=1e-6)

    counts = jnp.array([[3, 1, 0], [1, 1, 1]])
    prf = jax.vmap(lambda c: precision_recall_f1(*c))(counts)
    np.testing.assert_allclose(np.array(prf['precision']), [0.75, 0.5])
    np.testing.assert_allclose(np.array(prf['recall']), [1.0, 0.5])
