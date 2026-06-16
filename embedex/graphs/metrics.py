import jax
import jax.numpy as jnp


def roc_auc(scores: jax.Array, labels: jax.Array) -> jax.Array:
    def _average_ranks(scores):
        n = scores.shape[0]

        order = jnp.argsort(scores, stable=True)
        sorted_scores = scores[order]

        is_new = jnp.concatenate([jnp.array([True]), sorted_scores[1:] != sorted_scores[:-1]])
        group = jnp.cumsum(is_new) - 1
        pos = jnp.arange(n, dtype=jnp.float32)

        group_sum = jax.ops.segment_sum(pos, group, num_segments=n)
        group_cnt = jax.ops.segment_sum(jnp.ones(n), group, num_segments=n)
        group_avg = group_sum / jnp.maximum(group_cnt, 1.0)

        avg_rank_sorted = group_avg[group] + 1.0

        return jnp.zeros(n).at[order].set(avg_rank_sorted)

    labels = labels.astype(jnp.bool_)
    num_pos = jnp.sum(labels)
    num_neg = labels.shape[0] - num_pos

    ranks = _average_ranks(scores)
    pos_rank_sum = jnp.sum(jnp.where(labels, ranks, 0.0))
    auc = (pos_rank_sum - num_pos * (num_pos + 1) / 2.0) / jnp.maximum(num_pos * num_neg, 1)

    return jnp.where((num_pos == 0) | (num_neg == 0), jnp.nan, auc)


def average_precision(scores: jax.Array, labels: jax.Array) -> jax.Array:
    labels = labels.astype(jnp.bool_)
    num_pos = jnp.sum(labels)

    order = jnp.argsort(-scores, stable=True)
    labels = labels[order]

    tp_cum = jnp.cumsum(labels)
    fp_cum = jnp.cumsum(~labels)

    precision = tp_cum / jnp.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / jnp.maximum(num_pos, 1)
    recall_prev = jnp.concatenate([jnp.array([0.0]), recall[:-1]])

    return jnp.where(num_pos == 0, jnp.nan, jnp.sum((recall - recall_prev) * precision))


def confusion_counts(pred: jax.Array, true: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    if mask is None:
        mask = jnp.ones_like(pred, jnp.bool_)

    tp = jnp.sum(mask & pred & true)
    fp = jnp.sum(mask & pred & ~true)
    fn = jnp.sum(mask & ~pred & true)

    return jnp.array([tp, fp, fn])


def precision_recall_f1(tp: jax.Array, fp: jax.Array, fn: jax.Array) -> dict[str, jax.Array]:
    precision = tp / jnp.maximum(tp + fp, 1.0)
    recall = tp / jnp.maximum(tp + fn, 1.0)
    f1 = jnp.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

    return {'precision': precision, 'recall': recall, 'f1': f1}


def ortho_violation(u: jax.Array, weights: jax.Array) -> jax.Array:
    weights = weights / jnp.sum(weights)
    gram = jnp.transpose(u) @ (weights[:, jnp.newaxis] * u)

    return jnp.max(jnp.abs(gram - jnp.eye(gram.shape[0])))
