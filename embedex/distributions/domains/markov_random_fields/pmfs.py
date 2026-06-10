import numpy as np


def state_indices(samples: np.ndarray, num_spins: int, num_states: int) -> np.ndarray:
    """Map (..., num_spins) categorical vectors to support indices."""
    cats = np.clip(np.round(np.asarray(samples, dtype=float)), 0, num_states - 1)
    powers = num_states ** np.arange(num_spins, dtype=np.int64)
    return cats.astype(np.int64) @ powers


def _finite(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    return samples[np.all(np.isfinite(samples), axis=-1)]


def _class_marginals_from_samples(cats: np.ndarray, num_states: int) -> np.ndarray:
    onehot = cats[:, :, None] == np.arange(num_states)
    return onehot.mean(axis=0)


def empirical_pmf(samples: np.ndarray, num_spins: int, num_states: int) -> np.ndarray:
    """Histogram of the samples over the enumerated support."""
    support_size = num_states**num_spins
    samples = _finite(samples)
    if len(samples) == 0:
        return np.full(support_size, 1.0 / support_size)
    counts = np.bincount(state_indices(samples, num_spins, num_states), minlength=support_size)
    return counts / counts.sum()


def mean_field_pmf(samples: np.ndarray, states: np.ndarray, num_states: int) -> np.ndarray:
    """Product of per-node empirical class marginals (first-order structure only)."""
    cats = np.round(_finite(samples)).astype(np.int64)
    marginals = _class_marginals_from_samples(cats, num_states)

    with np.errstate(divide='ignore'):
        log_marginals = np.log(marginals)
    node_idx = np.arange(states.shape[1])
    log_pmf = log_marginals[node_idx[None, :], states].sum(axis=1)
    return np.exp(log_pmf)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (nats) between two PMFs; no smoothing, 0 log 0 = 0."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def kl_to_mixture(a: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / m[mask])))

    return 0.5 * kl_to_mixture(p) + 0.5 * kl_to_mixture(q)


def effective_support(pmf: np.ndarray, mass: float = 0.99) -> np.ndarray:
    """Indices of the smallest set of states covering `mass` of the PMF."""
    order = np.argsort(pmf)[::-1]
    cutoff = int(np.searchsorted(np.cumsum(pmf[order]), mass)) + 1
    return order[:cutoff]


def class_marginals(pmf: np.ndarray, states: np.ndarray, num_states: int) -> np.ndarray:
    onehot = states[:, :, None] == np.arange(num_states)
    return np.einsum('m,mic->ic', pmf, onehot)


def agreement_marginals(pmf: np.ndarray, states: np.ndarray) -> np.ndarray:
    agree = states[:, :, None] == states[:, None, :]
    return np.einsum('m,mij->ij', pmf, agree)


def pmf_metrics(
    true_pmf: np.ndarray,
    model_pmf: np.ndarray,
    states: np.ndarray,
    num_states: int,
    support_mass: float = 0.99,
    top_k: int = 8,
) -> dict[str, float]:
    """Scalar comparison metrics between the exact PMF and a model PMF."""
    support = effective_support(true_pmf, support_mass)
    class_diff = class_marginals(true_pmf, states, num_states) - class_marginals(
        model_pmf, states, num_states
    )
    upper = np.triu_indices(states.shape[1], k=1)
    agree_diff = (agreement_marginals(true_pmf, states) - agreement_marginals(model_pmf, states))[
        upper
    ]

    true_top = set(np.argsort(true_pmf)[::-1][:top_k].tolist())
    model_top = set(np.argsort(model_pmf)[::-1][:top_k].tolist())

    return {
        'jsd': js_divergence(true_pmf, model_pmf),
        'class_mae': float(np.abs(class_diff).mean()),
        'agree_mae': float(np.abs(agree_diff).mean()),
        'mass_outside': float(1.0 - model_pmf[support].sum()),
        'mode_coverage': len(true_top & model_top) / top_k,
    }
