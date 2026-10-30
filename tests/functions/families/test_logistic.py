"""The logistic ODE y' = r y (1 - y / K), sampled from its closed-form solution."""

import numpy as np
import pytest

BATCH, SAMPLES = 32, 64


@pytest.fixture
def samples(families, key):
    ts, ys, params = families['logistic'].generator().sample(BATCH, SAMPLES, key)
    y0, r, K = (np.asarray(params[:, i, np.newaxis]) for i in range(3))
    return np.asarray(ts[..., 0]), np.asarray(ys[..., 0]), y0, r, K


def test_log_odds_of_the_capacity_grow_linearly_at_rate_r(samples):
    # Integrating the ODE gives ln|y / (K - y)| = ln|y0 / (K - y0)| + r t.
    ts, ys, y0, r, K = np.broadcast_arrays(*samples)
    resolved = np.abs(K - ys) > 1e-2 * K  # float32 cannot resolve y right next to K
    ts, ys, y0, r, K = (a[resolved] for a in (ts, ys, y0, r, K))

    log_odds = np.log(np.abs(ys / (K - ys)))
    expected = np.log(np.abs(y0 / (K - y0))) + r * ts

    assert resolved.mean() > 0.5
    np.testing.assert_allclose(log_odds, expected, rtol=1e-3, atol=1e-3)


def test_populations_move_monotonically_from_y0_towards_the_capacity(samples):
    ts, ys, y0, _, K = samples

    for t, y, start, capacity in zip(ts, ys, y0[:, 0], K[:, 0], strict=True):
        y = y[np.argsort(t)]
        steps = np.diff(y) * np.sign(capacity - start)

        assert np.all(steps >= -1e-6)
        assert np.all((y - start) * (y - capacity) <= 1e-6)  # between y0 and K
