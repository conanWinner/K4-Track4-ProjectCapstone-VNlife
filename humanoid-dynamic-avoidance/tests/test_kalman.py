import numpy as np
import pytest

from humanoid_avoidance.tracking.kalman_cv import (
    F_cv, H_CV, Q_cv, init_state, kf_predict, kf_update, mahalanobis_sq,
)

SIGMA_A = 0.7
INIT_V_SIGMA = 2.0


def _assert_sym_psd(P):
    assert np.allclose(P, P.T, atol=1e-12)
    assert np.linalg.eigvalsh(P).min() >= -1e-12


def test_q_matrix_structure():
    dt, s = 0.1, 0.5
    Q = Q_cv(dt, s)
    s2 = s * s
    assert Q.shape == (4, 4)
    for ax in (0, 1):
        assert Q[ax, ax] == pytest.approx(s2 * dt ** 4 / 4)
        assert Q[ax, ax + 2] == pytest.approx(s2 * dt ** 3 / 2)
        assert Q[ax + 2, ax] == pytest.approx(s2 * dt ** 3 / 2)
        assert Q[ax + 2, ax + 2] == pytest.approx(s2 * dt ** 2)
    # no cross-axis coupling
    assert Q[0, 1] == Q[0, 3] == Q[1, 2] == Q[2, 3] == 0.0
    _assert_sym_psd(Q)
    # piecewise-constant accel: Q = G G^T sigma^2 with G = [dt^2/2, dt]
    G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
    assert np.allclose(Q, s2 * G @ G.T)
    assert np.allclose(Q_cv(0.0, s), 0.0)
    assert np.allclose(F_cv(0.3) @ [1, 2, 3, 4], [1.9, 3.2, 3, 4])
    assert H_CV.shape == (2, 4)


def test_init_state():
    R = np.diag([0.01, 0.02])
    x, P = init_state(np.array([1.0, 2.0]), R, INIT_V_SIGMA)
    assert np.allclose(x, [1, 2, 0, 0])
    assert np.allclose(P[:2, :2], R) and np.allclose(np.diag(P)[2:], INIT_V_SIGMA ** 2)


def test_predict_dt_zero_noop_and_negative_raises():
    x, P = init_state(np.zeros(2), np.eye(2) * 0.01, INIT_V_SIGMA)
    x2, P2 = kf_predict(x, P, 0.0, SIGMA_A)
    assert np.array_equal(x2, x) and np.array_equal(P2, P)
    assert x2 is not x
    with pytest.raises(ValueError):
        kf_predict(x, P, -0.01, SIGMA_A)


def _run_cv(rng, times, v_true, p0, sigma_m, sigma_a=SIGMA_A, drop=frozenset()):
    R = np.eye(2) * sigma_m ** 2
    z0 = p0 + v_true * times[0] + rng.normal(0, sigma_m, 2)
    x, P = init_state(z0, R, INIT_V_SIGMA)
    t_prev = times[0]
    traces = []
    for k, t in enumerate(times[1:], start=1):
        x, P = kf_predict(x, P, t - t_prev, sigma_a)
        t_prev = t
        _assert_sym_psd(P)
        if k not in drop:
            z = p0 + v_true * t + rng.normal(0, sigma_m, 2)
            x, P, nis = kf_update(x, P, z, R)
            assert np.isfinite(nis) and nis >= 0
            _assert_sym_psd(P)
        traces.append(np.trace(P))
    return x, P, traces


# For a truly constant-velocity target the matched tuning is a small sigma_a. With the default
# sigma_a=0.7 m/s^2 the steady-state velocity std is ~0.13 m/s per axis at sigma_m=0.05 m, so a
# single-run "< 0.1 m/s" check would fail often by design, not because of a bug.
CV_SIGMA_A = 0.2


def test_converges_to_true_velocity_10hz():
    v_true = np.array([0.8, -0.4])
    p0 = np.array([1.0, 2.0])
    times = np.arange(0, 3.0 + 1e-9, 0.1)          # ~3 s at 10 Hz
    v_err, p_err = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        x, P, _ = _run_cv(rng, times, v_true, p0, sigma_m=0.05, sigma_a=CV_SIGMA_A)
        v_err.append(np.linalg.norm(x[2:] - v_true))
        p_err.append(np.linalg.norm(x[:2] - (p0 + v_true * times[-1])))
    v_err = np.array(v_err)
    assert np.sqrt(np.mean(v_err ** 2)) < 0.1      # RMS velocity error over 10 seeds
    assert v_err.max() < 0.2
    assert max(p_err) < 0.15


def test_velocity_starts_at_zero_and_uncertain():
    rng = np.random.default_rng(42)
    times = np.arange(0, 0.25, 0.1)
    x, P, _ = _run_cv(rng, times[:1], np.array([1.0, 0.0]), np.zeros(2), sigma_m=0.05)
    assert np.allclose(x[2:], 0.0) and P[2, 2] == pytest.approx(INIT_V_SIGMA ** 2)


def test_variable_dt():
    rng = np.random.default_rng(7)
    dts = rng.uniform(0.03, 0.2, 60)
    times = np.concatenate([[0.0], np.cumsum(dts)])
    v_true = np.array([-0.5, 1.0])
    x, P, _ = _run_cv(rng, times, v_true, np.zeros(2), sigma_m=0.05, sigma_a=CV_SIGMA_A)
    assert np.linalg.norm(x[2:] - v_true) < 0.15


@pytest.mark.parametrize("n_drop", [2, 3, 4, 5])
def test_dropout_predict_only_grows_covariance(n_drop):
    rng = np.random.default_rng(10 + n_drop)
    times = np.arange(0, 5.0 + 1e-9, 0.1)
    start = 30
    drop = frozenset(range(start, start + n_drop))
    v_true = np.array([0.6, 0.2])
    x, P, traces = _run_cv(rng, times, v_true, np.zeros(2), sigma_m=0.05, sigma_a=CV_SIGMA_A,
                           drop=drop)
    # traces[k-1] is P after step k. From the last update (k=start-1) through the predict-only
    # steps (k=start .. start+n_drop-1) the trace strictly increases.
    seg = traces[start - 2: start + n_drop - 1]
    assert len(seg) == n_drop + 1
    assert all(b > a for a, b in zip(seg, seg[1:]))
    # first update after the gap (k=start+n_drop) shrinks it again
    assert traces[start + n_drop - 1] < traces[start + n_drop - 2]
    assert np.linalg.norm(x[2:] - v_true) < 0.2


def test_nis_consistent_filter():
    """Simulate exactly the filter's model (piecewise-constant accel + Gaussian R): mean NIS ~ 2."""
    rng = np.random.default_rng(123)
    dt, sigma_a, sigma_m = 0.1, 0.7, 0.1
    R = np.eye(2) * sigma_m ** 2
    F, Q = F_cv(dt), Q_cv(dt, sigma_a)
    L = np.linalg.cholesky(Q + 1e-15 * np.eye(4))
    x_true = np.array([0.0, 0.0, 0.5, -0.2])
    x, P = init_state(x_true[:2] + rng.normal(0, sigma_m, 2), R, INIT_V_SIGMA)
    nis_vals = []
    for k in range(3000):
        x_true = F @ x_true + L @ rng.normal(size=4)
        x, P = kf_predict(x, P, dt, sigma_a)
        z = x_true[:2] + rng.normal(0, sigma_m, 2)
        d2 = mahalanobis_sq(x, P, z, R)
        x, P, nis = kf_update(x, P, z, R)
        assert d2 == pytest.approx(nis)
        if k >= 50:
            nis_vals.append(nis)
    mean_nis = float(np.mean(nis_vals))
    assert 1.7 < mean_nis < 2.3
    _assert_sym_psd(P)


def test_joseph_matches_standard_form_and_inputs_untouched():
    x, P = init_state(np.array([1.0, 1.0]), np.eye(2) * 0.04, INIT_V_SIGMA)
    x, P = kf_predict(x, P, 0.1, SIGMA_A)
    x0, P0 = x.copy(), P.copy()
    R = np.diag([0.01, 0.03])
    z = np.array([1.1, 0.9])
    x1, P1, _ = kf_update(x, P, z, R)
    S = H_CV @ P @ H_CV.T + R
    K = P @ H_CV.T @ np.linalg.inv(S)
    assert np.allclose(x1, x + K @ (z - H_CV @ x))
    assert np.allclose(P1, (np.eye(4) - K @ H_CV) @ P)
    assert np.array_equal(x, x0) and np.array_equal(P, P0)
