import numpy as np
import pytest

from humanoid_avoidance.prediction.risk import (
    assess_risk, assess_track_risk, predict_track, time_of_closest_approach,
)
from humanoid_avoidance.tracking.kalman_cv import init_state, kf_predict, kf_update
from humanoid_avoidance.types import NO_RISK, RobotState, Track

PRED = {
    "horizon_s": 2.5,
    "step_s": 0.1,
    "nominal_margin_m": 0.35,
    "uncertainty_k": 2.0,
    "approach_speed_eps_mps": 0.02,
}
SIGMA_A = 0.7
R_ROBOT = 0.35


def _robot(xy=(0.0, 0.0), v=(0.0, 0.0), t=0.0):
    return RobotState(time_s=t, xy_world_m=np.array(xy, float), yaw_rad=0.0,
                      v_world_mps=np.array(v, float), yaw_rate_rps=0.0)


def _track(xy, v, pos_sigma=0.05, vel_sigma=0.05, tid=1, confirmed=True, radius=0.25, t=0.0):
    P = np.diag([pos_sigma ** 2] * 2 + [vel_sigma ** 2] * 2)
    return Track(track_id=tid, time_s=t, state=np.array([*xy, *v], float), covariance=P,
                 radius_m=radius, confirmed=confirmed, missed_updates=0, hits=5)


def test_tca_formula():
    tau, dca = time_of_closest_approach(np.array([4.0, 0.0]), np.array([-2.0, 0.0]), 10.0)
    assert tau == pytest.approx(2.0, abs=1e-5) and dca == pytest.approx(0.0, abs=1e-4)
    tau, dca = time_of_closest_approach(np.array([4.0, 1.0]), np.array([-2.0, 0.0]), 1.0)
    assert tau == 1.0 and dca == pytest.approx(np.hypot(2.0, 1.0))   # clipped to horizon


def test_head_on_approaching_is_risk():
    r = assess_track_risk(_robot(v=(0.3, 0.0)), _track((3.0, 0.0), (-1.0, 0.0)), R_ROBOT, PRED, SIGMA_A)
    assert r.approaching and r.risk
    assert r.tca_s == pytest.approx(3.0 / 1.3, abs=1e-3)
    assert r.dca_m == pytest.approx(0.0, abs=1e-3)
    assert 0.0 < r.score <= 1.0


def test_crossing_path_risk_with_expected_tau():
    # robot east at 1 m/s, pedestrian north at 1 m/s, both reach (2, 0) at t = 2 s
    r = assess_track_risk(_robot(v=(1.0, 0.0)), _track((2.0, -2.0), (0.0, 1.0)), R_ROBOT, PRED, SIGMA_A)
    assert r.risk and r.approaching
    assert r.tca_s == pytest.approx(2.0, abs=1e-3)
    assert r.dca_m < 1e-3


def test_parallel_same_speed_no_risk():
    r = assess_track_risk(_robot(v=(0.5, 0.0)), _track((1.0, 1.5), (0.5, 0.0)), R_ROBOT, PRED, SIGMA_A)
    assert not r.risk and not r.approaching
    assert r.tca_s == 0.0 and r.score == 0.0
    # same lane, 2 m ahead, same speed: also no risk (w = 0)
    r = assess_track_risk(_robot(v=(0.5, 0.0)), _track((2.0, 0.0), (0.5, 0.0)), R_ROBOT, PRED, SIGMA_A)
    assert not r.risk and r.tca_s == 0.0


def test_moving_away_no_risk():
    r = assess_track_risk(_robot(v=(0.3, 0.0)), _track((2.0, 0.0), (1.0, 0.0)), R_ROBOT, PRED, SIGMA_A)
    assert r.tca_s == 0.0 and not r.approaching and not r.risk and r.score == 0.0
    assert r.dca_m == pytest.approx(2.0)


def test_currently_overlapping_is_risk():
    r = assess_track_risk(_robot(), _track((0.5, 0.0), (0.0, 0.0)), R_ROBOT, PRED, SIGMA_A)
    assert r.risk and r.approaching and r.score == 1.0
    # overlapping even while separating
    r = assess_track_risk(_robot(), _track((0.5, 0.0), (0.5, 0.0)), R_ROBOT, PRED, SIGMA_A)
    assert r.risk


def test_zero_relative_velocity_handled():
    r = assess_track_risk(_robot(v=(0.4, 0.0)), _track((2.0, 0.0), (0.4, 1e-9)), R_ROBOT, PRED, SIGMA_A)
    assert np.isfinite([r.tca_s, r.dca_m, r.margin_m, r.score]).all()
    assert r.tca_s == 0.0 and r.dca_m == pytest.approx(2.0) and not r.risk
    tau, dca = time_of_closest_approach(np.array([1.0, 1.0]), np.zeros(2), 2.5)
    assert tau == 0.0 and dca == pytest.approx(np.sqrt(2))


def test_margin_grows_with_covariance():
    # near-miss passing 1.1 m to the side at tau = 1 s. Nominal clearance = 0.35+0.25+0.35 = 0.95 m.
    # Small sigma_a isolates the effect of the track covariance P.
    robot, sa = _robot(), 0.1
    low = assess_track_risk(robot, _track((1.0, 1.1), (-1.0, 0.0), 0.02, 0.02), R_ROBOT, PRED, sa)
    high = assess_track_risk(robot, _track((1.0, 1.1), (-1.0, 0.0), 0.3, 0.5), R_ROBOT, PRED, sa)
    assert low.tca_s == pytest.approx(1.0, abs=1e-4)
    assert high.margin_m > low.margin_m > PRED["nominal_margin_m"]
    assert high.required_clearance_m > low.required_clearance_m
    assert not low.risk and high.risk                  # uncertainty flips the decision


def test_margin_grows_with_lookahead_and_covariance_models():
    robot = _robot()
    near = assess_track_risk(robot, _track((1.0, 1.5), (-1.0, 0.0)), R_ROBOT, PRED, SIGMA_A)
    far = assess_track_risk(robot, _track((2.0, 1.5), (-1.0, 0.0)), R_ROBOT, PRED, SIGMA_A)
    assert far.tca_s > near.tca_s and far.margin_m > near.margin_m
    step = assess_track_risk(robot, _track((2.0, 1.5), (-1.0, 0.0)), R_ROBOT,
                             {**PRED, "covariance_model": "stepwise"}, SIGMA_A)
    assert PRED["nominal_margin_m"] < step.margin_m < far.margin_m
    # stepwise with a single step equal to tau reproduces the single-interval formula
    tr = _track((2.0, 1.5), (-1.0, 0.0))
    _, _, c1 = predict_track(tr, 1.0, 1.0, SIGMA_A)
    _, _, c2 = predict_track(tr, 1.0, 1.0, SIGMA_A, covariance_model="stepwise")
    assert np.allclose(c1, c2)
    with pytest.raises(ValueError):
        predict_track(tr, 1.0, 0.1, SIGMA_A, covariance_model="bogus")


def test_robot_velocity_is_measured_and_track_extrapolated():
    # stationary obstacle ahead; robot standing still -> no risk, robot walking -> risk
    tr = _track((1.5, 0.0), (0.0, 0.0))
    assert not assess_track_risk(_robot(v=(0.0, 0.0)), tr, R_ROBOT, PRED, SIGMA_A).risk
    assert assess_track_risk(_robot(v=(0.8, 0.0)), tr, R_ROBOT, PRED, SIGMA_A).risk
    # track at t=0 moving toward robot evaluated at robot time t=1: extrapolated 1 m closer
    tr = _track((3.0, 0.0), (-1.0, 0.0))
    r = assess_track_risk(_robot(t=1.0), tr, R_ROBOT, PRED, SIGMA_A)
    assert r.tca_s == pytest.approx(2.0, abs=1e-3)


def test_predict_track_shapes_and_growth():
    tr = _track((1.0, 2.0), (0.5, -0.5))
    times, means, covs = predict_track(tr, 2.5, 0.1, SIGMA_A)
    assert times.shape == (26,) and means.shape == (26, 2) and covs.shape == (26, 2, 2)
    assert times[0] == 0.0 and times[-1] == pytest.approx(2.5)
    assert np.allclose(means[-1], [2.25, 0.75])
    assert np.allclose(covs[0], tr.covariance[:2, :2])
    tr_ = [np.trace(c) for c in covs]
    assert all(b > a for a, b in zip(tr_, tr_[1:]))
    for c in covs:
        assert np.allclose(c, c.T) and np.linalg.eigvalsh(c).min() > 0


def test_assess_risk_worst_confirmed_only():
    robot = _robot(v=(0.3, 0.0))
    assert assess_risk(robot, [], R_ROBOT, PRED, SIGMA_A) == NO_RISK
    tent = _track((1.5, 0.0), (-1.0, 0.0), tid=1, confirmed=False)
    assert assess_risk(robot, [tent], R_ROBOT, PRED, SIGMA_A).track_id == -1
    far = _track((3.0, 0.0), (-1.0, 0.0), tid=2)        # risk, tau ~ 2.3 s
    near = _track((1.8, 0.2), (-1.0, 0.0), tid=3)       # risk, tau ~ 1.4 s
    away = _track((-2.0, 0.0), (-1.0, 0.0), tid=4)      # no risk
    worst = assess_risk(robot, [tent, far, away, near], R_ROBOT, PRED, SIGMA_A)
    assert worst.track_id == 3 and worst.risk
    only_safe = assess_risk(robot, [away], R_ROBOT, PRED, SIGMA_A)
    assert only_safe.track_id == 4 and not only_safe.risk


CAPPED = {**PRED, "nominal_margin_m": 0.25, "max_uncertainty_margin_m": 0.35}


def _steady_state_track(xy, sigma_m=0.1, sigma_a=SIGMA_A, n=50, dt=0.1):
    """Stationary target tracked for n frames at 1/dt Hz: steady-state covariance."""
    R = np.eye(2) * sigma_m ** 2
    x, P = init_state(np.asarray(xy, float), R, 2.0)
    for _ in range(n):
        x, P = kf_predict(x, P, dt, sigma_a)
        x, P, _ = kf_update(x, P, np.asarray(xy, float), R)
    return Track(track_id=7, time_s=0.0, state=x, covariance=P, radius_m=0.25,
                 confirmed=True, missed_updates=0, hits=n)


@pytest.mark.parametrize("x_obs", [0.6, 2.0, 4.0])  # tau* = 2 s, and clipped to horizon
def test_parked_obstacle_beside_path_no_risk_with_cap(x_obs):
    robot = _robot(v=(0.3, 0.0))
    tr = _steady_state_track((x_obs, 1.4))
    r = assess_track_risk(robot, tr, R_ROBOT, CAPPED, SIGMA_A)
    assert not r.risk
    assert r.margin_m <= CAPPED["nominal_margin_m"] + CAPPED["max_uncertainty_margin_m"] + 1e-12
    # uncapped spec formula would flag it (the integration failure this cap fixes)
    if x_obs >= 2.0:
        assert assess_track_risk(robot, tr, R_ROBOT, {**CAPPED, "max_uncertainty_margin_m": np.inf},
                                 SIGMA_A).risk


def test_margin_never_exceeds_cap():
    cap_total = CAPPED["nominal_margin_m"] + CAPPED["max_uncertainty_margin_m"]
    robot = _robot(v=(0.3, 0.0))
    for pos_s, vel_s in [(0.01, 0.01), (0.1, 0.2), (0.5, 1.0), (3.0, 5.0)]:
        for xy, v in [((1.0, 0.5), (-1.0, 0.0)), ((3.0, 0.0), (-1.2, 0.0)), ((2.0, -2.0), (0.0, 1.0))]:
            r = assess_track_risk(robot, _track(xy, v, pos_s, vel_s), R_ROBOT, CAPPED, SIGMA_A)
            assert CAPPED["nominal_margin_m"] <= r.margin_m <= cap_total + 1e-12
    # head-on still risky with the cap
    assert assess_track_risk(robot, _track((3.0, 0.0), (-1.0, 0.0)), R_ROBOT, CAPPED, SIGMA_A).risk
