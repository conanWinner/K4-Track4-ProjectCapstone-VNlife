"""NavController FSM tests (no simulator). Predictive risk is injected through `risk_fn`."""
from __future__ import annotations

import math

import numpy as np
import pytest

from humanoid_avoidance.config import load_config
from humanoid_avoidance.navigation.controller import NavController
from humanoid_avoidance.types import NO_RISK, Detection, RiskAssessment, RobotState, Track

DT = 0.1
GOAL = np.array([5.0, 0.0])


@pytest.fixture()
def cfg():
    return load_config()


def robot(x=0.0, y=0.0, yaw=0.0, t=0.0):
    return RobotState(time_s=t, xy_world_m=np.array([x, y]), yaw_rad=yaw,
                      v_world_mps=np.zeros(2), yaw_rate_rps=0.0)


def det(x, y, t=0.0):
    return Detection(time_s=t, xy_world_m=np.array([x, y]), radius_m=0.25,
                     covariance_xy=np.eye(2) * 0.01, num_points=5)


def track(x, y, vx, vy, confirmed=True, tid=7):
    return Track(track_id=tid, time_s=0.0, state=np.array([x, y, vx, vy]), covariance=np.eye(4) * 0.05,
                 radius_m=0.25, confirmed=confirmed, missed_updates=0, hits=5)


def risky(tca=1.0, tid=7):
    return RiskAssessment(track_id=tid, tca_s=tca, dca_m=0.2, required_clearance_m=0.95,
                          margin_m=0.35, approaching=True, risk=True, score=0.8)


def times(t_end, t0=0.0):
    n = int(round((t_end - t0) / DT))
    return [round(t0 + k * DT, 10) for k in range(n + 1)]


def run(nav, ts, tracks=(), dets=(), rs=None):
    """Step at each time (fresh scan each tick); returns list of (t, cmd, debug)."""
    out = []
    for t in ts:
        cmd = nav.step(t, rs or robot(t=t), GOAL, list(tracks), list(dets), t)
        out.append((t, cmd, dict(nav.debug)))
    return out


# ------------------------------------------------------------------ sensor / goal
def test_stale_scan_stops(cfg):
    nav = NavController(cfg, "predictive", risk_fn=lambda *a: NO_RISK)
    for t in times(1.0):
        cmd = nav.step(t, robot(t=t), GOAL, [], [], t)
    assert cmd.mode == "GO" and cmd.vx_mps > 0
    timeout = cfg["navigation"]["stale_scan_timeout_s"]
    t_stale = 1.0 + timeout + 0.05
    cmd = nav.step(t_stale, robot(t=t_stale), GOAL, [], [], 1.0)
    assert cmd.mode == "STOP_SENSOR_FAILURE"
    assert "scan age" in nav.debug["reason"]
    # keeps braking to zero (slew limited), never accelerates
    v_prev = cmd.vx_mps
    for k in range(1, 30):
        t = t_stale + k * DT
        cmd = nav.step(t, robot(t=t), GOAL, [], [], 1.0)
        assert cmd.mode == "STOP_SENSOR_FAILURE"
        assert cmd.vx_mps <= v_prev + 1e-12
        v_prev = cmd.vx_mps
    assert cmd.vx_mps == 0.0 and cmd.yaw_rate_rps == 0.0


def test_no_scan_ever_stops_after_grace(cfg):
    nav = NavController(cfg, "reactive")
    cmd = nav.step(0.0, robot(), GOAL, [], [], None)
    assert cmd.mode == "STOP_SENSOR_FAILURE" and cmd.vx_mps == 0.0
    cmd = nav.step(1.5, robot(t=1.5), GOAL, [], [], None)
    assert cmd.mode == "STOP_SENSOR_FAILURE"
    assert "no scan" in nav.debug["reason"]


def test_sensor_recovery_requires_hysteresis(cfg):
    nav = NavController(cfg, "reactive")
    nav.step(0.0, robot(), GOAL, [], [], 0.0)
    nav.step(1.0, robot(t=1.0), GOAL, [], [], 0.0)          # stale
    assert nav.state == "STOP_SENSOR_FAILURE"
    hold = cfg["navigation"]["safe_resume_duration_s"]
    states = [c.mode for _, c, _ in run(nav, times(1.0 + hold + 0.2, 1.1))]
    assert states[0] == "STOP_SENSOR_FAILURE"
    assert "RESUME" in states
    i = states.index("RESUME")
    assert all(s == "STOP_SENSOR_FAILURE" for s in states[:i])


def test_goal_reached_zero_command(cfg):
    nav = NavController(cfg, "predictive", risk_fn=lambda *a: NO_RISK)
    cmd = nav.step(0.0, robot(4.8, 0.1), GOAL, [], [], 0.0)
    assert cmd.mode == "GOAL_REACHED"
    assert (cmd.vx_mps, cmd.vy_mps, cmd.yaw_rate_rps) == (0.0, 0.0, 0.0)


def test_goal_reached_after_driving_brakes_to_zero_and_latches(cfg):
    nav = NavController(cfg, "reactive")
    run(nav, times(2.0))
    assert nav.state == "GO" and nav._vx > 0
    tol = cfg["navigation"]["goal_tolerance_m"]
    x_goal = GOAL[0] - 0.5 * tol
    acc = cfg["navigation"]["accel_limit_mps2"]
    cmd = None
    for t in times(4.0, 2.1):
        cmd = nav.step(t, robot(x_goal, t=t), GOAL, [], [], t)
        assert cmd.mode == "GOAL_REACHED"
    assert cmd.vx_mps == 0.0 and cmd.yaw_rate_rps == 0.0
    # latched even if the robot drifts slightly outside the tolerance
    cmd = nav.step(4.1, robot(GOAL[0] - 1.5 * tol, t=4.1), GOAL, [], [], 4.1)
    assert cmd.mode == "GOAL_REACHED" and cmd.vx_mps == 0.0
    assert acc > 0


# ------------------------------------------------------------------ reactive baseline
def test_reactive_yields_for_obstacle_ahead_in_corridor(cfg):
    stop = cfg["navigation"]["reactive"]["stop_distance_m"]
    nav = NavController(cfg, "reactive")
    res = run(nav, times(0.5), dets=[det(0.7 * stop, 0.1)])
    assert all(c.mode == "YIELD" for _, c, _ in res)
    assert res[-1][2]["risk"] is True
    assert all(c.vx_mps == 0.0 for _, c, _ in res)


@pytest.mark.parametrize("dxy", [(-0.5, 0.0),     # behind
                                 (0.5, 1.2),      # off to the side
                                 (0.5, -1.2),
                                 (2.0, 0.0)])     # ahead but beyond stop distance
def test_reactive_ignores_obstacles_outside_front_box(cfg, dxy):
    nav = NavController(cfg, "reactive")
    res = run(nav, times(0.5), dets=[det(*dxy)])
    assert all(c.mode == "GO" for _, c, _ in res)


def test_reactive_box_is_in_body_frame(cfg):
    nav = NavController(cfg, "reactive")
    # robot facing +y; obstacle 0.5 m along +y is "ahead"; along +x is "to the side"
    rs = robot(0.0, 0.0, yaw=math.pi / 2)
    assert run(nav, [0.0], dets=[det(0.0, 0.5)], rs=rs)[-1][1].mode == "YIELD"
    nav.reset()
    # world (1.2, 0.2) -> body (0.2, -1.2): ahead in world x, but to the right of a +y-facing robot
    assert run(nav, [0.0], dets=[det(1.2, 0.2)], rs=rs)[-1][1].mode == "GO"


def test_reactive_uses_no_tracks_or_prediction(cfg):
    def boom(*a):
        raise AssertionError("reactive mode must not call the risk function")
    nav = NavController(cfg, "reactive", risk_fn=boom)
    # fast track heading straight at the robot but no detection in the box -> still GO
    res = run(nav, times(0.5), tracks=[track(2.0, 0.0, -1.5, 0.0)], dets=[det(2.0, 0.0)])
    assert all(c.mode == "GO" for _, c, _ in res)


# ------------------------------------------------------------------ predictive (stubbed risk)
class ScheduledRisk:
    """risk_fn stub: risky while predicate(time) is True. Records the tracks it receives."""

    def __init__(self, predicate, tca=1.0):
        self.predicate, self.tca, self.calls = predicate, tca, []

    def __call__(self, robot_state, tracks, robot_radius_m, cfg_prediction, sigma_a):
        self.calls.append((robot_state.time_s, [t.track_id for t in tracks], robot_radius_m, sigma_a))
        return risky(self.tca) if self.predicate(robot_state.time_s) else NO_RISK


def test_predictive_yield_hysteresis_resume_ramp(cfg):
    nav_cfg = cfg["navigation"]
    hold, ramp = nav_cfg["safe_resume_duration_s"], nav_cfg["resume_ramp_s"]
    acc, yacc = nav_cfg["accel_limit_mps2"], nav_cfg["yaw_accel_limit_rps2"]
    t_on, t_off = 2.0, 3.0
    stub = ScheduledRisk(lambda t: t_on - 1e-9 <= t < t_off - 1e-9)
    nav = NavController(cfg, "predictive", risk_fn=stub)
    res = run(nav, times(8.0), tracks=[track(3.0, -1.0, 0.0, 0.5)])
    st = {round(t, 6): c.mode for t, c, _ in res}

    assert st[1.9] == "GO"
    assert st[2.0] == "YIELD"
    assert res[20][2]["track_id"] == 7 and res[20][2]["risk"] is True
    # risk cleared at 3.0 but must stay YIELD until risk-free for `hold`
    t_resume = round(t_off + hold, 6)
    for t, s in st.items():
        if t_on <= t < t_resume - 1e-9:
            assert s == "YIELD", (t, s)
    assert st[t_resume] == "RESUME"
    t_go = round(t_resume + ramp, 6)
    for t, s in st.items():
        if t_resume <= t < t_go - 1e-9:
            assert s == "RESUME", (t, s)
    assert st[t_go] == "GO"

    # slew limits respected on every tick (no jumps), vx never negative
    vx = np.array([c.vx_mps for _, c, _ in res])
    wz = np.array([c.yaw_rate_rps for _, c, _ in res])
    assert np.all(np.abs(np.diff(vx)) <= acc * DT + 1e-9)
    assert np.all(np.abs(np.diff(wz)) <= yacc * DT + 1e-9)
    assert vx.min() >= 0.0
    # actually stops while yielding and ramps monotonically back up during RESUME
    i_resume = int(round(t_resume / DT))
    assert vx[i_resume - 1] == 0.0
    i_go = int(round(t_go / DT))
    assert np.all(np.diff(vx[i_resume:i_go + 1]) >= -1e-12)
    assert vx[-1] == pytest.approx(nav_cfg["max_vx_mps"])


def test_predictive_no_oscillation_with_flickering_risk(cfg):
    hold = cfg["navigation"]["safe_resume_duration_s"]
    period = hold / 3.0          # toggles much faster than the hysteresis
    stub = ScheduledRisk(lambda t: 1.0 <= t < 5.0 and int((t - 1.0) / period + 1e-6) % 2 == 0)
    nav = NavController(cfg, "predictive", risk_fn=stub)
    res = run(nav, times(5.0), tracks=[track(3.0, -1.0, 0.0, 0.5)])
    modes = [c.mode for t, c, _ in res if t >= 1.0]
    assert modes[0] == "YIELD"
    assert set(modes) == {"YIELD"}, "flicker faster than safe_resume_duration_s must not release YIELD"
    changes = sum(a != b for a, b in zip([c.mode for _, c, _ in res][:-1], [c.mode for _, c, _ in res][1:]))
    assert changes == 1


def test_predictive_slow_band(cfg):
    nav_cfg = cfg["navigation"]
    tca_mid = 0.5 * (nav_cfg["yield_tca_s"] + nav_cfg["slow_tca_s"])
    stub = ScheduledRisk(lambda t: True, tca=tca_mid)
    nav = NavController(cfg, "predictive", risk_fn=stub)
    res = run(nav, times(3.0), tracks=[track(3.0, -1.0, 0.0, 0.5)])
    assert all(c.mode == "SLOW" for _, c, _ in res)
    assert res[-1][1].vx_mps == pytest.approx(nav_cfg["slow_vx_mps"])


def test_predictive_passes_only_confirmed_tracks_and_cfg_values(cfg):
    stub = ScheduledRisk(lambda t: False)
    nav = NavController(cfg, "predictive", risk_fn=stub)
    tr = [track(3.0, 0.0, -0.5, 0.0, confirmed=True, tid=1),
          track(2.0, 0.0, -0.5, 0.0, confirmed=False, tid=2)]
    run(nav, [0.0], tracks=tr)
    _, ids, r, sigma = stub.calls[0]
    assert ids == [1]
    assert r == cfg["navigation"]["nominal_robot_radius_m"]
    assert sigma == cfg["tracking"]["sigma_accel_mps2"]
    # no confirmed tracks -> risk function not needed, NO_RISK
    stub.calls.clear()
    run(nav, [0.1], tracks=[tr[1]])
    assert nav.state == "GO" and nav.debug["track_id"] == -1


def test_debug_keys_present(cfg):
    nav = NavController(cfg, "predictive", risk_fn=lambda *a: NO_RISK)
    nav.step(0.0, robot(), GOAL, [], [], 0.0)
    for k in ("state", "risk", "tca_s", "dca_m", "score", "margin_m", "reason", "track_id"):
        assert k in nav.debug


# ------------------------------------------------------------------ heading controller
def test_heading_controller_turns_toward_goal(cfg):
    nav_cfg = cfg["navigation"]
    wmax = nav_cfg["max_yaw_rate_rps"]
    for goal_y, sign in ((5.0, +1), (-5.0, -1)):
        nav = NavController(cfg, "reactive")
        cmd = None
        for t in times(2.0):
            cmd = nav.step(t, robot(t=t), np.array([0.0, goal_y]), [], [], t)
        assert np.sign(cmd.yaw_rate_rps) == sign
        assert abs(cmd.yaw_rate_rps) == pytest.approx(wmax)
        assert cmd.vx_mps == pytest.approx(0.0, abs=1e-9)     # 90 deg error -> no forward speed

    # small heading error -> proportional yaw rate, full forward speed
    nav = NavController(cfg, "reactive")
    goal = np.array([5.0, 5.0 * math.tan(0.1)])
    for t in times(3.0):
        cmd = nav.step(t, robot(t=t), goal, [], [], t)
    assert cmd.yaw_rate_rps == pytest.approx(nav_cfg["k_yaw"] * 0.1, rel=1e-6)
    assert cmd.vx_mps == pytest.approx(nav_cfg["max_vx_mps"])


def test_modes_share_controller(cfg):
    """Without obstacles, reactive and predictive must produce identical commands."""
    a = NavController(cfg, "reactive")
    b = NavController(cfg, "predictive", risk_fn=lambda *x: NO_RISK)
    goal = np.array([4.0, 1.5])
    for t in times(3.0):
        rs = robot(0.1 * t, 0.0, 0.05, t=t)
        ca = a.step(t, rs, goal, [], [], t)
        cb = b.step(t, rs, goal, [], [], t)
        assert (ca.vx_mps, ca.yaw_rate_rps, ca.mode) == (cb.vx_mps, cb.yaw_rate_rps, cb.mode)


def test_manual_mode_ignores_obstacles(cfg):
    nav = NavController(cfg, "manual")
    res = run(nav, times(2.0), dets=[det(0.4, 0.0)])
    assert all(c.mode == "MANUAL" for _, c, _ in res)
    assert res[-1][1].vx_mps == pytest.approx(cfg["navigation"]["max_vx_mps"])


def test_time_backwards_raises(cfg):
    nav = NavController(cfg, "reactive")
    nav.step(1.0, robot(t=1.0), GOAL, [], [], 1.0)
    with pytest.raises(ValueError):
        nav.step(0.5, robot(t=0.5), GOAL, [], [], 0.5)
