import numpy as np
import pytest

from humanoid_avoidance.tracking.multi_tracker import MultiTracker
from humanoid_avoidance.types import Detection

CFG = {
    "sigma_accel_mps2": 0.7,
    "init_velocity_sigma_mps": 2.0,
    "mahalanobis_gate_sq": 9.21,
    "min_confirmations": 2,
    "confirmation_window": 3,
    "max_misses": 5,
}
SIGMA_M = 0.05


def _det(t, xy, radius=0.25, sigma=SIGMA_M):
    return Detection(time_s=t, xy_world_m=np.asarray(xy, dtype=float), radius_m=radius,
                     covariance_xy=np.eye(2) * sigma ** 2, num_points=5)


def test_two_crossing_targets_keep_ids():
    """X-shaped crossing at t=2 s; detection order shuffled every frame."""
    rng = np.random.default_rng(3)
    tr = MultiTracker(CFG)
    pA, vA = np.array([0.0, 0.0]), np.array([1.0, 0.5])
    pB, vB = np.array([0.0, 2.0]), np.array([1.0, -0.5])
    id_a = id_b = None
    t = 0.0
    for k in range(41):                               # 0 .. 4 s at 10 Hz
        t = k * 0.1
        dets = [_det(t, pA + vA * t + rng.normal(0, SIGMA_M, 2)),
                _det(t, pB + vB * t + rng.normal(0, SIGMA_M, 2))]
        if rng.random() < 0.5:
            dets.reverse()
        tracks = tr.step(t, dets)
        if k == 5:
            assert len(tracks) == 2 and all(tk.confirmed for tk in tracks)
            id_a = min(tracks, key=lambda tk: np.linalg.norm(tk.state[:2] - (pA + vA * t))).track_id
            id_b = min(tracks, key=lambda tk: np.linalg.norm(tk.state[:2] - (pB + vB * t))).track_id
            assert id_a != id_b
    assert len(tracks) == 2
    by_id = {tk.track_id: tk for tk in tracks}
    assert set(by_id) == {id_a, id_b}                  # no new / swapped tracks
    assert np.linalg.norm(by_id[id_a].state[:2] - (pA + vA * t)) < 0.15
    assert np.linalg.norm(by_id[id_b].state[:2] - (pB + vB * t)) < 0.15
    assert np.linalg.norm(by_id[id_a].state[2:] - vA) < 0.3
    assert np.linalg.norm(by_id[id_b].state[2:] - vB) < 0.3


def test_out_of_gate_detection_spawns_new_track():
    tr = MultiTracker(CFG)
    for k in range(10):                               # stationary target at origin
        tracks = tr.step(k * 0.1, [_det(k * 0.1, [0.0, 0.0])])
    assert len(tracks) == 1 and tracks[0].confirmed
    tid = tracks[0].track_id
    tracks = tr.step(1.0, [_det(1.0, [3.0, 3.0])])     # far outside the gate
    assert len(tracks) == 2
    old = next(tk for tk in tracks if tk.track_id == tid)
    new = next(tk for tk in tracks if tk.track_id != tid)
    assert np.linalg.norm(old.state[:2]) < 0.05        # not dragged toward the far detection
    assert old.missed_updates == 1
    assert not new.confirmed and np.allclose(new.state[:2], [3.0, 3.0])


def test_single_track_never_forced_onto_far_detection_with_two_tracks():
    tr = MultiTracker(CFG)
    for k in range(10):
        t = k * 0.1
        tr.step(t, [_det(t, [0.0, 0.0]), _det(t, [0.0, 1.5])])
    # only one detection, near track 2; track 1 must not grab it, track 2 must
    tracks = tr.step(1.0, [_det(1.0, [0.0, 1.52])])
    assert len(tracks) == 2
    t1, t2 = sorted(tracks, key=lambda tk: tk.track_id)
    assert t1.missed_updates == 1 and t2.missed_updates == 0


def test_lifecycle_tentative_confirmed_deleted():
    tr = MultiTracker(CFG)
    tracks = tr.step(0.0, [_det(0.0, [1.0, 1.0], radius=0.2)])
    assert len(tracks) == 1 and not tracks[0].confirmed and tracks[0].hits == 1
    assert tr.confirmed_tracks() == []
    tracks = tr.step(0.1, [_det(0.1, [1.0, 1.0], radius=0.4)])
    assert tracks[0].confirmed and tracks[0].hits == 2
    assert 0.2 < tracks[0].radius_m < 0.4               # EMA of radii
    tid = tracks[0].track_id
    t = 0.1
    for miss in range(1, CFG["max_misses"]):
        t += 0.1
        tracks = tr.step(t, [])
        assert len(tracks) == 1 and tracks[0].missed_updates == miss
    t += 0.1
    assert tr.step(t, []) == []                        # deleted after max_misses
    # a re-appearing object gets a fresh id
    tracks = tr.step(t + 0.1, [_det(t + 0.1, [1.0, 1.0])])
    assert tracks[0].track_id != tid


def test_tentative_dies_without_confirmation():
    tr = MultiTracker(CFG)
    tr.step(0.0, [_det(0.0, [1.0, 1.0])])
    assert len(tr.step(0.1, [])) == 1
    assert tr.step(0.2, []) == []                      # 1 hit in a full window of 3 < 2


def test_tentative_confirms_with_gap_inside_window():
    tr = MultiTracker(CFG)
    tr.step(0.0, [_det(0.0, [1.0, 1.0])])
    tr.step(0.1, [])
    tracks = tr.step(0.2, [_det(0.2, [1.0, 1.0])])
    assert len(tracks) == 1 and tracks[0].confirmed


def test_time_backwards_raises_and_same_time_is_noop():
    tr = MultiTracker(CFG)
    tr.step(0.0, [_det(0.0, [0.0, 0.0])])
    tracks1 = tr.step(0.1, [_det(0.1, [0.05, 0.0])])
    tracks2 = tr.step(0.1, [])
    assert tracks2[0].missed_updates == 0
    assert np.array_equal(tracks1[0].state, tracks2[0].state)
    assert np.array_equal(tracks1[0].covariance, tracks2[0].covariance)
    with pytest.raises(ValueError):
        tr.step(0.05, [])
    # a miss at a timestamp is counted only once
    tr.step(0.2, [])
    assert tr.step(0.2, [])[0].missed_updates == 1


def test_variable_dt_and_returns_copies():
    tr = MultiTracker(CFG)
    rng = np.random.default_rng(5)
    v = np.array([0.5, 0.0])
    t = 0.0
    for _ in range(40):
        t += rng.uniform(0.05, 0.25)                   # irregular scan timing
        tracks = tr.step(t, [_det(t, v * t + rng.normal(0, SIGMA_M, 2))])
    assert len(tracks) == 1 and tracks[0].time_s == t
    assert np.linalg.norm(tracks[0].state[2:] - v) < 0.3
    tracks[0].state[:] = 99.0                          # mutate the copy
    assert np.all(tr.tracks()[0].state < 50.0)
    tr.reset()
    assert tr.tracks() == []
