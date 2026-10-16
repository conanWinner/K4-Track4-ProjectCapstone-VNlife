import numpy as np
import pytest

from humanoid_avoidance.geometry.transforms import (
    body_to_world_vel, polar_to_points, pose3, quat_wxyz_to_rotmat, quat_wxyz_to_yaw, rot2d, se2,
    se2_inverse, transform_points_2d, transform_points_3d, world_to_body_vel, wrap_angle,
    yaw_from_rotmat,
)


def _quat_z(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def test_wrap_angle_range():
    a = np.linspace(-20, 20, 2001)
    w = wrap_angle(a)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    assert np.allclose(np.cos(w), np.cos(a)) and np.allclose(np.sin(w), np.sin(a))
    assert wrap_angle(np.pi) == pytest.approx(-np.pi)
    assert isinstance(wrap_angle(0.5), float)


def test_se2_inverse_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = se2(*rng.uniform(-5, 5, 2), rng.uniform(-np.pi, np.pi))
        p = rng.uniform(-3, 3, (10, 2))
        back = transform_points_2d(se2_inverse(T), transform_points_2d(T, p))
        assert np.allclose(back, p)
        assert np.allclose(se2_inverse(T) @ T, np.eye(3))


@pytest.mark.parametrize("yaw,expected", [
    (np.pi / 2, [0.0, 1.0]), (-np.pi / 2, [0.0, -1.0]), (np.pi, [-1.0, 0.0])])
def test_rotation_known_yaws(yaw, expected):
    assert np.allclose(rot2d(yaw) @ [1.0, 0.0], expected, atol=1e-12)
    assert np.allclose(transform_points_2d(se2(1.0, 2.0, yaw), np.array([1.0, 0.0])),
                       np.array(expected) + [1.0, 2.0], atol=1e-12)


def test_sensor_offset_composition_2d_and_3d():
    # robot at (2, 1) heading +90 deg, sensor mounted 0.1 m forward; hit 1 m ahead of the sensor
    T_wb = se2(2.0, 1.0, np.pi / 2)
    T_bs = se2(0.1, 0.0, 0.0)
    p_s = np.array([[1.0, 0.0]])
    p_w = transform_points_2d(T_wb @ T_bs, p_s)
    assert np.allclose(p_w, [[2.0, 2.1]])
    # same in 3D with a sensor mounted 1 m up
    R = quat_wxyz_to_rotmat(_quat_z(np.pi / 2))
    T3_wb = pose3(np.array([2.0, 1.0, 0.0]), R)
    T3_bs = pose3(np.array([0.1, 0.0, 1.0]), np.eye(3))
    p3 = transform_points_3d(T3_wb @ T3_bs, np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(p3, [[2.0, 2.1, 1.0]])


def test_body_world_velocity_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = rng.normal(size=2)
        yaw = rng.uniform(-4, 4)
        assert np.allclose(world_to_body_vel(body_to_world_vel(v, yaw), yaw), v)
    # forward motion while facing +y is +y in world
    assert np.allclose(body_to_world_vel([1.0, 0.0], np.pi / 2), [0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2, -np.pi / 2, 0.3, -2.5, np.pi - 1e-9])
def test_quat_to_yaw_known(yaw):
    assert wrap_angle(quat_wxyz_to_yaw(_quat_z(yaw)) - yaw) == pytest.approx(0.0, abs=1e-9)


def test_quat_rotmat_properties():
    assert np.allclose(quat_wxyz_to_rotmat([1, 0, 0, 0]), np.eye(3))
    # non-normalised input is normalised
    R = quat_wxyz_to_rotmat(2 * _quat_z(0.7))
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    assert yaw_from_rotmat(R) == pytest.approx(0.7)
    # 180 deg about z
    assert quat_wxyz_to_yaw([0, 0, 0, 1]) == pytest.approx(-np.pi)
    # pitch-only rotation has zero yaw
    pitch = np.array([np.cos(0.2), 0.0, np.sin(0.2), 0.0])
    assert quat_wxyz_to_yaw(pitch) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        quat_wxyz_to_rotmat([0, 0, 0, 0])


def test_polar_to_points():
    pts = polar_to_points(np.array([1.0, 2.0, 3.0]), np.array([0.0, np.pi / 2, -np.pi / 2]))
    assert np.allclose(pts, [[1, 0], [0, 2], [0, -3]], atol=1e-12)
    with pytest.raises(ValueError):
        polar_to_points(np.ones(3), np.ones(2))
