"""Tests for the demo video's control derivation.

`demo_control` is the only part of the demo pipeline that computes numbers rather than
drawing them, so it is the only part worth unit-testing: everything the video claims
about steering, speed and manoeuvre is a pure function of a planned trajectory, and a
sign error here would silently mislabel every frame.

These import only numpy, so they run anywhere -- no mmcv, no dataset, no checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from demo_control import (WHEELBASE, CanBus, PLAN_DT, as_waypoints, ego_status_fields,
                          maneuver_label, plan_to_control, resample)


# ---------------------------------------------------------------- geometry helpers

def straight_plan(speed=5.0, steps=6):
    """A plan going straight ahead at constant speed. Frame is x=right, y=forward."""
    return np.stack([np.zeros(steps),
                     np.arange(1, steps + 1) * speed * PLAN_DT], axis=1)


def arc_plan(radius, speed=5.0, steps=6, left=True):
    """Constant-curvature arc from the origin, tangent to straight ahead."""
    s = np.arange(1, steps + 1) * speed * PLAN_DT
    theta = s / radius
    x = radius * (1 - np.cos(theta))
    return np.stack([-x if left else x, radius * np.sin(theta)], axis=1)


def test_as_waypoints_prepends_ego_origin():
    plan = straight_plan()
    path = as_waypoints(plan)
    assert path.shape == (7, 2)
    assert np.allclose(path[0], 0.0)
    assert np.allclose(path[1:], plan)


def test_resample_preserves_endpoints_and_spaces_uniformly():
    path = as_waypoints(arc_plan(radius=20.0))
    dense = resample(path, 100)
    assert dense.shape == (100, 2)
    assert np.allclose(dense[0], path[0], atol=1e-9)
    assert np.allclose(dense[-1], path[-1], atol=1e-9)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    assert seg.std() / seg.mean() < 0.02


def test_resample_survives_a_stationary_plan():
    """A stopped ego produces an all-zero plan; resampling it must not divide by zero."""
    dense = resample(np.zeros((7, 2)), 50)
    assert dense.shape == (50, 2)
    assert np.isfinite(dense).all()


# ---------------------------------------------------------------- longitudinal

def test_speed_profile_recovers_the_plan_speed():
    ctrl, _ = plan_to_control(straight_plan(speed=7.0), v_measured=7.0, steer_ratio=15.0)
    assert ctrl.speed == pytest.approx(7.0, abs=1e-6)
    assert np.allclose(ctrl.speed_profile, 7.0)
    assert ctrl.accel == pytest.approx(0.0, abs=1e-6)


def test_accel_is_read_off_the_plan_not_the_measurement():
    """The reported acceleration must describe the plan, so a wrong CAN speed cannot
    change it -- otherwise the orange trace would be a function of the cyan one."""
    plan = np.stack([np.zeros(6), np.cumsum([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])], axis=1)
    a = plan_to_control(plan, v_measured=2.0, steer_ratio=15.0)[0].accel
    b = plan_to_control(plan, v_measured=9.0, steer_ratio=15.0)[0].accel
    assert a == pytest.approx(b)
    assert a > 0


def test_throttle_and_brake_are_mutually_exclusive():
    fast = plan_to_control(straight_plan(speed=10.0), v_measured=2.0, steer_ratio=15.0)[0]
    slow = plan_to_control(straight_plan(speed=1.0), v_measured=9.0, steer_ratio=15.0)[0]
    assert fast.throttle > 0 and fast.brake == 0
    assert slow.brake > 0 and slow.throttle == 0
    for c in (fast, slow):
        assert 0.0 <= c.throttle <= 1.0 and 0.0 <= c.brake <= 1.0


# ---------------------------------------------------------------- lateral

@pytest.mark.parametrize("radius", [12.0, 25.0, 60.0, 100.0])
def test_pure_pursuit_recovers_the_true_curvature(radius):
    """On a constant-curvature arc the pure-pursuit formula is exact *for a point on the
    arc*, so this pins the curvature, yaw-rate and road-wheel formulas together.

    The recovered value sits a few percent high, consistently, and that is not an error
    to tune away: a plan is 6 waypoints 0.5 s apart, and interpolating between them
    linearly puts the lookahead point on a chord, which for a convex path lies further
    from the centreline than the arc does. A real controller reading these waypoints
    sees exactly the same bias. Assert its size and its sign rather than hiding it.
    """
    ctrl, _ = plan_to_control(arc_plan(radius, speed=6.0), v_measured=6.0,
                              steer_ratio=15.0)
    true = 1.0 / radius
    assert ctrl.curvature == pytest.approx(true, rel=0.06)
    assert ctrl.curvature > true                      # chords cut outside the arc
    assert ctrl.road_wheel == pytest.approx(np.arctan(WHEELBASE * ctrl.curvature))
    assert ctrl.yaw_rate == pytest.approx(6.0 * ctrl.curvature)


def test_turn_direction_signs_are_left_positive():
    """Left-positive matches the sign convention of the CAN yaw-rate channel, which the
    video plots on the same axes as the planned yaw rate."""
    left, _ = plan_to_control(arc_plan(20.0, left=True), 5.0, 15.0)
    right, _ = plan_to_control(arc_plan(20.0, left=False), 5.0, 15.0)
    assert left.curvature > 0 and left.yaw_rate > 0 and left.steer_wheel_deg > 0
    assert right.curvature < 0 and right.yaw_rate < 0 and right.steer_wheel_deg < 0
    assert left.curvature == pytest.approx(-right.curvature)


def test_straight_plan_commands_no_steering():
    ctrl, _ = plan_to_control(straight_plan(), v_measured=5.0, steer_ratio=15.0)
    assert ctrl.curvature == pytest.approx(0.0, abs=1e-9)
    assert ctrl.steer_wheel_deg == pytest.approx(0.0, abs=1e-9)


def test_steer_ratio_scales_the_wheel_angle_only():
    a, _ = plan_to_control(arc_plan(18.0), 5.0, steer_ratio=10.0)
    b, _ = plan_to_control(arc_plan(18.0), 5.0, steer_ratio=20.0)
    assert b.steer_wheel_deg == pytest.approx(2.0 * a.steer_wheel_deg)
    assert b.road_wheel == pytest.approx(a.road_wheel)


def test_lookahead_grows_with_speed_and_stays_bounded():
    slow = plan_to_control(straight_plan(speed=1.0), 1.0, 15.0)[0]
    fast = plan_to_control(straight_plan(speed=9.0), 9.0, 15.0)[0]
    assert slow.lookahead_dist == pytest.approx(3.0, abs=0.4)   # floored, not zero
    assert fast.lookahead_dist > slow.lookahead_dist


# ---------------------------------------------------------------- action labels

def test_maneuver_labels_name_the_turn_direction():
    assert maneuver_label(arc_plan(12.0, left=True), 5.0)[0] == "TURN LEFT"
    assert maneuver_label(arc_plan(12.0, left=False), 5.0)[0] == "TURN RIGHT"
    assert maneuver_label(straight_plan(), 5.0)[0] == "LANE KEEP"


def test_maneuver_labels_name_the_speed_change():
    accel = np.stack([np.zeros(6), np.cumsum([1.0, 1.4, 1.8, 2.2, 2.6, 3.0])], axis=1)
    decel = np.stack([np.zeros(6), np.cumsum([3.0, 2.6, 2.2, 1.8, 1.4, 1.0])], axis=1)
    assert maneuver_label(accel, 3.0)[1] == "ACCELERATE"
    assert maneuver_label(decel, 6.0)[1] == "BRAKE"
    assert maneuver_label(straight_plan(speed=5.0), 5.0)[1] == "MAINTAIN SPEED"
    assert maneuver_label(np.zeros((6, 2)), 0.0)[1] == "HOLD / STOP"


# ---------------------------------------------------------------- CAN bus

@pytest.fixture
def can_scene(tmp_path):
    """A synthetic scene whose steering ratio and channel values are known exactly."""
    ratio = 14.0
    v_kph = 36.0                       # 10 m/s
    # Enough samples clear the |wheel| > 20 deg gate that `steer_ratio` actually fits,
    # rather than bailing to its fallback -- the gate wants >= 5 informative samples.
    deltas_deg = np.array([-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0])
    yaw = 10.0 * np.tan(np.radians(deltas_deg)) / WHEELBASE      # rad/s
    utimes = (1_000_000 + np.arange(len(deltas_deg)) * 500_000).astype(int)
    rows = [dict(utime=int(t), steering=float(ratio * d), vehicle_speed=v_kph,
                 throttle=50.0, brake=0.0, yaw_rate=float(np.degrees(y)))
            for t, d, y in zip(utimes, deltas_deg, yaw)]
    (tmp_path / "demo_vehicle_monitor.json").write_text(json.dumps(rows))
    (tmp_path / "demo_steeranglefeedback.json").write_text(json.dumps(
        [dict(utime=int(t), value=float(np.radians(ratio * d)))
         for t, d in zip(utimes, deltas_deg)]))
    (tmp_path / "demo_ms_imu.json").write_text(json.dumps(
        [dict(utime=int(t), rotation_rate=[0.0, 0.0, float(y)],
              linear_accel=[0.5, 0.0, 9.8]) for t, y in zip(utimes, yaw)]))
    return CanBus(tmp_path, "demo"), ratio, utimes, deltas_deg


def test_can_interpolates_between_samples(can_scene):
    can, ratio, utimes, deltas = can_scene
    mid = float((utimes[0] + utimes[1]) / 2)
    assert can.at("speed_kph", mid) == pytest.approx(36.0)
    expected = ratio * (deltas[0] + deltas[1]) / 2
    assert can.at("steer_wheel_deg", mid) == pytest.approx(expected, rel=1e-6)


def test_can_reads_a_vector_channel_by_component(can_scene):
    can, _, utimes, deltas = can_scene
    # rotation_rate[2] is yaw; at the straight-ahead sample it is zero
    straight = int(np.argmin(np.abs(deltas)))
    assert can.at("yaw_rate_dps", float(utimes[straight])) == pytest.approx(0.0, abs=1e-9)
    assert can.at("accel_long", float(utimes[0])) == pytest.approx(0.5)


def test_can_missing_channel_falls_back_instead_of_raising(tmp_path):
    can = CanBus(tmp_path, "absent")
    assert can.at("speed_kph", 1.0, default=7.5) == 7.5
    assert len(can.series("speed_kph")[0]) == 0


def test_steer_ratio_is_recovered_from_yaw_rate_and_speed(can_scene):
    can, ratio, _, _ = can_scene
    assert can.steer_ratio() == pytest.approx(ratio, rel=0.02)


def test_steer_ratio_falls_back_when_the_scene_is_uninformative(tmp_path):
    """A scene driven straight has no signal to fit, and must not return a wild slope."""
    rows = [dict(utime=1_000_000 + i * 500_000, steering=0.0, vehicle_speed=30.0,
                 throttle=0.0, brake=0.0, yaw_rate=0.0) for i in range(10)]
    (tmp_path / "s_vehicle_monitor.json").write_text(json.dumps(rows))
    assert CanBus(tmp_path, "s").steer_ratio(fallback=15.5) == 15.5


# ---------------------------------------------------------------- ego_status

def test_ego_status_unpacks_to_the_documented_layout():
    e = np.array([0.1, 0.2, 9.8, 0.01, 0.02, -0.3, 8.0, 0.0, 0.0, -1.25])
    f = ego_status_fields(e)
    assert np.allclose(f["accel"], [0.1, 0.2, 9.8])
    assert np.allclose(f["ang_vel"], [0.01, 0.02, -0.3])
    assert f["speed"] == pytest.approx(8.0)
    assert f["steer_rad"] == pytest.approx(-1.25)
