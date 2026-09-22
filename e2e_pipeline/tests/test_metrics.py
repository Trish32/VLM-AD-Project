"""Closed-loop metrics.

The definitions matter more than the plumbing here: a metric that silently means
something other than its name is worse than no metric, so each test pins one
definition against a hand-computable case.
"""

import numpy as np
import pytest

from e2e_pipeline.metrics import (COMFORT_LAT_ACCEL, StepRecord, comfort_metrics,
                                  evaluate, latency_metrics, obb_overlap,
                                  polygon_distance, prediction_metrics,
                                  route_completion, safety_metrics)
from e2e_pipeline.scene import ego_footprint_corners as box


def rec(t, x, y=0.0, yaw=0.0, v=5.0, **kw):
    return StepRecord(t=t, ego_xy=np.array([x, y]), ego_yaw=yaw, ego_v=v,
                      accel=kw.pop('accel', 0.0), steer=kw.pop('steer', 0.0), **kw)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_overlapping_boxes_detected():
    a = box(np.array([0.0, 0.0]), 0.0, 4.6, 1.8)
    b = box(np.array([2.0, 0.0]), 0.0, 4.6, 1.8)
    assert obb_overlap(a, b)


def test_side_by_side_cars_do_not_collide():
    """The case a circumscribed-disc test gets wrong.

    Two 4.6x1.8 m cars 2.2 m apart laterally are clearly not touching, but their
    circumscribed discs (r ~ 2.47 m) overlap. Reporting that as a collision would
    inflate every safety number.
    """
    a = box(np.array([0.0, 0.0]), 0.0, 4.6, 1.8)
    b = box(np.array([0.0, 2.2]), 0.0, 4.6, 1.8)
    assert not obb_overlap(a, b)
    assert polygon_distance(a, b) == pytest.approx(0.4, abs=0.05)


def test_rotation_matters():
    """Orientation alone decides overlap — an axis-aligned test would miss this.

    `a` spans x [-2.3, 2.3], y [-0.9, 0.9]. A same-size box centred at (2, 2)
    lying parallel occupies y [1.1, 2.9] and clears it. Turn that box across the
    path and it sweeps y [-0.3, 4.3] while still covering x [1.1, 2.9] — now it
    overlaps, from the rotation and nothing else.
    """
    a = box(np.array([0.0, 0.0]), 0.0, 4.6, 1.8)
    parallel = box(np.array([2.0, 2.0]), 0.0, 4.6, 1.8)
    crossing = box(np.array([2.0, 2.0]), np.pi / 2, 4.6, 1.8)
    assert not obb_overlap(a, parallel)
    assert obb_overlap(a, crossing)


def test_distance_is_zero_when_overlapping():
    a = box(np.array([0.0, 0.0]), 0.0, 4.6, 1.8)
    assert polygon_distance(a, a) == 0.0


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def test_clear_run_reports_no_collision():
    recs = [rec(i * 0.5, i * 2.5,
                agent_boxes=[((0.0, 20.0), 0.0, 4.6, 1.8, 1)]) for i in range(6)]
    m = safety_metrics(recs)
    assert not m['collision'] and m['n_collision_steps'] == 0
    assert m['min_clearance_m'] > 0


def test_driving_into_a_parked_car_is_a_collision():
    recs = [rec(i * 0.5, i * 4.0,
                agent_boxes=[((12.0, 0.0), 0.0, 4.6, 1.8, 7)]) for i in range(6)]
    m = safety_metrics(recs)
    assert m['collision']
    assert m['min_clearance_m'] == 0.0
    assert m['collisions'][0]['track_id'] == 7


def test_emergency_brakes_are_counted():
    recs = [rec(0.0, 0.0, emergency=True), rec(0.5, 1.0), rec(1.0, 2.0, emergency=True)]
    assert safety_metrics(recs)['emergency_brakes'] == 2


# ---------------------------------------------------------------------------
# Route completion
# ---------------------------------------------------------------------------


def test_completion_is_arc_length_not_waypoint_count():
    route = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    recs = [rec(i * 0.5, i * 5.0) for i in range(11)]      # travels 0 -> 50 m
    r = route_completion(recs, route)
    assert r['route_length_m'] == pytest.approx(100.0)
    assert r['completion'] == pytest.approx(0.5, abs=0.01)


def test_progress_is_the_furthest_point_reached():
    """Ground covered is not surrendered by stopping or backing up."""
    route = np.array([[0.0, 0.0], [100.0, 0.0]])
    recs = [rec(0.0, 0.0), rec(0.5, 60.0), rec(1.0, 40.0)]
    assert route_completion(recs, route)['completion'] == pytest.approx(0.6, abs=0.01)


def test_lateral_deviation_is_tracked_separately():
    """Driving parallel to the route still counts progress, but is flagged."""
    route = np.array([[0.0, 0.0], [100.0, 0.0]])
    recs = [rec(0.0, 0.0, 5.0), rec(0.5, 50.0, 5.0)]
    r = route_completion(recs, route)
    assert r['completion'] == pytest.approx(0.5, abs=0.01)
    assert r['max_lateral_dev_m'] == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------------
# Comfort
# ---------------------------------------------------------------------------


def test_constant_speed_straight_line_is_comfortable():
    recs = [rec(i * 0.5, i * 2.5, v=5.0) for i in range(8)]
    c = comfort_metrics(recs, dt=0.5)
    assert c['long_accel_rms'] == pytest.approx(0.0, abs=1e-9)
    assert c['lat_accel_rms'] == pytest.approx(0.0, abs=1e-9)
    assert c['long_violations'] == 0 and c['lat_violations'] == 0


def test_hard_braking_is_flagged():
    speeds = [10.0, 10.0, 2.0, 2.0]          # -16 m/s^2 across one 0.5 s step
    recs = [rec(i * 0.5, i, v=s) for i, s in enumerate(speeds)]
    c = comfort_metrics(recs, dt=0.5)
    assert c['long_accel_max'] > 10.0
    assert c['long_violations'] >= 1


def test_tight_turn_at_speed_flags_lateral_accel():
    """a_lat = v^2 * kappa, measured from the realised path, not the command."""
    th = np.linspace(0, 1.2, 8)
    R = 8.0
    recs = [rec(i * 0.5, R * np.sin(t), R * (1 - np.cos(t)), v=10.0)
            for i, t in enumerate(th)]
    c = comfort_metrics(recs, dt=0.5)
    assert c['lat_accel_max'] > COMFORT_LAT_ACCEL
    assert c['lat_violations'] >= 1


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_perfect_forecast_scores_zero_error():
    recs = []
    for i in range(5):
        pos = np.array([10.0 + i * 2.0, 0.0])
        future = np.stack([[10.0 + (i + 1 + h) * 2.0, 0.0] for h in range(2)])
        recs.append(rec(i * 0.5, 0.0,
                        agent_boxes=[(pos, 0.0, 4.6, 1.8, 3)],
                        predictions={3: future}))
    p = prediction_metrics(recs, dt=0.5)
    assert p['minADE_m'] == pytest.approx(0.0, abs=1e-6)
    assert p['n_scored'] > 0


def test_biased_forecast_scores_that_bias():
    recs = []
    for i in range(5):
        pos = np.array([10.0, 0.0])
        future = np.stack([[10.0, 3.0]] * 2)      # always 3 m off
        recs.append(rec(i * 0.5, 0.0,
                        agent_boxes=[(pos, 0.0, 4.6, 1.8, 3)],
                        predictions={3: future}))
    assert prediction_metrics(recs, dt=0.5)['minADE_m'] == pytest.approx(3.0, abs=1e-6)


def test_agent_leaving_the_scene_is_not_scored_as_error():
    """Unscoreable is not the same as wrong."""
    recs = [rec(0.0, 0.0, agent_boxes=[((10.0, 0.0), 0.0, 4.6, 1.8, 3)],
                predictions={3: np.array([[12.0, 0.0], [14.0, 0.0]])}),
            rec(0.5, 2.0, agent_boxes=[])]
    p = prediction_metrics(recs, dt=0.5)
    assert p['n_unscoreable'] == 1 and p['n_scored'] == 0


# ---------------------------------------------------------------------------
# Latency + integration
# ---------------------------------------------------------------------------


def test_latency_reports_tail_not_just_mean():
    recs = [rec(i * 0.5, i, latency_ms={'planner': v})
            for i, v in enumerate([10.0] * 19 + [500.0])]
    lat = latency_metrics(recs)
    assert lat['planner']['p50_ms'] == pytest.approx(10.0, abs=0.1)
    assert lat['planner']['max_ms'] == 500.0
    assert lat['total']['hz_at_p50'] == pytest.approx(100.0, abs=1.0)


def test_evaluate_returns_all_five_families():
    route = np.array([[0.0, 0.0], [50.0, 0.0]])
    recs = [rec(i * 0.5, i * 2.5, latency_ms={'planner': 5.0}) for i in range(6)]
    m = evaluate(recs, route, dt=0.5)
    assert set(m) >= {'safety', 'route', 'comfort', 'prediction', 'latency'}
    assert m['n_steps'] == 6
