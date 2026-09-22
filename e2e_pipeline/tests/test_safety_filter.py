"""Safety / feasibility filter (item 2).

Each gate gets a test that isolates it: a candidate that is fine on every axis except
one, and must be rejected for that one reason.  The last few tests cover the
behaviours that matter operationally — that the planner's preference still decides
among equally safe plans, and that total failure produces a brake rather than the
least-bad guess.
"""

import numpy as np
import pytest

from e2e_pipeline.freespace import FREE_CLASS, FreeSpaceExtractor, GridConfig
from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.scene import Agent, EgoState, SceneRepresentation
from e2e_pipeline.uncertainty import RiskModel, TrackCovarianceTracker

ROAD, CAR, SIDEWALK = 11, 4, 13
GRID = GridConfig(x=(-10.0, 50.0, 0.4), y=(-20.0, 20.0, 0.4), z=(-1.0, 5.4, 0.4))


def road_scene(obstacle_xy=None, agents=None, ego_speed=10.0,
               road_half_width=8.0) -> SceneRepresentation:
    """A straight road along +x, optionally with one obstacle block and some agents."""
    sem = np.full(GRID.shape, FREE_CLASS, dtype=np.int64)
    k_ground = GRID.z_band_indices(-0.4, 0.2)[0]
    k_lo, k_hi = GRID.z_band_indices(0.2, 2.2)

    # Lay road down the middle, sidewalk beyond the shoulder.
    ny = GRID.shape[1]
    iy = np.arange(ny)
    y_m = GRID.y[0] + (iy + 0.5) * GRID.y[2]
    on_road = np.abs(y_m) <= road_half_width
    sem[:, on_road, k_ground] = ROAD
    sem[:, ~on_road, k_ground] = SIDEWALK

    if obstacle_xy is not None:
        ix = int((obstacle_xy[0] - GRID.x[0]) / GRID.x[2])
        iy0 = int((obstacle_xy[1] - GRID.y[0]) / GRID.y[2])
        sem[ix - 3:ix + 3, iy0 - 3:iy0 + 3, k_lo:k_hi] = CAR

    fs = FreeSpaceExtractor(GRID)(sem)
    return SceneRepresentation(agents=agents or [], freespace=fs,
                               ego=EgoState(speed=ego_speed))


def straight(speed: float, T: int = 6, dt: float = 0.5, lateral: float = 0.0):
    """A constant-speed plan down the road, optionally offset sideways.

    With `lateral` set this is deliberately a *step* — the ego teleports sideways in
    the first 0.5 s — which the dynamics gate should reject.  Use `lane_change` for
    plans that are meant to be feasible.
    """
    x = np.arange(1, T + 1) * speed * dt
    return np.stack([x, np.full(T, lateral)], axis=1)


def lane_change(speed: float, lateral: float, T: int = 6, dt: float = 0.5):
    """A smooth lateral transition — what a real avoidance manoeuvre looks like.

    Smoothstep has zero slope at both ends, so the plan leaves and rejoins the
    heading without the curvature spike a linear ramp would produce.
    """
    x = np.arange(1, T + 1) * speed * dt
    s = np.linspace(0.0, 1.0, T)
    y = lateral * (3.0 * s ** 2 - 2.0 * s ** 3)
    return np.stack([x, y], axis=1)


# ---------------------------------------------------------------------------
# Gate 1 — drivable area
# ---------------------------------------------------------------------------


def test_accepts_a_clean_plan_on_open_road():
    sf = SafetyFilter()
    result = sf(straight(10.0)[None, ...], road_scene())
    assert not result.emergency
    assert result.chosen_index == 0
    assert result.verdicts[0].feasible


def test_rejects_plan_leaving_the_drivable_surface():
    """Driving onto the sidewalk must fail even though nothing is *in* the way.

    Geometric clearance alone would pass this — the sidewalk is empty space.  Only
    the semantic traversability mask catches it.
    """
    sf = SafetyFilter()
    result = sf(straight(10.0, lateral=12.0)[None, ...], road_scene())
    assert result.emergency
    assert any("off-road" in r for r in result.verdicts[0].reasons)


def test_unknown_space_blocks_by_default_and_can_be_allowed():
    """Unobserved cells are refused unless explicitly permitted."""
    sem = np.full(GRID.shape, FREE_CLASS, dtype=np.int64)
    sem[:, :, GRID.z_band_indices(-0.4, 0.2)[0]] = ROAD
    mask = np.ones(GRID.shape, dtype=bool)
    mask[40:, :, :] = False                       # nothing observed past x ~ 6 m

    fs = FreeSpaceExtractor(GRID)(sem, mask_camera=mask)
    scene = SceneRepresentation(agents=[], freespace=fs, ego=EgoState(speed=10.0))
    cand = straight(10.0)[None, ...]

    assert SafetyFilter()(cand, scene).emergency
    permissive = SafetyFilter(limits=FeasibilityLimits(allow_unknown=True))
    assert not permissive(cand, scene).emergency


# ---------------------------------------------------------------------------
# Gate 2 — collision
# ---------------------------------------------------------------------------


def test_rejects_plan_through_an_unclassified_obstacle():
    """The headline case: an obstacle with no detection box still stops the plan.

    `agents` is empty here, so an object-centric planner has nothing to avoid.  The
    occupancy branch is the only thing standing between the ego and the obstacle.
    """
    scene = road_scene(obstacle_xy=(15.0, 0.0))
    result = SafetyFilter()(straight(10.0)[None, ...], scene)

    assert result.emergency
    assert any("clearance" in r for r in result.verdicts[0].reasons)


def test_lane_change_can_clear_the_same_obstacle():
    """With an obstacle present, a smooth swerve survives where straight-ahead fails.

    Both candidates are dynamically feasible; only the geometry separates them, which
    is what makes this a test of the collision gate rather than of the dynamics gate.
    """
    # The obstacle sits far enough ahead (22 m) that a smoothstep swerve has room to
    # complete its lateral shift before arriving — a 4 m manoeuvre against an
    # obstacle at 15 m is simply too late, and the filter is right to refuse it.
    scene = road_scene(obstacle_xy=(22.0, 0.0), ego_speed=8.0)
    candidates = np.stack([straight(8.0), lane_change(8.0, lateral=5.0)])

    result = SafetyFilter()(candidates, scene)
    assert not result.emergency, result.report()
    assert result.chosen_index == 1


# ---------------------------------------------------------------------------
# Gate 3 — dynamics
# ---------------------------------------------------------------------------


def test_rejects_physically_unreachable_curvature():
    """A plan tighter than the steering clamp cannot be executed by the controller."""
    T = 6
    t = np.arange(1, T + 1) * 0.5
    hairpin = np.stack([2.0 * np.sin(t * 2.0), 2.0 * (1 - np.cos(t * 2.0))], axis=1)

    result = SafetyFilter()(hairpin[None, ...], road_scene(ego_speed=2.0))
    assert any("kappa" in r for r in result.verdicts[0].reasons)


def test_rejects_impossible_acceleration():
    """Jumping from 2 m/s to 30 m/s inside one 0.5 s step is not a plan."""
    scene = road_scene(ego_speed=2.0)
    result = SafetyFilter()(straight(30.0)[None, ...], scene)
    assert any("accel" in r for r in result.verdicts[0].reasons)


def test_max_curvature_matches_the_bicycle_model():
    """The limit must be derived from wheelbase and steering clamp, not guessed.

    Keeping this pinned means a change to simulator/kbm.py that desyncs the filter
    shows up here rather than as an untrackable controller error.
    """
    lim = FeasibilityLimits(wheelbase=2.85, max_steer_rad=0.6)
    assert lim.max_curvature == pytest.approx(np.tan(0.6) / 2.85)


# ---------------------------------------------------------------------------
# Gate 4 — risk
# ---------------------------------------------------------------------------


def test_rejects_plan_with_excessive_collision_risk():
    """An uncertain agent on the path vetoes the plan even with geometry clear.

    The occupancy grid here is empty — the agent is a *prediction*, not an observed
    obstacle.  Only the probabilistic gate can see it.
    """
    agent = Agent(track_id=1, xy=np.array([15.0, 0.0]), yaw=0.0,
                  lwh=np.array([4.5, 1.9, 1.6]), vxy=np.zeros(2),
                  score=0.9, label=0)
    scene = road_scene(agents=[agent])

    tracker = TrackCovarianceTracker()
    tracker.update(scene.agents, timestamp=0.0)
    rm = RiskModel(ego=scene.ego, tracker=tracker)

    result = SafetyFilter()(straight(10.0)[None, ...], scene, risk_model=rm)
    assert result.emergency
    assert any("risk" in r for r in result.verdicts[0].reasons)


def test_risk_gate_is_optional():
    """Omitting the risk model must run geometry-only, not crash or reject."""
    agent = Agent(track_id=1, xy=np.array([15.0, 0.0]), yaw=0.0,
                  lwh=np.array([4.5, 1.9, 1.6]), vxy=np.zeros(2), score=0.9, label=0)
    result = SafetyFilter()(straight(10.0)[None, ...], road_scene(agents=[agent]))
    assert not result.emergency
    assert result.verdicts[0].risk is None


# ---------------------------------------------------------------------------
# Ranking and fallback
# ---------------------------------------------------------------------------


def test_planner_preference_breaks_ties_among_safe_plans():
    """Among equally safe candidates, DiffusionDrive's own ranking should win.

    The filter is a veto, not a replacement policy — it should not quietly substitute
    "hug the widest gap" for the learned preference.
    """
    scene = road_scene(ego_speed=8.0)
    candidates = np.stack([straight(8.0),
                           lane_change(8.0, lateral=1.5),
                           lane_change(8.0, lateral=-1.5)])

    result = SafetyFilter()(candidates, scene, planner_scores=np.array([0.1, 0.8, 0.1]))
    assert result.feasible_count == 3, result.report()
    assert result.chosen_index == 1


def test_emergency_brake_when_every_candidate_fails():
    """Total failure must brake explicitly rather than return the least-bad plan."""
    scene = road_scene(obstacle_xy=(12.0, 0.0))
    candidates = np.stack([straight(10.0), straight(12.0), straight(8.0)])

    result = SafetyFilter()(candidates, scene)
    assert result.emergency
    assert result.chosen_index is None
    assert result.feasible_count == 0
    # The brake profile must decelerate and stay on the current heading.
    traj = result.trajectory
    assert traj.shape == (6, 2)
    assert np.allclose(traj[:, 1], 0.0)
    steps = np.diff(np.concatenate([[0.0], traj[:, 0]]))
    assert np.all(np.diff(steps) <= 1e-9), "brake profile is not monotonically slowing"


def test_verdicts_record_every_violated_gate():
    """Diagnosis needs all failures, not just the first one hit."""
    scene = road_scene(obstacle_xy=(15.0, 0.0))
    result = SafetyFilter()(straight(30.0, lateral=12.0)[None, ...], scene)
    reasons = " ".join(result.verdicts[0].reasons)
    assert "off-road" in reasons and "accel" in reasons


def test_rejects_malformed_candidate_shape():
    with pytest.raises(ValueError, match=r"\(K, T, 2\)"):
        SafetyFilter()(np.zeros((6, 2)), road_scene())


def test_rejects_score_length_mismatch():
    with pytest.raises(ValueError, match="planner_scores"):
        SafetyFilter()(straight(10.0)[None, ...], road_scene(),
                       planner_scores=np.array([0.5, 0.5]))
