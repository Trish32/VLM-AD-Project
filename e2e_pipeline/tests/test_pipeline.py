"""End-to-end orchestration with stub adapters.

The four real networks are separately-trained ports costing ~2-3 s/frame between
them, so the wiring is exercised here against stubs.  That is the point of the
Protocol boundary: the integration logic under test is identical whether the boxes
come from Sparse4D or from a fixture.
"""

import numpy as np
import pytest

from e2e_pipeline.freespace import FREE_CLASS, FreeSpaceExtractor, GridConfig
from e2e_pipeline.pipeline import (LIDAR_TO_EGO_YAW, E2EPipeline,
                                   lidar_boxes_to_agents, rotate_2d,
                                   to_controller_waypoints)
from e2e_pipeline.scene import EgoState

ROAD, CAR = 11, 4
GRID = GridConfig(x=(-10.0, 50.0, 0.4), y=(-20.0, 20.0, 0.4), z=(-1.0, 5.4, 0.4))


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------


def test_lidar_forward_maps_to_ego_forward():
    """nuScenes LiDAR +y is forward; the ego frame's forward is +x.

    This is the transform that silently ruins every downstream clearance query when
    it is wrong — a 90 deg BEV rotation raises nothing at all.
    """
    lidar_forward = np.array([[0.0, 10.0]])          # 10 m ahead in LiDAR
    ego = rotate_2d(lidar_forward, LIDAR_TO_EGO_YAW)[0]
    assert ego[0] == pytest.approx(10.0)             # ...is 10 m ahead in ego
    assert ego[1] == pytest.approx(0.0, abs=1e-9)


def test_lidar_right_maps_to_ego_negative_y():
    """LiDAR +x is right, which is the ego frame's -y (ego +y is left)."""
    ego = rotate_2d(np.array([[10.0, 0.0]]), LIDAR_TO_EGO_YAW)[0]
    assert ego[0] == pytest.approx(0.0, abs=1e-9)
    assert ego[1] == pytest.approx(-10.0)


def test_boxes_convert_position_velocity_and_yaw_together():
    """Rotating position but forgetting velocity is the Sparse4D temporal bug (BUG 4)."""
    # [x, y, z, w, l, h, yaw, vx, vy] — 10 m ahead in LiDAR, moving forward at 5 m/s.
    boxes = np.array([[0.0, 10.0, 0.0, 1.9, 4.5, 1.6, 0.0, 0.0, 5.0]])
    agents = lidar_boxes_to_agents(boxes, [7], [0.9], [0])

    a = agents[0]
    assert a.xy[0] == pytest.approx(10.0)
    assert a.vxy[0] == pytest.approx(5.0)            # velocity rotated too
    assert a.vxy[1] == pytest.approx(0.0, abs=1e-9)
    assert a.yaw == pytest.approx(LIDAR_TO_EGO_YAW)
    assert a.track_id == 7
    assert a.lwh[0] == pytest.approx(4.5)            # length, not width


def test_rejects_malformed_box_array():
    with pytest.raises(ValueError, match=r"\(N, >=9\)"):
        lidar_boxes_to_agents(np.zeros((3, 7)), [1, 2, 3], [1, 1, 1], [0, 0, 0])


# ---------------------------------------------------------------------------
# Stub adapters
# ---------------------------------------------------------------------------


EGO_SPEED = 6.0


class StubDetector:
    """One stationary vehicle 40 m ahead, in the LiDAR frame (+y forward).

    Far enough that it does not drive the risk gate — the override test below is
    deliberately decided by geometry, which is deterministic, rather than by where
    exactly a one-frame-old track's covariance lands.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, images, meta):
        self.calls += 1
        boxes = np.array([[0.0, 40.0, 0.0, 1.9, 4.5, 1.6, 0.0, 0.0, 0.0]])
        return boxes, [1], [0.9], [0]


class StubOccupancy:
    """A straight road, optionally with an unlabelled obstacle at x = 18 m.

    The obstacle has no detection box — it exists only in the occupancy branch, which
    is precisely the case an object-centric planner cannot see.
    """

    def __init__(self, obstacle_x: float | None = None):
        self.calls = 0
        self.obstacle_x = obstacle_x

    def __call__(self, images, meta):
        self.calls += 1
        sem = np.full(GRID.shape, FREE_CLASS, dtype=np.int64)
        sem[:, :, GRID.z_band_indices(-0.4, 0.2)[0]] = ROAD
        if self.obstacle_x is not None:
            ix = int((self.obstacle_x - GRID.x[0]) / GRID.x[2])
            iy = int((0.0 - GRID.y[0]) / GRID.y[2])
            k_lo, k_hi = GRID.z_band_indices(0.2, 2.2)
            sem[ix - 3:ix + 3, iy - 3:iy + 3, k_lo:k_hi] = CAR
        return sem, None


class StubPlanner:
    """Two straight candidates, both dynamically reachable from EGO_SPEED.

    The planner prefers the faster one (0.7 vs 0.3) — so if the filter ever returns
    the slow plan, it did so over the planner's objection.
    """

    def __call__(self, scene, command, meta):
        T, dt = 6, 0.5
        slow = np.stack([np.arange(1, T + 1) * 4.0 * dt, np.zeros(T)], axis=1)
        fast = np.stack([np.arange(1, T + 1) * 7.0 * dt, np.zeros(T)], axis=1)
        return np.stack([slow, fast]), np.array([0.3, 0.7])


def build(occupancy_every=1, obstacle_x=None):
    return E2EPipeline(
        detector=StubDetector(), occupancy=StubOccupancy(obstacle_x),
        planner=StubPlanner(), extractor=FreeSpaceExtractor(GRID),
        occupancy_every=occupancy_every,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_single_frame_runs_end_to_end():
    pipe = build()
    out = pipe.step(images=np.zeros((6, 3, 8, 8)), meta={"timestamp": 0.0},
                    ego=EgoState(speed=EGO_SPEED), command=1)

    assert out.trajectory.shape == (6, 2)
    assert len(out.scene.agents) == 1
    assert out.scene.agents[0].cov is not None, "tracker never attached a covariance"
    assert out.filter_result.feasible_count == 2, out.filter_result.report()
    assert "SafetyFilter" in out.summary()


def test_clean_scene_keeps_the_planner_favourite():
    """With nothing in the way the filter must be inert, not opinionated."""
    pipe = build()
    out = pipe.step(np.zeros((6, 3, 8, 8)), {"timestamp": 0.0},
                    EgoState(speed=EGO_SPEED))

    assert out.filter_result.chosen_index == 1        # the planner's 0.7 candidate
    assert out.chosen_is_planner_favourite


def test_occupancy_only_obstacle_overrides_the_planner_preference():
    """The headline integration case for item 1.

    The obstacle at x = 18 m has no detection box — `agents` contains only the
    vehicle 40 m out — so nothing in the object-centric branch knows it exists.  The
    planner prefers the fast plan, which drives into it; the filter must fall back to
    the slow plan on the strength of the occupancy branch alone.
    """
    pipe = build(obstacle_x=18.0)
    out = pipe.step(np.zeros((6, 3, 8, 8)), {"timestamp": 0.0},
                    EgoState(speed=EGO_SPEED))

    assert not out.filter_result.emergency, out.filter_result.report()
    assert out.filter_result.chosen_index == 0, out.filter_result.report()
    assert out.chosen_is_planner_favourite is False

    fast = out.filter_result.verdicts[1]
    assert not fast.feasible
    assert any("clearance" in r or "off-road" in r for r in fast.reasons)


def test_detected_agent_lands_ahead_of_the_ego():
    """A vehicle 40 m forward in LiDAR must end up 40 m forward in the scene."""
    pipe = build()
    out = pipe.step(np.zeros((6, 3, 8, 8)), {"timestamp": 0.0}, EgoState(speed=EGO_SPEED))

    a = out.scene.agents[0]
    assert a.xy[0] == pytest.approx(40.0)
    assert a.xy[1] == pytest.approx(0.0, abs=1e-9)


def test_covariance_contracts_over_repeated_frames():
    """Running several frames must tighten the track posterior, not reset it."""
    pipe = build()
    first = pipe.step(np.zeros((6, 3, 8, 8)), {"timestamp": 0.0}, EgoState(speed=EGO_SPEED))
    p0 = first.scene.agents[0].cov[0, 0]

    for i in range(1, 5):
        out = pipe.step(np.zeros((6, 3, 8, 8)), {"timestamp": 0.5 * i},
                        EgoState(speed=EGO_SPEED))
    assert out.scene.agents[0].cov[0, 0] < p0


def test_occupancy_rate_decoupling_reuses_the_cached_grid():
    """The dense branch is the expensive one; `occupancy_every` must actually skip it."""
    pipe = build(occupancy_every=3)
    for i in range(6):
        pipe.step(np.zeros((6, 3, 8, 8)), {"timestamp": 0.5 * i}, EgoState(speed=EGO_SPEED))

    assert pipe.occupancy.calls == 2, "occupancy ran more often than requested"
    assert pipe.detector.calls == 6, "detector must run every frame"


def test_reset_clears_temporal_state():
    """Carrying tracks or a cached grid across a scene boundary is a silent bug."""
    pipe = build()
    pipe.step(np.zeros((6, 3, 8, 8)), {"timestamp": 0.0}, EgoState(speed=EGO_SPEED))
    assert pipe._cached_freespace is not None

    pipe.reset()
    assert pipe._cached_freespace is None
    assert pipe._frame == 0
    assert not pipe.tracker._tracks


def test_controller_handoff_prepends_the_ego_origin():
    """Pure pursuit needs a segment to start from, not just future waypoints."""
    traj = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)
    wp = to_controller_waypoints(traj, EgoState(speed=10.0))

    assert wp.shape == (7, 2)
    assert np.allclose(wp[0], [0.0, 0.0])
    assert np.allclose(wp[1:], traj)


def test_controller_handoff_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(T, 2\)"):
        to_controller_waypoints(np.zeros((6, 3)), EgoState(speed=1.0))
