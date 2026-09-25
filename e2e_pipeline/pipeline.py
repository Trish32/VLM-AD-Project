"""
Orchestration for the modular end-to-end stack.

    multi-camera RGB
      |-- Sparse4D v3   (3D detection + tracking)  --.
      |                                              >-- unified scene repr
      '-- FlashOcc      (3D occupancy -> free space) -'
                                   |
                        QCNet / motion + planning heads
                                   |
                            DiffusionDrive  (K candidate plans)
                                   |
                              safety filter
                                   |
                         controller -> kinematic bicycle

WHY THE MODELS ARE BEHIND PROTOCOLS
-----------------------------------
The four networks in this diagram are separately-trained ports that do not share a
backbone, a checkpoint, or in some cases a runnable configuration (BEVFusion-PP needs
`--device cpu` for its int64 `scatter_reduce`; Sparse4D training needs the MPS
fallback flag).  Serially they cost roughly 2-3 s/frame on an M3 Max, so a live
single-process loop is not the useful artifact here.

What *is* useful — and what was actually missing — is the integration layer: the frame
contracts, the fusion into one scene object, the uncertainty plumbing, and the safety
gate.  Those are pure functions of the models' outputs, so they are defined against
`Protocol`s.  Supply live adapters when you have the environments up; supply cached
per-frame tensors when you are iterating on the planning logic.  The logic under test
is identical either way.

RATE DECOUPLING
---------------
`occupancy_every` runs the FlashOcc branch every N frames and reuses the cached
`FreeSpace` in between.  Dense geometry is dominated by static structure that changes
slowly relative to the agent motion Sparse4D tracks, so this buys most of the latency
back.  The cached grid is *not* ego-motion compensated here — at 2 Hz keyframes and
N=2 the drift is under a metre, but if you raise N, warp it first (the same shift +
rotate trick BEVFormer uses for `prev_bev`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .freespace import FreeSpace, FreeSpaceExtractor, GridConfig
from .scene import Agent, EgoState, SceneRepresentation, TrajectoryDistribution
from .planner.safety_filter import FilterResult, SafetyFilter
from .uncertainty import RiskModel, TrackCovarianceTracker


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

# nuScenes LiDAR_TOP is +x right / +y forward; the ego frame is +x forward / +y left.
# Mapping (0, 1) -> (1, 0) and (1, 0) -> (0, -1) is a rotation by -pi/2, matching the
# "~-90 deg lidar/ego offset" recorded in the bevformer / sparse4d / bevfusion logs.
#
# The exact transform belongs to the sample's calibrated_sensor record — this constant
# is the nominal value, used only as a default and as a sanity anchor in tests.
# Getting it wrong is silent: a 90 deg BEV rotation raises nothing and simply makes
# every clearance query answer about the wrong direction, which is why the sign is
# pinned by test_pipeline.py rather than left to inspection.
LIDAR_TO_EGO_YAW = -np.pi / 2


def rotate_2d(xy: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate (..., 2) points by `yaw` radians CCW."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(xy, dtype=np.float64) @ rot.T


def lidar_boxes_to_agents(boxes: np.ndarray, track_ids: Sequence[int],
                          scores: Sequence[float], labels: Sequence[int],
                          yaw_offset: float = LIDAR_TO_EGO_YAW) -> list[Agent]:
    """Sparse4D LiDAR-frame boxes -> ego-frame `Agent`s.

    Parameters
    ----------
    boxes : (N, 9) as the detector emits them — [x, y, z, w, l, h, yaw, vx, vy].
        Note the width-before-length ordering; slot 3 is LENGTH in the *anchor*
        convention but WIDTH here, a distinction that cost real debugging time in the
        Sparse4D port (bug_log BUG 5).  This function assumes the eval-side ordering
        used by `tools/eval.py`.
    yaw_offset : rotation from LiDAR to ego frame.  Pass the value derived from the
        sample's calibrated_sensor rather than the default when you have it.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] < 9:
        raise ValueError(f"expected (N, >=9) boxes, got {boxes.shape}")

    xy = rotate_2d(boxes[:, 0:2], yaw_offset)
    vxy = rotate_2d(boxes[:, 7:9], yaw_offset)
    yaws = boxes[:, 6] + yaw_offset

    return [
        Agent(track_id=int(track_ids[i]), xy=xy[i], yaw=float(yaws[i]),
              lwh=np.array([boxes[i, 4], boxes[i, 3], boxes[i, 5]]),
              vxy=vxy[i], score=float(scores[i]), label=int(labels[i]))
        for i in range(len(boxes))
    ]


# ---------------------------------------------------------------------------
# Model protocols
# ---------------------------------------------------------------------------


class DetectorTracker(Protocol):
    """Sparse4D v3 branch."""

    def __call__(self, images: np.ndarray, meta: dict) -> tuple[
            np.ndarray, Sequence[int], Sequence[float], Sequence[int]]:
        """-> (boxes (N, 9) lidar frame, track_ids, scores, labels)."""


class OccupancyModel(Protocol):
    """FlashOcc branch."""

    def __call__(self, images: np.ndarray, meta: dict) -> tuple[
            np.ndarray, np.ndarray | None]:
        """-> (semantics (200, 200, 16) int, mask_camera or None)."""


class MotionForecaster(Protocol):
    """QCNet branch."""

    def __call__(self, agents: list[Agent], meta: dict
                 ) -> dict[int, TrajectoryDistribution]:
        """-> {track_id: TrajectoryDistribution}, Laplace loc/scale + mode probs."""


class CandidatePlanner(Protocol):
    """DiffusionDrive branch."""

    def __call__(self, scene: SceneRepresentation, command: int, meta: dict
                 ) -> tuple[np.ndarray, np.ndarray]:
        """-> (candidates (K, T, 2) ego frame, planner_scores (K,))."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineOutput:
    """One frame's result, kept whole so a caller can log or visualise any stage."""

    scene: SceneRepresentation
    candidates: np.ndarray
    filter_result: FilterResult
    command: int
    planner_scores: np.ndarray | None = None

    @property
    def trajectory(self) -> np.ndarray:
        return self.filter_result.trajectory

    @property
    def chosen_is_planner_favourite(self) -> bool:
        """Did the filter keep the planner's own top-ranked candidate?

        Aggregated over a run this is the headline number for "is the safety layer
        earning its place": an override rate of zero means the filter is inert, and a
        very high one means the planner and the constraints disagree systematically.
        Braking counts as an override.
        """
        if self.filter_result.emergency or self.planner_scores is None:
            return False
        return int(np.argmax(self.planner_scores)) == self.filter_result.chosen_index

    def summary(self) -> str:
        return "\n".join([self.scene.summary(), self.filter_result.report()])


class E2EPipeline:
    """Wire the branches together for one frame at a time.

    Parameters
    ----------
    detector / occupancy / forecaster / planner : the four model adapters.
        `forecaster` may be None, in which case every agent falls back to a
        constant-velocity rollout with honestly growing covariance.
    occupancy_every : run the dense branch on every Nth frame (see module docstring).
    horizon : planning steps; 6 at dt=0.5 gives the 3 s horizon the ported heads use.
    """

    def __init__(self,
                 detector: DetectorTracker,
                 occupancy: OccupancyModel,
                 planner: CandidatePlanner,
                 forecaster: MotionForecaster | None = None,
                 extractor: FreeSpaceExtractor | None = None,
                 tracker: TrackCovarianceTracker | None = None,
                 safety: SafetyFilter | None = None,
                 occupancy_every: int = 1,
                 horizon: int = 6,
                 dt: float = 0.5) -> None:
        self.detector = detector
        self.occupancy = occupancy
        self.planner = planner
        self.forecaster = forecaster
        self.extractor = extractor or FreeSpaceExtractor(GridConfig())
        self.tracker = tracker or TrackCovarianceTracker()
        self.safety = safety or SafetyFilter(dt=dt)
        self.occupancy_every = max(1, int(occupancy_every))
        self.horizon = int(horizon)
        self.dt = float(dt)

        self._frame = 0
        self._cached_freespace: FreeSpace | None = None

    # -- per-frame ----------------------------------------------------------

    def step(self, images: np.ndarray, meta: dict, ego: EgoState,
             command: int = 1) -> PipelineOutput:
        """Run one keyframe end to end.

        `command` indexes DiffusionDrive's anchor set (0=right, 1=straight, 2=left).
        It is a *routing* input and the safety filter never overrides it — see the
        note in safety_filter.py about why re-ranking is scoped to within one command.
        """
        timestamp = float(meta.get("timestamp", self._frame * self.dt))

        # --- object branch -------------------------------------------------
        boxes, track_ids, scores, labels = self.detector(images, meta)
        agents = lidar_boxes_to_agents(
            boxes, track_ids, scores, labels,
            yaw_offset=float(meta.get("lidar_to_ego_yaw", LIDAR_TO_EGO_YAW)))
        agents = self.tracker.update(agents, timestamp)

        # --- dense branch (rate-decoupled) ---------------------------------
        if self._frame % self.occupancy_every == 0 or self._cached_freespace is None:
            semantics, mask_camera = self.occupancy(images, meta)
            self._cached_freespace = self.extractor(semantics, mask_camera)
        freespace = self._cached_freespace

        # --- motion forecasting --------------------------------------------
        if self.forecaster is not None:
            preds = self.forecaster(agents, meta)
            for a in agents:
                a.pred = preds.get(int(a.track_id))

        scene = SceneRepresentation(agents=agents, freespace=freespace, ego=ego,
                                    timestamp=timestamp)

        # --- candidate plans ------------------------------------------------
        candidates, planner_scores = self.planner(scene, command, meta)

        # --- safety gate ----------------------------------------------------
        risk_model = RiskModel(ego=ego, tracker=self.tracker)
        result = self.safety(candidates, scene, planner_scores, risk_model)

        self._frame += 1
        return PipelineOutput(scene=scene, candidates=candidates,
                              filter_result=result, command=command,
                              planner_scores=np.asarray(planner_scores))

    def reset(self) -> None:
        """Clear temporal state between scenes.

        Both the Kalman tracks and the cached occupancy are per-scene; carrying them
        across a scene boundary is the same class of bug as leaving `prev_bev`
        populated, and produces plausible-looking garbage rather than a crash.
        """
        self._frame = 0
        self._cached_freespace = None
        self.tracker = TrackCovarianceTracker(
            accel_noise=self.tracker.accel_noise,
            pos_noise=self.tracker.pos_noise,
            vel_noise=self.tracker.vel_noise,
            score_floor=self.tracker.score_floor,
            max_misses=self.tracker.max_misses,
        )


# ---------------------------------------------------------------------------
# Controller hand-off
# ---------------------------------------------------------------------------


def to_controller_waypoints(trajectory: np.ndarray, ego: EgoState) -> np.ndarray:
    """Adapt a filtered plan for `simulator/controller.py::TrajectoryController`.

    The controller runs pure pursuit over ego-frame waypoints at `dt_plan=0.5`, which
    is exactly what the filter emits, so this is a shape assertion plus the ego's own
    position prepended as the path origin — pure pursuit needs a segment to start
    from, not just future points.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 2:
        raise ValueError(f"expected (T, 2) trajectory, got {traj.shape}")
    return np.vstack([np.zeros((1, 2)), traj])
