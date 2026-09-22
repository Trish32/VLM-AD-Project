"""
Unified scene representation — the single hand-off between perception and planning.

The stack has two perception branches that answer different questions:

  Sparse4D v3  — "what objects are there, and where are they going?"  (object-centric)
  FlashOcc     — "what space is occupied, regardless of what occupies it?"  (dense)

Neither subsumes the other.  Sparse4D only ever reports things that fall in the 10
scored nuScenes classes and clear a score threshold, so an unclassified obstacle —
debris, a jersey barrier at an odd angle, an overhanging truck bed — is simply absent
from its output.  FlashOcc marks those voxels occupied without needing a name for
them, but it has no notion of identity and therefore cannot be tracked or predicted.
The planner needs both, so this module is where they are fused into one object.

FRAME CONTRACT
--------------
Everything in a `SceneRepresentation` is expressed in the **ego frame at the current
keyframe**, right-handed, metres:

    +x forward,  +y left,  +z up,  yaw measured CCW from +x

This is NOT the frame either branch natively produces:

  * Sparse4D operates in the **LiDAR** frame, which on nuScenes carries a ~-90 deg
    yaw offset relative to ego (+x right, +y forward).  See
    `bevfusion_vldrive/viz_bug_log.txt` BUG 1 for the same trap hit in visualisation.
  * FlashOcc produces its grid in the **key-ego** frame (camera-0's ego pose).

Both therefore need a transform on the way in, and getting it wrong is silent — a 90
deg BEV rotation raises no exception, it just makes every downstream collision check
confidently wrong.  `adapters.py` owns those transforms, and
`tests/test_scene.py::test_frame_roundtrip` is the guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    """One tracked object, with its uncertainty.

    Sparse4D emits a box and a track id but no covariance — it is a detector with an
    instance bank, not a filter.  `cov` is therefore populated downstream by
    `uncertainty.TrackCovarianceTracker`, which runs a constant-velocity Kalman filter
    over the association that Sparse4D already provides for free.

    Attributes
    ----------
    track_id : stable across frames (from Sparse4D v3's instance bank).
    xy       : (2,)  position in ego frame, metres.
    yaw      : heading, rad CCW from +x.
    lwh      : (3,)  length, width, height, metres.
    vxy      : (2,)  velocity in ego frame, m/s.
    score    : detection confidence in [0, 1]; feeds the Kalman measurement noise.
    label    : nuScenes class index.
    cov      : (4, 4) covariance over [x, y, vx, vy], or None before filtering.
    pred     : optional `TrajectoryDistribution` from QCNet — filled in by the
               motion branch, left None for agents we do not forecast.
    """

    track_id: int
    xy: np.ndarray
    yaw: float
    lwh: np.ndarray
    vxy: np.ndarray
    score: float
    label: int
    cov: np.ndarray | None = None
    pred: "TrajectoryDistribution | None" = None

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vxy))

    @property
    def radius(self) -> float:
        """Circumscribed radius of the footprint — the cheap conservative proxy used
        for collision probability, where an exact box-box integral is not worth it."""
        return 0.5 * float(np.hypot(self.lwh[0], self.lwh[1]))


@dataclass
class TrajectoryDistribution:
    """A multi-modal future for one agent, as QCNet actually emits it.

    QCNet's decoder regresses a Laplace per mode / timestep / dim — `loc` and `scale`
    are both already in the checkpoint's output (see
    `motionForecasting/QCNet/modules/qcnet_decoder.py`, `to_scale_refine_pos`), they
    were simply never plumbed anywhere.  Item 3 of this pipeline is mostly about
    *using* what the model already predicts rather than adding new modelling.

    loc   : (K, T, 2)  per-mode mean waypoints, ego frame, metres
    scale : (K, T, 2)  per-mode Laplace scale b  (NOT a standard deviation — see
            `sigma` below), metres
    probs : (K,)       mode probabilities, sum to 1
    dt    : seconds between consecutive steps
    """

    loc: np.ndarray
    scale: np.ndarray
    probs: np.ndarray
    dt: float = 0.5

    # A Laplace with scale b has variance 2b^2, so the equivalent Gaussian sigma is
    # b*sqrt(2).  Converting lets us combine prediction spread with the Kalman
    # covariance (which is genuinely Gaussian) in a single second-moment budget.
    @property
    def sigma(self) -> np.ndarray:
        """(K, T, 2) Gaussian-equivalent standard deviation."""
        return self.scale * np.sqrt(2.0)

    def __post_init__(self) -> None:
        if self.loc.shape != self.scale.shape:
            raise ValueError(
                f"loc {self.loc.shape} and scale {self.scale.shape} must match")
        if self.probs.shape[0] != self.loc.shape[0]:
            raise ValueError(
                f"probs has {self.probs.shape[0]} modes, loc has {self.loc.shape[0]}")


# ---------------------------------------------------------------------------
# Ego
# ---------------------------------------------------------------------------


@dataclass
class EgoState:
    """Ego state at the current keyframe, in its own frame (so xy = 0, yaw = 0).

    Kept explicit rather than implied because the feasibility checks need the current
    speed to evaluate whether a candidate's implied acceleration is reachable, and the
    footprint to sweep against the free-space grid.
    """

    speed: float                       # m/s, forward
    yaw_rate: float = 0.0              # rad/s
    accel: float = 0.0                 # m/s^2, longitudinal
    length: float = 4.6                # a typical sedan; matches the KBM's 2.85 m
    width: float = 1.8                 # wheelbase plus overhangs
    wheelbase: float = 2.85            # keep in sync with simulator/kbm.py

    @property
    def footprint_radius(self) -> float:
        return 0.5 * float(np.hypot(self.length, self.width))


# ---------------------------------------------------------------------------
# The fused representation
# ---------------------------------------------------------------------------


@dataclass
class SceneRepresentation:
    """Everything the planner and the safety filter are allowed to see.

    Deliberately holds *no* raw tensors.  Feeding FlashOcc's (200, 200, 16, 18) logit
    volume into a planner would be both enormous and redundant — what planning needs
    from occupancy is where it can go and how much room it has, which is two orders of
    magnitude smaller and reusable by every downstream consumer.  `freespace.py` does
    that reduction; this class just carries the result.
    """

    agents: list[Agent]
    freespace: "FreeSpace"             # from freespace.py, forward-declared
    ego: EgoState
    timestamp: float = 0.0
    frame_id: str = "ego"

    def moving_agents(self, min_speed: float = 0.5) -> list[Agent]:
        return [a for a in self.agents if a.speed >= min_speed]

    def agents_within(self, radius: float) -> list[Agent]:
        return [a for a in self.agents if float(np.linalg.norm(a.xy)) <= radius]

    def summary(self) -> str:
        n_pred = sum(a.pred is not None for a in self.agents)
        n_cov = sum(a.cov is not None for a in self.agents)
        return (f"SceneRepresentation(t={self.timestamp:.2f} "
                f"agents={len(self.agents)} [cov={n_cov} pred={n_pred}] "
                f"ego_speed={self.ego.speed:.1f} m/s "
                f"free={self.freespace.free_fraction:.1%})")


# ---------------------------------------------------------------------------
# Geometry helpers shared by the free-space and safety modules
# ---------------------------------------------------------------------------


def ego_footprint_corners(xy: np.ndarray, yaw: float,
                          length: float, width: float) -> np.ndarray:
    """Four corners (4, 2) of a rectangle centred at `xy` with heading `yaw`.

    Corner order is CCW starting from rear-right, which keeps the polygon convex for
    the sampling in `safety_filter.swept_footprint_cells`.
    """
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[-hl, -hw], [+hl, -hw], [+hl, +hw], [-hl, +hw]], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.asarray(xy, dtype=np.float64)


def yaw_from_waypoints(waypoints: np.ndarray, initial_yaw: float = 0.0) -> np.ndarray:
    """Per-waypoint heading (T,) inferred from finite differences.

    Planners emit positions, but a footprint sweep needs an orientation at each step.
    The first heading falls back to `initial_yaw` (ego is at yaw 0 in its own frame),
    and near-stationary segments hold the previous heading rather than producing a
    meaningless atan2 of numerical noise.
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    deltas = np.diff(wp, axis=0, prepend=wp[:1])
    yaws = np.empty(len(wp), dtype=np.float64)
    prev = float(initial_yaw)
    for i, d in enumerate(deltas):
        if np.hypot(d[0], d[1]) < 1e-3:
            yaws[i] = prev
        else:
            prev = yaws[i] = float(np.arctan2(d[1], d[0]))
    return yaws
