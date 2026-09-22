"""Closed-loop driver: pipeline -> safety filter -> controller -> kinematic bicycle.

WHAT "CLOSED LOOP" MEANS HERE, PRECISELY
----------------------------------------
The ego is simulated. Its pose comes from integrating the KBM under the controls
its own planner produced, so a bad plan moves the car somewhere bad and the next
plan is made from there. That is the property open-loop replay cannot test: in
open loop the ego is teleported back onto the logged path every frame, which
hides exactly the compounding errors that matter.

The WORLD is log-replay and the agents are NON-REACTIVE. They follow their
recorded trajectories regardless of what the ego does, so this measures "can the
ego avoid a recorded world" and NOT "can the ego negotiate with other drivers".
An agent will drive through the ego without flinching. That is a real limitation
of log-replay closed loop, shared with nuPlan's non-reactive mode, and any
collision number here should be read with it in mind.

The reference route is the logged ego trajectory. Route completion therefore
measures progress along the path a human actually drove.

SENSOR SOURCE
-------------
Because the simulated ego diverges from the logged pose, camera frames stop
corresponding to where the car actually is, and running image-based perception on
them would be measuring the wrong scene. So the perception layer is pluggable:
`GTWorldModel` reads agents and free space from the annotations at the current
timestamp (an oracle), which isolates the planning stack under test. Swap in a
live adapter when the divergence is small enough to be worth it.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .freespace import FreeSpace, FreeSpaceExtractor, GridConfig
from .metrics import StepRecord, evaluate, format_report
from .safety_filter import SafetyFilter
from .scene import Agent, EgoState, SceneRepresentation
from .uncertainty import RiskModel, TrackCovarianceTracker

_SIM_DIR = Path(__file__).resolve().parent.parent / 'simulator'
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))


class WorldModel(Protocol):
    """Supplies the scene at a simulated ego pose and time."""

    def agents_at(self, t: float, ego_xy: np.ndarray, ego_yaw: float) -> list[Agent]:
        """Agents in EGO frame at simulation time `t`."""

    def freespace_at(self, t: float, ego_xy: np.ndarray,
                     ego_yaw: float) -> FreeSpace:
        """Drivable / obstacle rasters in EGO frame."""

    def route(self) -> np.ndarray:
        """(N, 2) world-frame reference polyline."""

    def duration(self) -> float:
        """Seconds of world available."""

    def measurement_noise(self) -> tuple[float, float]:
        """(pos_sigma_m, vel_sigma_mps) this source's estimates actually carry.

        The world declares its own fidelity because the risk gate is acutely
        sensitive to it — on scene-0655 the same 31 parked cars produced combined
        risk 0.475 with a detector-grade prior (0.5 m, 1.0 m/s) and 0.039 with a
        GT-grade one (0.1 m, 0.2 m/s), purely from the prior. Too loose and the
        filter rejects every candidate at 5 m of clearance; too tight and it waves
        through plans it should not. This is a calibration parameter, not a
        tuning knob, and it belongs with whoever knows the sensor.
        """


@dataclass
class LoopConfig:
    dt: float = 0.5                  # planning + control interval (nuScenes 2 Hz)
    horizon: int = 6                 # planner steps (3 s at dt=0.5)
    max_steps: int = 40
    wheelbase: float = 2.85
    ego_length: float = 4.6
    ego_width: float = 1.8
    initial_speed: float = 5.0
    substeps: int = 10               # KBM Euler sub-steps per dt


class ClosedLoopRunner:
    """Drive the planning stack against a world model and a bicycle model.

    `planner` gets the fused scene and returns (candidates, scores); the safety
    filter picks among them; the controller turns the winner into (steer, accel);
    the KBM integrates. Only the KBM advances ego state — nothing snaps the car
    back to the log, which is the whole point.
    """

    def __init__(self, world: WorldModel,
                 planner: Callable[[SceneRepresentation, int], tuple],
                 config: LoopConfig | None = None,
                 safety: SafetyFilter | None = None,
                 tracker: TrackCovarianceTracker | None = None) -> None:
        from controller import TrajectoryController
        # The simulator has its OWN EgoState (world pose x/y/yaw/v), distinct from
        # scene.EgoState (ego-frame dimensions + speed). Aliased so the collision
        # is explicit rather than a silent shadowing bug.
        from kbm import EgoState as SimEgoState
        from kbm import KinematicBicycleModel

        self.world = world
        self.planner = planner
        self.cfg = config or LoopConfig()
        self.safety = safety or SafetyFilter(dt=self.cfg.dt)
        if tracker is None:
            # Match the filter's noise model to what this world can actually
            # deliver, rather than assuming detector-grade error everywhere.
            pos_s, vel_s = (world.measurement_noise()
                            if hasattr(world, 'measurement_noise') else (0.5, 1.0))
            tracker = TrackCovarianceTracker(pos_noise=pos_s, vel_noise=vel_s)
        self.tracker = tracker
        self._KBM = KinematicBicycleModel
        self._SimEgoState = SimEgoState
        self._Controller = TrajectoryController

    def run(self, command: int = 1) -> tuple[list[StepRecord], dict]:
        cfg = self.cfg
        route = np.asarray(self.world.route(), dtype=np.float64)

        # Start the sim exactly where the log starts, heading along the route, so
        # route completion starts from zero rather than from an arbitrary offset.
        start = route[0]
        heading = route[min(3, len(route) - 1)] - start
        yaw0 = float(np.arctan2(heading[1], heading[0])) if np.linalg.norm(heading) > 1e-6 else 0.0

        kbm = self._KBM(wheelbase=cfg.wheelbase)
        kbm.reset(self._SimEgoState(x=float(start[0]), y=float(start[1]),
                                    yaw=yaw0, v=cfg.initial_speed))
        ctrl = self._Controller(wheelbase=cfg.wheelbase, dt_plan=cfg.dt)

        records: list[StepRecord] = []
        n_steps = min(cfg.max_steps, int(self.world.duration() / cfg.dt))

        for k in range(n_steps):
            t = k * cfg.dt
            x, y, yaw, v = [float(z) for z in kbm.state]
            ego_xy = np.array([x, y])
            lat: dict[str, float] = {}

            t0 = time.perf_counter()
            agents = self.world.agents_at(t, ego_xy, yaw)
            agents = self.tracker.update(agents, t)
            lat['perception'] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            fs = self.world.freespace_at(t, ego_xy, yaw)
            lat['occupancy'] = (time.perf_counter() - t0) * 1000

            ego = EgoState(speed=v, length=cfg.ego_length, width=cfg.ego_width,
                           wheelbase=cfg.wheelbase)
            scene = SceneRepresentation(agents=agents, freespace=fs, ego=ego,
                                        timestamp=t)

            t0 = time.perf_counter()
            candidates, scores = self.planner(scene, command)
            lat['planner'] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            result = self.safety(candidates, scene, scores,
                                 RiskModel(ego=ego, tracker=self.tracker))
            lat['safety_filter'] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            traj = np.asarray(result.trajectory, dtype=np.float64)
            control = ctrl.control(traj, v)
            lat['controller'] = (time.perf_counter() - t0) * 1000

            # World-frame copies for the metrics, which score in world coords.
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            boxes = [((R @ a.xy) + ego_xy, yaw + a.yaw,
                      float(a.lwh[0]), float(a.lwh[1]), int(a.track_id))
                     for a in agents]
            preds = {int(a.track_id): (a.pred.loc[0] @ R.T) + ego_xy
                     for a in agents if a.pred is not None}

            records.append(StepRecord(
                t=t, ego_xy=ego_xy, ego_yaw=yaw, ego_v=v,
                accel=control.accel, steer=control.delta,
                decision=('BRAKE' if result.emergency
                          else f'cand{result.chosen_index}'),
                agent_boxes=boxes, predictions=preds, latency_ms=lat,
                emergency=result.emergency))

            kbm.step(control.delta, control.accel, cfg.dt, substeps=cfg.substeps)

        metrics = evaluate(records, route, cfg.dt,
                           cfg.ego_length, cfg.ego_width)
        return records, metrics


# ---------------------------------------------------------------------------
# Ground-truth world model
# ---------------------------------------------------------------------------


class GTWorldModel:
    """World built from nuScenes annotations — an oracle perception layer.

    Deliberately not a camera pipeline: once the simulated ego diverges from the
    logged pose, the recorded images no longer show the scene the car is in, so
    image-based perception would be answering about the wrong place. Using GT
    isolates the planning stack, which is what these metrics are for. Perception
    error can be layered back in afterwards (see `noise` below) to measure how
    much of the planning margin it consumes.
    """

    def __init__(self, nusc, scene_idx: int, grid: GridConfig | None = None,
                 road_half_width: float = 10.0,
                 noise: float = 0.0, rng_seed: int = 0) -> None:
        from nuscenes.eval.detection.utils import category_to_detection_name
        from pyquaternion import Quaternion

        self.nusc = nusc
        self._cat = category_to_detection_name
        self._Q = Quaternion
        self.grid = grid or GridConfig(x=(-20.0, 60.0, 0.4),
                                       y=(-30.0, 30.0, 0.4),
                                       z=(-1.0, 5.4, 0.4))
        self.extractor = FreeSpaceExtractor(self.grid)
        self.road_half_width = road_half_width
        self.noise = float(noise)
        self.rng = np.random.default_rng(rng_seed)

        scene = nusc.scene[scene_idx]
        self.samples, tok = [], scene['first_sample_token']
        while tok:
            s = nusc.get('sample', tok)
            ep = nusc.get('ego_pose', nusc.get('sample_data',
                                               s['data']['LIDAR_TOP'])['ego_pose_token'])
            self.samples.append({'token': tok, 'anns': s['anns'],
                                 'xy': np.array(ep['translation'][:2]),
                                 'yaw': self._Q(ep['rotation']).yaw_pitch_roll[0]})
            tok = s['next']
        self._route = np.array([s['xy'] for s in self.samples])
        self.dt = 0.5

    def duration(self) -> float:
        return len(self.samples) * self.dt

    def measurement_noise(self) -> tuple[float, float]:
        """Annotations are exact, so only the injected `noise` is real error.

        A small floor remains because even GT boxes have annotation jitter and
        `box_velocity` is a finite difference over neighbouring keyframes.
        """
        return (max(0.1, self.noise), max(0.2, self.noise))

    def route(self) -> np.ndarray:
        return self._route

    def _frame(self, t: float) -> dict:
        return self.samples[int(np.clip(round(t / self.dt), 0, len(self.samples) - 1))]

    def agents_at(self, t: float, ego_xy: np.ndarray, ego_yaw: float) -> list[Agent]:
        fr = self._frame(t)
        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        R = np.array([[c, -s], [s, c]])          # world -> ego
        out = []
        for i, ann_tok in enumerate(fr['anns']):
            ann = self.nusc.get('sample_annotation', ann_tok)
            name = self._cat(ann['category_name'])
            if name is None:
                continue
            p = R @ (np.asarray(ann['translation'][:2]) - ego_xy)
            if np.linalg.norm(p) > 60.0:
                continue
            v = self.nusc.box_velocity(ann_tok)[:2]
            if np.isnan(v).any():
                v = np.zeros(2)
            if self.noise > 0:
                p = p + self.rng.normal(0, self.noise, 2)
                v = v + self.rng.normal(0, self.noise, 2)
            w, l, h = ann['size']
            yaw_w = self._Q(ann['rotation']).yaw_pitch_roll[0]
            out.append(Agent(
                track_id=abs(hash(ann['instance_token'])) % 100000,
                xy=p, yaw=float(yaw_w - ego_yaw),
                lwh=np.array([l, w, h]), vxy=R @ v,
                score=1.0, label=0))
        return out

    def freespace_at(self, t: float, ego_xy: np.ndarray,
                     ego_yaw: float) -> FreeSpace:
        """Road corridor around the route, minus agent footprints.

        The corridor is a band around the logged path rather than the map's
        drivable-area polygon: it keeps this module independent of the map
        expansion, and it makes the free-space constraint meaningful even on the
        two mini scenes whose maps carry no useful road geometry nearby.
        """
        from .freespace import FREE_CLASS
        nx, ny, nz = self.grid.shape
        sem = np.full((nx, ny, nz), FREE_CLASS, dtype=np.int64)
        k_ground = self.grid.z_band_indices(-0.4, 0.2)[0]
        k_lo, k_hi = self.grid.z_band_indices(0.2, 2.2)

        ix = self.grid.x[0] + (np.arange(nx) + 0.5) * self.grid.x[2]
        iy = self.grid.y[0] + (np.arange(ny) + 0.5) * self.grid.y[2]
        gx, gy = np.meshgrid(ix, iy, indexing='ij')

        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        R = np.array([[c, -s], [s, c]])
        route_ego = (self._route - ego_xy) @ R.T
        # Distance from every cell to the route polyline, vectorised per segment.
        dist = np.full((nx, ny), np.inf)
        for i in range(len(route_ego) - 1):
            a, b = route_ego[i], route_ego[i + 1]
            d = b - a
            L2 = float(d @ d)
            if L2 < 1e-9:
                continue
            tt = np.clip(((gx - a[0]) * d[0] + (gy - a[1]) * d[1]) / L2, 0.0, 1.0)
            dist = np.minimum(dist, np.hypot(gx - (a[0] + tt * d[0]),
                                             gy - (a[1] + tt * d[1])))
        sem[dist <= self.road_half_width, k_ground] = 11        # driveable_surface

        for a in self.agents_at(t, ego_xy, ego_yaw):
            cx = int((a.xy[0] - self.grid.x[0]) / self.grid.x[2])
            cy = int((a.xy[1] - self.grid.y[0]) / self.grid.y[2])
            hl = max(1, int(0.5 * a.lwh[0] / self.grid.x[2]))
            hw = max(1, int(0.5 * a.lwh[1] / self.grid.y[2]))
            x0, x1 = max(0, cx - hl), min(nx, cx + hl + 1)
            y0, y1 = max(0, cy - hw), min(ny, cy + hw + 1)
            if x0 < x1 and y0 < y1:
                sem[x0:x1, y0:y1, k_lo:k_hi] = 4                # car
        return self.extractor(sem)


def constant_velocity_planner(horizon: int = 6, dt: float = 0.5):
    """A deliberately simple planner: straight lines at a spread of speeds.

    Stands in for DiffusionDrive so the loop, the safety filter and the metrics
    can be exercised and validated before a learned planner is attached. Because
    it proposes candidates and nothing else, any avoidance visible in a rollout is
    attributable to the safety filter rather than to planning intelligence.
    """
    def plan(scene: SceneRepresentation, command: int):
        v = scene.ego.speed
        speeds = [max(0.0, v - 2.0), v, v + 1.0]
        cands, scores = [], []
        for sp in speeds:
            for lat in (0.0, 2.5, -2.5):
                x = np.arange(1, horizon + 1) * sp * dt
                ss = np.linspace(0.0, 1.0, horizon)
                y = lat * (3 * ss ** 2 - 2 * ss ** 3)
                cands.append(np.stack([x, y], axis=1))
                # Prefer holding speed and going straight.
                scores.append(1.0 / (1.0 + abs(sp - v) + abs(lat)))
        return np.stack(cands), np.array(scores)
    return plan


def diffusiondrive_anchor_planner(anchor_npy: str, dt: float = 0.5,
                                  speed_condition: bool = True,
                                  max_accel: float = 3.0):
    """Candidate set from DiffusionDrive's learned trajectory vocabulary.

    WHAT THIS IS, AND IS NOT.  This uses DiffusionDrive's `kmeans_plan_6.npy`
    anchors — the (3 commands x 6 anchors x 6 steps x 2) vocabulary its head
    denoises from — as the candidate set handed to the safety filter. It does NOT
    run the truncated-diffusion denoiser, because `DiffPlanner.forward` requires
    `feature_maps` and `agent_feature` from the Sparse4D image backbone, and in
    closed loop the ego diverges from the logged pose, so those images describe a
    scene the car is no longer in. Features from the wrong place are worse than no
    features. Call this "DiffusionDrive anchors", never "DiffusionDrive".

    Even so it is a real upgrade on hand-made straight lines: the shapes and the
    speed range are learned from driving data, and the command structure matches
    the head's.

    FRAME.  The anchors live in DiffusionDrive's planning frame, x = lateral
    (right positive), y = forward. This package uses x = forward, y = left. So
    (x, y)_ego = (y_dd, -x_dd). Getting this wrong rotates every anchor 90 degrees
    while leaving all the shapes looking plausible, which is precisely the kind of
    silent convention error that does not announce itself.

    COMMANDS follow the generator: 0 = right, 1 = left, 2 = straight.

    SPEED CONDITIONING.  Raw anchors are absolute waypoints and carry the speed of
    whatever manoeuvre they were clustered from — measured here, one command's six
    anchors imply 0.1 to 14.6 m/s. Offered unmodified to a car doing 5 m/s, four of
    six fail the dynamics gate on acceleration alone, and the filter emergency-brakes
    with nothing to choose from (93 brakes across five scenes, against 55 for a
    speed-aware stand-in).

    That is not an anchor defect; it is what the denoiser is for. DiffusionDrive
    conditions on ego state and deforms the anchor into something reachable, and
    skipping that step leaves a shape prior masquerading as a candidate set. With
    `speed_condition` the along-track extent is rescaled to a reachable speed while
    the lateral profile is preserved — a crude stand-in for the denoiser, but it
    makes the comparison about trajectory SHAPE rather than about arithmetic the
    denoiser would have done.
    """
    raw = np.load(anchor_npy)                      # (3, K, T, 2) in DD frame
    if raw.ndim != 4 or raw.shape[0] != 3 or raw.shape[-1] != 2:
        raise ValueError(f'expected (3, K, T, 2) anchors, got {raw.shape}')
    # DD (lateral, forward) -> ego (forward, left)
    anchors = np.stack([raw[..., 1], -raw[..., 0]], axis=-1)

    def plan(scene: SceneRepresentation, command: int):
        cmd = int(np.clip(command, 0, 2))
        cands = anchors[cmd].astype(np.float64).copy()    # (K, T, 2)
        T = cands.shape[1]

        if speed_condition:
            v0 = scene.ego.speed
            reach_lo = max(0.0, v0 - max_accel * T * dt)
            reach_hi = v0 + max_accel * T * dt
            for i in range(len(cands)):
                span = float(np.linalg.norm(cands[i, -1]))
                implied = span / (T * dt)
                if implied < 1e-3:
                    continue
                target = float(np.clip(implied, reach_lo, reach_hi))
                # Scale along-track only: the lateral profile is the shape we are
                # actually testing and must survive the rescale.
                cands[i] *= target / implied
        # Prefer anchors whose implied speed is closest to what the car is doing;
        # a standing start should not be handed a 14 m/s highway anchor as its
        # top-scored option.
        v = max(scene.ego.speed, 0.1)
        implied = np.linalg.norm(cands[:, -1, :], axis=1) / (cands.shape[1] * dt)
        scores = 1.0 / (1.0 + np.abs(implied - v))
        return cands, scores / scores.sum()
    return plan
