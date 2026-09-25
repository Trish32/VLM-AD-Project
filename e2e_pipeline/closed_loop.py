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
from .metrics.metrics import StepRecord, evaluate, format_report
from .planner.safety_filter import SafetyFilter
from .calibration.calibration import (
    PlattCalibrator, expected_calibration_error, scale_for_risk_budget,
    scale_trajectory)
from .planner.structured import build_structured, structured_gate

# Beyond this drift the logged future is not a counterfactual (see run()).
COUNTERFACTUAL_MAX_DIVERGENCE_M = 3.0
from .planner.verifier import (
    DEGRADE_NONE, TrajectoryVerifier, compare_to_shadow, decelerate_along)
from .planner.world_model import (AnalyticCritic, KinematicWorldModel,
                                  plan_with_world_model)
from .scene import Agent, EgoState, SceneRepresentation
from .temporal_occlusion import TemporalOcclusionMemory
from .uncertainty import RiskModel, TrackCovarianceTracker

_SIM_DIR = Path(__file__).resolve().parent.parent / 'simulator'
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))


class WorldModel(Protocol):
    """Supplies the scene at a simulated ego pose and time."""

    def _drivable_paths(self):
        """Drivable-area polygon boundaries for this scene's location, cached.

        Uses the polygons directly rather than `get_map_mask`, whose `patch_angle`
        rotates the PATCH -- the inverse of rotating the world. That sign produced
        a silent bug in the visualiser earlier in this project (boxes correctly
        oriented on a wrongly-rotated road); point-in-polygon has no such trap.
        """
        if self._map_paths is not None:
            return self._map_paths
        from matplotlib.path import Path as _Path
        from nuscenes.map_expansion.map_api import NuScenesMap
        scene = self.nusc.get('scene',
                              self.nusc.get('sample',
                                            self.samples[0]['token'])['scene_token'])
        loc = self.nusc.get('log', scene['log_token'])['location']
        self._map = NuScenesMap(dataroot=self.nusc.dataroot, map_name=loc)
        paths = []
        for rec in self._map.drivable_area:
            for tok in rec['polygon_tokens']:
                poly = self._map.extract_polygon(tok)
                paths.append(_Path(np.asarray(poly.exterior.coords)))
        self._map_paths = paths
        return paths

    def _map_drivable(self, gx, gy, ego_xy, ego_yaw):
        """Ego-frame cell centres -> on-drivable-area mask, via the real map."""
        c, s = np.cos(ego_yaw), np.sin(ego_yaw)
        wx = ego_xy[0] + c * gx - s * gy          # ego -> global
        wy = ego_xy[1] + s * gx + c * gy
        pts = np.column_stack([wx.ravel(), wy.ravel()])
        out = np.zeros(len(pts), dtype=bool)
        for path in self._drivable_paths():
            lo, hi = path.vertices.min(0), path.vertices.max(0)
            box = ((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) &
                   (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))
            if not box.any():
                continue                      # bbox reject before the costly test
            out[box] |= path.contains_points(pts[box])
        return out.reshape(gx.shape)

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
    use_world_model: bool = False    # rank filter survivors by rollout score
    use_verifier: bool = False       # independent rule check before the controller
    use_shadow: bool = False         # compare against an independent safe plan
    use_structured: bool = False     # TTC-response gate on the structured view
    calibrate_risk: bool = False     # Platt-map the risk model's output
    risk_budget: float = 0.0         # >0: solve speed scale for this budget
    unknown_prior: float = 0.0       # P(obstacle) per unobserved swept cell
    veto_unknown: bool = True        # hard-reject unknown, vs price it softly
    follow_logged_ego: bool = False  # pin the ego to the logged trajectory
    rollout_steps: int = 0           # 0 = use the full candidate horizon
    world_model_keep: int = 3        # candidates surviving the rollout prune


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
                 tracker: TrackCovarianceTracker | None = None,
                 latent_model=None, critic=None) -> None:
        # NOTE two distinct senses of "world model" meet here. `world` is the
        # ENVIRONMENT the ego drives in (GTWorldModel, the oracle). `latent_model`
        # is the PREDICTIVE model the planner imagines with before acting. They
        # are unrelated objects; the shared word is unfortunate and is called out
        # rather than left for a reader to trip over.
        from controller import TrajectoryController
        # The simulator has its OWN EgoState (world pose x/y/yaw/v), distinct from
        # scene.EgoState (ego-frame dimensions + speed). Aliased so the collision
        # is explicit rather than a silent shadowing bug.
        from kbm import EgoState as SimEgoState
        from kbm import KinematicBicycleModel

        self.world = world
        self.verifier = TrajectoryVerifier(dt=(config or LoopConfig()).dt)
        self.verifier_fired = 0
        self.verifier_rules: dict = {}
        self.shadow_fired: dict = {}
        self.shadow_excess: list = []
        self.structured_fired = 0
        self.structured_reasons: dict = {}
        # Fitted in 2a812f4, validated across disjoint scenes in abf16d8.
        self.calibrator = PlattCalibrator(a=0.457, b=-2.333, fitted=True,
                                          n_positive=11)
        self.risk_pred: list = []        # for closed-loop ECE monitoring
        self.risk_outcome: list = []
        self.commands: list = []         # command actually used, per step

        self.latent_model = latent_model or KinematicWorldModel(
            wheelbase=(config or LoopConfig()).wheelbase)
        self.critic = critic or AnalyticCritic(dt=(config or LoopConfig()).dt)
        self.planner = planner
        self.cfg = config or LoopConfig()
        self.safety = safety or SafetyFilter(dt=self.cfg.dt)
        # `veto_unknown=False` is documented as "price unknown space softly
        # instead of rejecting it", but it only ever reached the FlashOcc
        # adapter's masking of `traversable`. The gate that actually rejects is
        # FeasibilityLimits.allow_unknown, which it never touched -- so turning
        # the veto off left gate 1 removing every occlusion-entering candidate,
        # and the soft prior had nothing left to price. Measured: candidates
        # enter unknown space on 20% of steps with the six straight anchors and
        # 84% with the full vocabulary, and not one of them survived to gate 4.
        if not self.cfg.veto_unknown and not self.safety.limits.allow_unknown:
            from dataclasses import replace as _replace
            self.safety.limits = _replace(self.safety.limits,
                                          allow_unknown=True)
        if tracker is None:
            # Match the filter's noise model to what this world can actually
            # deliver, rather than assuming detector-grade error everywhere.
            pos_s, vel_s = (world.measurement_noise()
                            if hasattr(world, 'measurement_noise') else (0.5, 1.0))
            # The calibrated curve was fitted to BEVFormer detections and
            # carries their scale, INCLUDING a 0.6 m floor at score 1.0.
            # Applying it to the GT oracle would hand annotations a detector's
            # error, and leaving it off for live detections would keep tracking
            # them at the oracle's 0.1 m. The world is the only object that
            # knows which it is, so it declares it -- same argument as
            # `measurement_noise` itself.
            tracker = TrackCovarianceTracker(
                pos_noise=pos_s, vel_noise=vel_s,
                calibrated_noise=bool(getattr(world, 'detector_grade', False)))
        self.tracker = tracker
        self._KBM = KinematicBicycleModel
        self._SimEgoState = SimEgoState
        self._Controller = TrajectoryController

    def calibration_report(self, n_bins: int = 5) -> dict:
        """Closed-loop ECE over this rollout's (prediction, outcome) pairs.

        Reported from the loop itself so miscalibration is visible where it does
        damage, rather than only in an offline script that nobody reruns after a
        change to the risk model.
        """
        pairs = [(p, o) for p, o in zip(self.risk_pred, self.risk_outcome)
                 if p is not None]
        if not pairs:
            return {'ece': float('nan'), 'n': 0}
        p, o = zip(*pairs)
        return expected_calibration_error(np.array(p), np.array(o),
                                          n_bins, 'quantile')

    def run(self, command: int | None = 1) -> tuple[list[StepRecord], dict]:
        """`command=None` derives the command per step from the world's route.

        A fixed int keeps the old behaviour and is what every existing caller
        passes, so no committed result moves. It is also the wrong default for a
        20-step rollout through a scene where the ego turns: one command for the
        whole rollout cannot describe a route that changes.
        """
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
            # Pin the ego to the logged pose before reading state, so
            # perception sees the scene from the viewpoint its detections were
            # actually computed at.
            #
            # WHY THIS MODE EXISTS. Replacing the GT oracle with live detections
            # confounded two effects: the detector's own error, and a 7.7 m mean
            # divergence between the simulated and logged ego. Detections are
            # computed once, from the logged pose, so an object near the
            # simulated ego but occluded from the logged one is simply absent --
            # a field-of-view mismatch, not a coordinate error, and unfixable
            # without sensor data from a pose the car never occupied.
            #
            # Pinning the ego to the log drives that divergence to zero. The
            # planner, filter and metrics all still run; only the ego's realised
            # motion is taken from the recording. So a GT-vs-live difference
            # measured in this mode is DETECTOR ERROR ALONE.
            #
            # The cost is that it is no longer a closed loop: the planner's
            # output does not affect where the ego goes, so route completion
            # becomes trivially the logged route and collision counts describe
            # the human's trajectory rather than the planner's. Safety and
            # clearance remain meaningful because they are evaluated against the
            # agents the pipeline actually perceived.
            if cfg.follow_logged_ego:
                fr = self.world._frame(t) if hasattr(self.world, '_frame') else None
                if fr is not None:
                    kbm.reset(self._SimEgoState(
                        x=float(fr['xy'][0]), y=float(fr['xy'][1]),
                        yaw=float(fr['yaw']), v=float(kbm.state[3])))

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
            cmd_k = (command if command is not None
                     else (self.world.command_at(t)
                           if hasattr(self.world, 'command_at') else 2))
            self.commands.append(int(cmd_k))
            candidates, scores = self.planner(scene, cmd_k)
            lat['planner'] = (time.perf_counter() - t0) * 1000

            # Rollout ranking runs BEFORE the safety filter, not after. Placed
            # after, it only ever saw the 28.5% of steps where the filter had
            # left something feasible -- measured: 57 rankable of 200. Placed
            # here it scores every candidate on every step, which was the
            # largest measured cause of its earlier failure.
            #
            # It PRUNES, it does not overrule: the hard gate still runs on
            # whatever survives, so a soft score never sits in front of a safety
            # veto. What it adds that the filter structurally cannot is
            # INTERACTION -- the filter propagates agents as though the ego were
            # not there, so it cannot see a candidate that is probabilistically
            # clear only because nobody reacted to it.
            lat['world_model'] = 0.0
            if cfg.use_world_model and len(candidates):
                t0 = time.perf_counter()
                ranked = plan_with_world_model(candidates, scene,
                                               model=self.latent_model,
                                               critic=self.critic, dt=cfg.dt,
                                               prior=scores)
                keep = [r.index for r in ranked[:max(1, cfg.world_model_keep)]]
                candidates = np.asarray(candidates)[keep]
                scores = np.asarray(scores)[keep] if scores is not None else None
                lat['world_model'] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            risk_model = RiskModel(
                ego=ego, tracker=self.tracker,
                calibrator=self.calibrator if cfg.calibrate_risk else None,
                unknown_prior=cfg.unknown_prior)
            result = self.safety(candidates, scene, scores, risk_model)
            lat['safety_filter'] = (time.perf_counter() - t0) * 1000


            # (b) Speed planning from the risk budget. Risk SCORES, this
            # DECIDES, and the filter above already vetoed the unacceptable --
            # so this only ever slows an already-admissible plan. Requires the
            # calibrated model: solving a budget against a 6x-inflated input
            # saturates the response on situations carrying 2% real risk.
            traj = np.asarray(result.trajectory, dtype=np.float64)
            if cfg.risk_budget > 0 and not result.emergency and len(traj):
                scale = scale_for_risk_budget(
                    traj, scene, cfg.risk_budget, self.calibrator,
                    lambda p: risk_model.evaluate(np.asarray(p, float),
                                                  scene.agents, dt=cfg.dt,
                                                  freespace=scene.freespace).total)
                traj = scale_trajectory(traj, scale)

            # (c) ECE monitoring: record what was predicted for the plan about
            # to be executed. The outcome is filled in below once the step has
            # happened, so calibration drift is observable in the loop rather
            # than only in an offline script.
            #
            # `freespace` omitted here until measured: the recorded risk then
            # excluded the occlusion prior that the FILTER had just applied, so
            # the monitored number was not the number that gated the decision,
            # and a prior sweep showed "mean risk identical at every prior"
            # because the column could not see it. Sixth instance of the same
            # omission; the argument is keyword-only and defaults to None, which
            # is why every one of them was silent.
            if not result.emergency and len(traj):
                self.risk_pred.append(float(risk_model.evaluate(
                    traj, scene.agents, dt=cfg.dt,
                    freespace=scene.freespace).total))
            else:
                self.risk_pred.append(None)

            lat['verifier'] = 0.0
            if cfg.use_verifier and len(traj):
                t0 = time.perf_counter()
                vr = self.verifier.verify(traj, scene, light=getattr(self, 'light', 'none'),
                                          decision=getattr(self, 'decision', ''))
                if not vr.passed:
                    self.verifier_fired += 1
                    for vv in vr.violations:
                        self.verifier_rules[vv.rule] = self.verifier_rules.get(vv.rule, 0) + 1
                    traj = np.asarray(vr.trajectory, dtype=np.float64)
                lat['verifier'] = (time.perf_counter() - t0) * 1000
            if cfg.use_structured and len(traj):
                t0 = time.perf_counter()
                st = build_structured(scene, light=getattr(self, 'light', 'none'))
                iv = structured_gate(st, traj, v, dt=cfg.dt)
                if iv:
                    self.structured_fired += 1
                    for x in iv:
                        self.structured_reasons[x.reason] = \
                            self.structured_reasons.get(x.reason, 0) + 1
                    traj = decelerate_along(traj, v, cfg.dt)
                lat['verifier'] += (time.perf_counter() - t0) * 1000

            if cfg.use_shadow and len(traj):
                t0 = time.perf_counter()
                sr = compare_to_shadow(traj, scene, dt=cfg.dt)
                self.shadow_excess.append(sr.excess_m)
                if sr.action != DEGRADE_NONE:
                    self.shadow_fired[sr.action] = self.shadow_fired.get(sr.action, 0) + 1
                    traj = np.asarray(sr.trajectory, dtype=np.float64)
                lat['verifier'] += (time.perf_counter() - t0) * 1000

            # Counterfactual safety, computed here because it needs the world
            # model: the same agents, from the same pose, scored against the
            # plan and against what the human actually did next. No rollout, so
            # no divergence -- this is the one measure in the report that asks
            # whether the PLANNER is unsafe rather than whether the simulation
            # drifted.
            # Only valid while the ego is NEAR the logged pose. The comparison
            # transplants the human's next few poses onto wherever the ego
            # currently is; at 22 m of drift that is not an alternative the
            # human could have driven, it is a teleport out of the agent cloud,
            # and it scores as low risk for the wrong reason. Measured on
            # scene 6 before this gate: +0.3879 mean delta, 80% worse -- against
            # -0.0004 and 14% across all scenes at low divergence.
            #
            # So the metric that was supposed to be divergence-free is only
            # divergence-free where divergence is small. Gating it keeps that
            # property instead of quietly losing it.
            risk_plan = risk_log = 0.0
            cf_ok = False
            _cf_div = (float(np.linalg.norm(np.array([x, y]) - self.world._frame(t)['xy']))
                       if hasattr(self.world, '_frame') else 0.0)
            if len(traj) and _cf_div <= COUNTERFACTUAL_MAX_DIVERGENCE_M \
                    and hasattr(self.world, 'samples'):
                try:
                    smp = self.world.samples
                    ki = int(np.clip(round(t / cfg.dt), 0, len(smp) - 1))
                    cy, sy = np.cos(-yaw), np.sin(-yaw)
                    fut = []
                    for h in range(1, len(traj) + 1):
                        j = min(ki + h, len(smp) - 1)
                        dv = smp[j]['xy'] - np.array([x, y])
                        fut.append([cy * dv[0] - sy * dv[1], sy * dv[0] + cy * dv[1]])
                    risk_plan = float(risk_model.evaluate(traj, scene.agents, dt=cfg.dt,
                                                          freespace=scene.freespace).total)
                    risk_log = float(risk_model.evaluate(np.asarray(fut, float), scene.agents,
                                                         dt=cfg.dt, freespace=scene.freespace).total)
                    cf_ok = True
                except Exception:
                    cf_ok = False

            t0 = time.perf_counter()
            control = ctrl.control(traj, v)
            lat['controller'] = (time.perf_counter() - t0) * 1000

            # Outcome for the prediction recorded above: closest approach this
            # step, thresholded the same way the calibration set was.
            _cl = min((float(np.linalg.norm(np.asarray(a.xy, float)))
                       for a in scene.agents), default=float('inf'))
            self.risk_outcome.append(1.0 if _cl <= 1.0 else 0.0)

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
                planned_traj=traj.copy(),
                risk_planner=risk_plan, risk_logged=risk_log,
                counterfactual_valid=cf_ok,
                divergence_m=float(np.linalg.norm(
                    np.array([x, y]) - self.world._frame(t)['xy']))
                if hasattr(self.world, '_frame') else 0.0,
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

    `detector_grade = False`: these are annotations, so the tracker must use the
    declared `measurement_noise`, not the curve fitted to BEVFormer output.
    Subclasses that swap in real detections override it.
    """

    detector_grade = False

    def __init__(self, nusc, scene_idx: int, grid: GridConfig | None = None,
                 road_half_width: float = 10.0,
                 noise: float = 0.0, rng_seed: int = 0,
                 use_map: bool = False) -> None:
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
        # Real drivable area instead of a band around the logged path. The band
        # keeps this module map-expansion-free, but it left 71.5% of closed-loop
        # steps with no admissible candidate at all -- the anchors project ~46 m
        # forward at 15 m/s and a 10 m corridor around a curving route cannot
        # contain that. See RESULT.md, "why the harness is the binding constraint".
        self.use_map = bool(use_map)
        self._map = None
        self._map_paths = None
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

    def initial_speed(self) -> float:
        """Logged ego speed at t=0, so the rollout starts where the scene does.

        Seeding a fixed speed instead makes the ego diverge from the logged
        corridor on step one for no reason other than the constant being wrong:
        nuScenes-mini ego speeds at frame 0 span 0 to 12 m/s.
        """
        if len(self._route) < 2:
            return 0.0
        return float(np.linalg.norm(self._route[1] - self._route[0]) / self.dt)

    def command_at(self, t: float, horizon: int = 6) -> int:
        """Drive command implied by the logged ego future at time `t`.

        nuScenes ships no navigation command: upstream DERIVES `gt_ego_fut_cmd`
        from where the ego actually went, thresholding the final future
        waypoint's lateral offset at +-2 m. This is the same rule against the
        same source, so it is the closest thing to "what the route planner would
        have said" that this dataset supports.

        It is an ORACLE, and calling it anything else would overstate it -- a
        real stack gets the command from a navigation layer, not from the
        recording. It is the right default here because the alternative in place
        was a hardcoded 'straight', which is also an oracle and additionally a
        wrong one whenever the ego turned.
        """
        from .planner.vlm_planner import command_from_future
        smp = self.samples
        if not smp:
            return 2
        k = int(np.clip(round(t / self.dt), 0, len(smp) - 1))
        fr = smp[k]
        c, s = np.cos(-fr['yaw']), np.sin(-fr['yaw'])
        fut = []
        for h in range(1, horizon + 1):
            j = min(k + h, len(smp) - 1)
            d = smp[j]['xy'] - fr['xy']
            fut.append([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        return command_from_future(fut)

    def camera_at(self, t: float) -> dict:
        """Front camera at the LOGGED pose, with an exact world->image matrix.

        There is no camera image at the *simulated* pose and there cannot be:
        nuScenes only holds frames from where the car actually drove. So this
        returns the logged camera, and callers project world-frame geometry
        through it. That projection is exact -- an agent box or a planned path
        expressed in world coordinates lands in the right pixels -- it is simply
        seen from the logged viewpoint rather than the simulated one. The
        distance between the two poses is returned as `divergence_m` so the
        overlay can state it rather than let the viewer assume it is zero.
        """
        fr = self._frame(t)
        sample = self.nusc.get('sample', fr['token'])
        sd = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ep = self.nusc.get('ego_pose', sd['ego_pose_token'])

        K = np.eye(4)
        K[:3, :3] = np.asarray(cs['camera_intrinsic'], dtype=np.float64)
        R_e = self._Q(ep['rotation']).rotation_matrix
        t_e = np.asarray(ep['translation'], dtype=np.float64)
        R_c = self._Q(cs['rotation']).rotation_matrix
        t_c = np.asarray(cs['translation'], dtype=np.float64)

        T_we = np.eye(4)                      # world -> ego
        T_we[:3, :3], T_we[:3, 3] = R_e.T, -R_e.T @ t_e
        T_ec = np.eye(4)                      # ego -> camera
        T_ec[:3, :3], T_ec[:3, 3] = R_c.T, -R_c.T @ t_c

        return {'path': self.nusc.get_sample_data_path(sd['token']),
                'P': K @ T_ec @ T_we,
                'ego_xy': np.asarray(ep['translation'][:2], dtype=np.float64),
                'ego_yaw': float(self._Q(ep['rotation']).yaw_pitch_roll[0])}

    def _frame(self, t: float) -> dict:
        return self.samples[int(np.clip(round(t / self.dt), 0, len(self.samples) - 1))]

    def _drivable_paths(self):
        """Drivable-area polygon boundaries for this scene's location, cached.

        Uses the polygons directly rather than `get_map_mask`, whose `patch_angle`
        rotates the PATCH -- the inverse of rotating the world. That sign produced
        a silent bug in the visualiser earlier in this project (boxes correctly
        oriented on a wrongly-rotated road); point-in-polygon has no such trap.
        """
        if self._map_paths is not None:
            return self._map_paths
        from matplotlib.path import Path as _Path
        from nuscenes.map_expansion.map_api import NuScenesMap
        scene = self.nusc.get('scene',
                              self.nusc.get('sample',
                                            self.samples[0]['token'])['scene_token'])
        loc = self.nusc.get('log', scene['log_token'])['location']
        self._map = NuScenesMap(dataroot=self.nusc.dataroot, map_name=loc)
        paths = []
        for rec in self._map.drivable_area:
            for tok in rec['polygon_tokens']:
                poly = self._map.extract_polygon(tok)
                paths.append(_Path(np.asarray(poly.exterior.coords)))
        self._map_paths = paths
        return paths

    def _map_drivable(self, gx, gy, ego_xy, ego_yaw):
        """Ego-frame cell centres -> on-drivable-area mask, via the real map."""
        c, s = np.cos(ego_yaw), np.sin(ego_yaw)
        wx = ego_xy[0] + c * gx - s * gy          # ego -> global
        wy = ego_xy[1] + s * gx + c * gy
        pts = np.column_stack([wx.ravel(), wy.ravel()])
        out = np.zeros(len(pts), dtype=bool)
        for path in self._drivable_paths():
            lo, hi = path.vertices.min(0), path.vertices.max(0)
            box = ((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) &
                   (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))
            if not box.any():
                continue                      # bbox reject before the costly test
            out[box] |= path.contains_points(pts[box])
        return out.reshape(gx.shape)

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
        on_road = (self._map_drivable(gx, gy, ego_xy, ego_yaw) if self.use_map
                   else dist <= self.road_half_width)
        sem[on_road, k_ground] = 11                             # driveable_surface

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
    anchors -- the (3 commands x 6 anchors x 6 steps x 2) vocabulary its head
    denoises from -- as the candidate set handed to the safety filter. It does
    NOT run the truncated-diffusion denoiser, because `DiffPlanner.forward`
    requires `feature_maps` and `agent_feature` from the Sparse4D image backbone,
    and in closed loop the ego diverges from the logged pose, so those images
    describe a scene the car is no longer in.

    Speed conditioning delegates to `vlm_planner.intent_conditioned_planner`,
    which re-times the anchors PER STEP. An earlier version here clipped the
    horizon-average speed and therefore did nothing: at dt=0.5 s and 3 m/s^2 the
    opening waypoint can only move +-1.5 m/s worth of distance, so an anchor
    whose first step is too aggressive fails the dynamics gate however sensible
    its endpoint looks. In closed loop that deadlocked the car -- once stopped,
    every candidate demanded ~16 m/s^2 off the line, nothing was feasible, and
    the filter emergency-braked forever.

    THE COMMAND ARGUMENT WAS ACCEPTED AND DISCARDED. This built
    `DrivingIntent(command='straight')` unconditionally, ignoring the `command`
    the runner passed it. Nobody noticed because every caller passed 2, which
    IS 'straight' -- the hardcode agreed with the argument by coincidence, so
    the bug was invisible until a caller wanted something else.

    One caller did. `fit_calibration.py` sweeps `cmd in (0, 1, 2)` to widen the
    calibration set; all three produced identical straight-ahead rollouts, so
    the Platt fit saw a third of the variation it was written to sample, each
    configuration triplicated.

    The anchors are clustered PER COMMAND upstream (`kmeans_plan.py` buckets on
    `gt_ego_fut_cmd` before k-means), so command 2's six anchors describe only
    trajectories that went straight -- 0.5 m of lateral endpoint spread against
    19.9 m for the full vocabulary. Hardcoding it did not merely ignore an
    argument; it discarded 97% of the candidate geometry the file contains.
    """
    from .planner.vlm_planner import INDEX_COMMAND, DrivingIntent, intent_conditioned_planner

    inner = intent_conditioned_planner(anchor_npy, dt=dt, max_accel=max_accel)

    def plan(scene: SceneRepresentation, command: int):
        # Hold current speed, with a floor so a stopped car can pull away.
        v = max(scene.ego.speed, 2.0) if speed_condition else scene.ego.speed
        name = INDEX_COMMAND.get(int(command), 'straight')
        intent = DrivingIntent(command=name, target_speed_mps=v)
        return inner(scene, intent)
    return plan


class ReactiveGTWorldModel(GTWorldModel):
    """Agents that respond to the simulated ego instead of replaying the log.

    THE PROBLEM THIS EXISTS FOR. `GTWorldModel.agents_at` re-reads annotations
    every step, so agents follow the trajectory they took around the ego that
    ACTUALLY drove. When the simulated ego does anything else -- brakes, yields,
    takes a different line -- the recording keeps agents on their original paths
    and they drive through where the ego now is.

    Measured consequence, across this whole project: every defensive layer looks
    harmful. The TTC gate raised collisions 35 -> 38 purely by slowing the ego.
    The verifier fires and averts nothing. The world model lost across three
    variants. None of those are verdicts on the components -- the metric
    penalises slowing down because the counterfactual (agents responding to a
    slower ego) was never recorded.

    This seeds agents from the log at t=0 and then propagates them under IDM
    response to the simulated ego, so the counterfactual is generated rather than
    looked up.

    WHAT IT COSTS. Agent motion is no longer ground truth -- it is a model, and a
    crude one (longitudinal response only, no lateral evasion, no inter-agent
    interaction). Trajectory realism goes down; causal validity goes up. Use
    GTWorldModel when you want to score against what really happened, this when
    you want to ask what WOULD have happened.
    """

    def __init__(self, *args, reaction=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from .planner.world_model import ReactiveWorldModel
        self._react = reaction or ReactiveWorldModel()
        self._state: list | None = None       # agents in WORLD frame
        self._t = -1.0

    def agents_at(self, t: float, ego_xy: np.ndarray, ego_yaw: float) -> list[Agent]:
        from dataclasses import replace as _replace

        # Seed once from the log, in world frame so ego motion cannot drag them.
        if self._state is None or t <= self._t:
            logged = super().agents_at(t, ego_xy, ego_yaw)
            c, s = np.cos(ego_yaw), np.sin(ego_yaw)
            R = np.array([[c, -s], [s, c]])           # ego -> world
            self._state = [_replace(a, xy=(R @ a.xy) + ego_xy, vxy=R @ a.vxy,
                                    yaw=a.yaw + ego_yaw) for a in logged]
            self._t = t
        else:
            dt = t - self._t
            self._state = self._advance(self._state, dt, ego_xy, ego_yaw)
            self._t = t

        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        R = np.array([[c, -s], [s, c]])               # world -> ego
        return [_replace(a, xy=R @ (a.xy - ego_xy), vxy=R @ a.vxy,
                         yaw=a.yaw - ego_yaw) for a in self._state]

    def _advance(self, agents, dt, ego_xy, ego_yaw):
        """One IDM step in world frame, with the ego as an obstacle."""
        from dataclasses import replace as _replace
        from .planner.world_model import (IDM_A_MAX, IDM_B, IDM_S0, IDM_T,
                                  LANE_HALF_WIDTH)
        out = []
        for a in agents:
            v = np.asarray(a.vxy, dtype=np.float64)
            speed = float(np.linalg.norm(v))
            if speed < 0.5:
                out.append(a)                          # parked stays parked
                continue
            fwd = v / speed
            lat = np.array([-fwd[1], fwd[0]])
            rel = np.asarray(ego_xy, dtype=np.float64) - np.asarray(a.xy, float)
            gap, offset = float(rel @ fwd), abs(float(rel @ lat))

            new_speed = speed
            if gap > 0.0 and offset <= LANE_HALF_WIDTH:
                s_star = IDM_S0 + max(0.0, speed * IDM_T)
                decel = IDM_A_MAX * (s_star / max(gap, 0.5)) ** 2
                new_speed = max(0.0, speed - min(decel, IDM_B * 3.0) * dt)
            nv = fwd * new_speed
            out.append(_replace(a, xy=np.asarray(a.xy, float) + nv * dt, vxy=nv))
        return out


class LivePerceptionWorldModel(GTWorldModel):
    """GTWorldModel with agents from a real detector instead of annotations.

    Everything else is unchanged -- same route, same free space, same ego
    dynamics -- so a difference in closed-loop metrics is attributable to
    perception and nothing else.

    Free space still comes from the logged corridor rather than FlashOcc. That
    is a deliberate partial substitution: replacing both branches at once would
    leave a metric difference unattributable between them, and the object branch
    is the one with saved output covering all ten scenes.
    """

    #: these boxes ARE the detections the covariance curve was fitted to, so the
    #: tracker should use it. Inherited `measurement_noise` declares the
    #: oracle's 0.1 m, which was wrong here and silently applied until measured.
    detector_grade = True

    #: 'bevformer' -- the detection submission this arm has always replayed. 10
    #: scored classes, no identity, so the tracker's association precondition is
    #: unmet and `LiveDetectionAdapter` has to synthesise ids.
    #: 'sparse4d'  -- Sparse4D v3's tracking export, which carries the real
    #: instance-bank ids the architecture was designed around (AMOTA 0.627), at
    #: the cost of the 3 detection-only classes it does not submit.
    SOURCES = {
        'bevformer': ('results_mini_train.json', 'results_mini_val.json'),
        'sparse4d': ('sparse4d_track_mini.json',),
    }

    def __init__(self, *args, detections=None, score_thr: float = 0.25,
                 associate: bool = True, source: str = 'bevformer',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from .live_adapter import DATA, LiveDetectionAdapter, load_detections
        if detections is None:
            files = self.SOURCES.get(source)
            if files is None:
                raise ValueError(f'unknown source {source!r}; '
                                 f'expected one of {sorted(self.SOURCES)}')
            detections = load_detections(*(DATA / f for f in files))
            if not detections:
                raise FileNotFoundError(
                    f'no detections for source {source!r} under {DATA}. '
                    f'For sparse4d, export them first:\n'
                    f'  cd sparse4d_vldrive && PYTORCH_ENABLE_MPS_FALLBACK=1 '
                    f'PYTHONPATH=. python sparse4d_vl/tools/eval_track.py '
                    f'--export ../e2e_pipeline/data/sparse4d_track_mini.json')
        self.source = source
        self.adapter = LiveDetectionAdapter(self.nusc, detections,
                                            score_thr=score_thr,
                                            associate=associate)
        self.divergence: list = []

    def agents_at(self, t: float, ego_xy: np.ndarray, ego_yaw: float) -> list[Agent]:
        fr = self._frame(t)
        # Detections were computed at the LOGGED pose; record how far the
        # simulated ego has drifted from it, so the caveat is measured.
        self.divergence.append(float(np.linalg.norm(
            np.asarray(ego_xy, dtype=np.float64) - fr['xy'])))
        return self.adapter.agents_at(fr['token'], ego_xy, ego_yaw)


class FlashOccWorldModel(GTWorldModel):
    """GT objects, FlashOcc free space -- the dense branch isolated.

    Deliberately partial. Substituting both branches at once would leave any
    metric difference unattributable between them, which is the mistake §17 made
    in reverse and §19 spent two sections undoing. Objects stay ground truth so
    the delta measured here belongs to occupancy alone.
    """

    def __init__(self, *args, occ_cache=None, veto_unknown: bool = True,
                 occlusion: str = 'raycast', memory_frames: int = 10,
                 **kwargs) -> None:
        """`occlusion` selects how `unknown` is defined.

        'raycast'  -- the geometric shadow, recomputed per frame and carrying no
                      memory. Kept so the comparison is reproducible.
        'temporal' -- cells not observed within `memory_frames`, accumulated in
                      the world frame, so unknown shrinks as the ego drives.
        'none'     -- no unknown at all.
        """
        super().__init__(*args, **kwargs)
        from pathlib import Path as _P
        from .freespace import FreeSpaceExtractor, GridConfig
        from .live_adapter import FlashOccFreeSpaceAdapter
        cache = occ_cache or (_P(__file__).resolve().parents[1] /
                              'Occupancy/FlashOcc/occ_outputs/occ_cache.npz')
        # FlashOcc's NATIVE grid, not GTWorldModel's. The corridor uses
        # x[-20,60] y[-30,30] -> (200,150,16); FlashOcc emits x/y[-40,40] ->
        # (200,200,16). Reusing the corridor's grid raised a shape error rather
        # than silently misaligning, which is the good outcome -- a resample
        # would have shifted every voxel by 10 m in y with no error at all.
        occ_grid = GridConfig(x=(-40.0, 40.0, 0.4), y=(-40.0, 40.0, 0.4),
                              z=(-1.0, 5.4, 0.4))
        self.occ = FlashOccFreeSpaceAdapter(
            cache, occ_grid, FreeSpaceExtractor(occ_grid),
            occlusion=(occlusion == 'raycast'), veto_unknown=veto_unknown)
        self.occ_missing = 0
        self.occlusion = str(occlusion)
        self.memory: TemporalOcclusionMemory | None = None
        if occlusion == 'temporal':
            r = np.asarray(self._route, float)
            self.memory = TemporalOcclusionMemory(
                bounds=((float(r[:, 0].min()), float(r[:, 0].max())),
                        (float(r[:, 1].min()), float(r[:, 1].max()))),
                res=0.4, horizon_frames=memory_frames)

    def freespace_at(self, t: float, ego_xy, ego_yaw):
        fr = self._frame(t)
        fs = self.occ.freespace_at(fr['token'])
        if fs is None:
            self.occ_missing += 1
            return super().freespace_at(t, ego_xy, ego_yaw)
        if self.memory is not None:
            # Observation is stamped at the LOGGED pose, because that is where
            # the occupancy was inferred from -- crediting the simulated ego
            # with seeing from a pose it occupies but the sensor never did would
            # manufacture observations, the same error the adapter's docstring
            # warns about for boxes.
            k = int(np.clip(round(t / self.dt), 0, len(self.samples) - 1))
            self.memory.observe(fs.obstacle, fs.origin, fs.res,
                                fr['xy'], fr['yaw'], frame=k)
            fs.unknown = self.memory.unknown_ego(
                fs.obstacle.shape, fs.origin, fs.res, fr['xy'], fr['yaw'],
                frame=k)
            if self.occ.veto_unknown:
                fs.traversable = fs.traversable & ~fs.unknown
        return fs
