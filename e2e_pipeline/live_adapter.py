"""Live perception adapters: real detector output in place of GT annotations.

`GTWorldModel` hands the planner nuScenes annotations -- perfect boxes, perfect
velocities, perfect recall. Every closed-loop number in this project was
produced against that oracle, so none of them says anything about the pipeline's
behaviour under perception it would actually have.

This substitutes a real detector's output, keyed by sample token, and changes
nothing else. The gap between the two runs is the cost of perception.

WHAT "LIVE" MEANS HERE, PRECISELY. The detector is not re-run per step; its
output was computed once over the dataset and is replayed by token. That is a
fair substitute for the perception INPUT -- these are genuine BEVFormer
detections with their real misses, false positives and velocity errors -- but it
is not a closed-loop perception system, and two limits follow:

  * Detections were computed at the LOGGED ego pose. The closed loop simulates
    ego motion and diverges, so boxes describe the scene from a viewpoint the
    ego no longer occupies. The error grows with divergence, exactly as in the
    visualiser, and `divergence_m` is reported so it is visible rather than
    assumed away.
  * There is no tracking across a diverging trajectory. Track IDs come from the
    detector's own association on the logged sequence.

Neither can be fixed without running perception inside the loop on synthesised
sensor data, which nuScenes cannot provide -- there is no camera image from a
pose the car never occupied.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scene import Agent

DATA = Path(__file__).resolve().parent / 'data'


def load_detections(*paths) -> dict:
    """Merge nuScenes-format submission files into {token: [box dicts]}."""
    out = {}
    for p in paths:
        p = Path(p)
        if p.exists():
            out.update(json.loads(p.read_text())['results'])
    return out


class LiveDetectionAdapter:
    """Detector output in ego frame, in place of ground-truth annotations.

    Drop-in for `GTWorldModel.agents_at`: same signature, same frame convention,
    same `Agent` type. Only the contents differ -- and that difference is the
    measurement.
    """

    def __init__(self, nusc, detections: dict, score_thr: float = 0.25,
                 max_range: float = 60.0) -> None:
        self.nusc = nusc
        self.det = detections
        self.score_thr = float(score_thr)
        self.max_range = float(max_range)
        self.missing = 0            # tokens with no detector output

    def agents_at(self, token: str, ego_xy: np.ndarray, ego_yaw: float
                  ) -> list[Agent]:
        from pyquaternion import Quaternion
        boxes = self.det.get(token)
        if boxes is None:
            self.missing += 1
            return []

        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        R = np.array([[c, -s], [s, c]])           # world -> ego
        out = []
        for b in boxes:
            if float(b.get('detection_score', 1.0)) < self.score_thr:
                continue
            p = R @ (np.asarray(b['translation'][:2], dtype=np.float64) - ego_xy)
            if np.linalg.norm(p) > self.max_range:
                continue
            v = np.asarray(list(b.get('velocity') or (0.0, 0.0)), dtype=np.float64)
            if np.isnan(v).any():
                v = np.zeros(2)
            w_, l_, h_ = b['size']
            yaw_w = Quaternion(b['rotation']).yaw_pitch_roll[0]
            # Track id from the detector's own association where present, else a
            # stable hash of the box identity -- the Kalman tracker needs
            # SOMETHING consistent, and a per-frame random id would make every
            # agent look brand new and reset its covariance every step.
            tid = b.get('tracking_id') or abs(hash(
                (token, round(float(b['translation'][0]), 1),
                 round(float(b['translation'][1]), 1)))) % 100000
            out.append(Agent(track_id=int(tid) if str(tid).isdigit() else
                             abs(hash(tid)) % 100000,
                             xy=p, yaw=float(yaw_w - ego_yaw),
                             lwh=np.array([l_, w_, h_], dtype=np.float64),
                             vxy=R @ v,
                             score=float(b.get('detection_score', 1.0)),
                             label=0))
        return out


class FlashOccFreeSpaceAdapter:
    """Cached FlashOcc occupancy in place of the synthetic corridor.

    `GTWorldModel.freespace_at` synthesises a band around the logged route. That
    is a stand-in for occupancy, not occupancy -- it knows the road because it
    knows where the car went. This replays real FlashOcc output, reduced by the
    same `FreeSpaceExtractor` the pipeline already uses, so the difference is
    the dense branch's error and nothing else.

    SAME CAVEAT AS THE DETECTOR. The volume was inferred once, from the logged
    pose. A simulated ego that has drifted is reading occupancy for a viewpoint
    it no longer occupies -- and unlike boxes, a voxel grid cannot be
    transformed into the new frame without resampling something that was never
    observed. So this is exact only at zero divergence, which is why the
    ablation reports both ego modes.
    """

    def __init__(self, cache_path, grid, extractor, occlusion: bool = True,
                 veto_unknown: bool = True) -> None:
        self.cache = np.load(str(cache_path))
        self.grid = grid
        self.extractor = extractor
        self.missing = 0
        self.occlusion = bool(occlusion)
        self.veto_unknown = bool(veto_unknown)

    def freespace_at(self, token: str):
        if token not in self.cache:
            self.missing += 1
            return None
        fs = self.extractor(self.cache[token].astype(np.int64))
        if self.occlusion:
            fs.unknown = ray_occlusion(fs.obstacle, fs.origin, fs.res)
            if self.veto_unknown:
                # HARD: unknown is not drivable. This is what freespace.py has
                # always said and what no caller ever exercised, because nothing
                # populated `unknown`.
                fs.traversable = fs.traversable & ~fs.unknown
            # SOFT (veto_unknown=False): unknown stays drivable and is priced by
            # RiskModel(unknown_prior). The two are mutually exclusive by
            # construction -- with the veto on, no surviving candidate ever
            # enters an unknown cell, so the prior has nothing to price and
            # measures exactly zero. Overlapping responsibilities again, the
            # same shape as the TTC gate against the risk gate.
        return fs


def ray_occlusion(obstacle: np.ndarray, origin, res: float,
                  n_rays: int = 720) -> np.ndarray:
    """Cells hidden behind an obstacle on the ray from the ego, marked unknown.

    Occupancy says what the sensor SAW. It cannot say what is behind a van, and
    treating that as free is the over-confidence that makes a risk number
    unbounded: a path into an occlusion scores identically to one down an
    observed empty road.

    Marking those cells unknown does not reveal what is there. It bounds how
    confident the estimate may be -- with RiskModel(unknown_prior=p) the hidden
    region carries at most p, so a missed hazard appears as elevated risk rather
    than as silence.

    Marches each ray outward in half-cell steps, which errs toward marking
    slightly too much as occluded. That is the conservative direction, and for a
    prior whose purpose is bounding confidence it is the right way to be wrong.
    """
    nx, ny = obstacle.shape
    unknown = np.zeros((nx, ny), dtype=bool)
    ex = int((0.0 - origin[0]) / res)
    ey = int((0.0 - origin[1]) / res)
    max_r = float(np.hypot(nx, ny))
    for ang in np.linspace(-np.pi, np.pi, n_rays, endpoint=False):
        dx, dy = np.cos(ang), np.sin(ang)
        blocked = False
        for r in np.arange(1.0, max_r, 0.5):
            ix, iy = int(ex + dx * r), int(ey + dy * r)
            if not (0 <= ix < nx and 0 <= iy < ny):
                break
            if blocked:
                unknown[ix, iy] = True
            elif obstacle[ix, iy]:
                blocked = True
    return unknown
