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
    """Merge nuScenes submission files into {token: [box dicts]}.

    Accepts both submission flavours. A *detection* submission carries
    `detection_name` / `detection_score` and no identity; a *tracking*
    submission carries `tracking_name` / `tracking_score` / `tracking_id`.
    Normalised to the detection field names here, with `tracking_id` preserved,
    so downstream code has one shape to read and the presence of real identity
    is the only difference.
    """
    out: dict = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for token, boxes in json.loads(p.read_text())['results'].items():
            norm = []
            for b in boxes:
                if 'detection_name' not in b and 'tracking_name' in b:
                    b = dict(b)
                    b['detection_name'] = b['tracking_name']
                    b['detection_score'] = b.get('tracking_score', 1.0)
                norm.append(b)
            out[token] = norm
    return out


class LiveDetectionAdapter:
    """Detector output in ego frame, in place of ground-truth annotations.

    Drop-in for `GTWorldModel.agents_at`: same signature, same frame convention,
    same `Agent` type. Only the contents differ -- and that difference is the
    measurement.
    """

    #: gate for nearest-neighbour association, metres. Generous relative to the
    #: 0.5 s step because the measured position error is itself 1.0-2.2 m.
    ASSOC_GATE_M = 3.0

    def __init__(self, nusc, detections: dict, score_thr: float = 0.25,
                 max_range: float = 60.0, associate: bool = True) -> None:
        self.nusc = nusc
        self.det = detections
        self.score_thr = float(score_thr)
        self.max_range = float(max_range)
        self.missing = 0            # tokens with no detector output
        # WHY THIS EXISTS. The fallback id was a hash of (token, rounded xy),
        # and the token in that tuple made every id unique to its frame --
        # measured 0.0% of ids carried to the next frame, against 97.1% under
        # GT. So the Kalman filter never ran a second update on any agent: every
        # covariance stayed at its seed R(score) forever, and anything needing
        # an agent's history (the residual fit, any id-switch count) had an
        # empty input rather than a poor one. The comment beside that hash
        # warned against exactly the behaviour it caused.
        #
        # These detections carry no tracking_id, so association has to be done
        # here. Greedy nearest-neighbour against the previous frame, constant-
        # velocity predicted, gated and class-constrained. That is a weak
        # tracker and is not claimed otherwise -- but it is the difference
        # between a filter that filters and one that does not.
        self.associate = bool(associate)
        self._tracks: dict[int, dict] = {}     # tid -> {xy, vxy, class}
        self._token_ids: dict[str, list] = {}  # token -> [tid per kept box]
        self._next_tid = 1
        self.switches = 0
        #: True once a frame supplied real `tracking_id`s, so a run can report
        #: whether it used the detector's own identity or the NN stand-in.
        self.used_real_ids = False

    def agents_at(self, token: str, ego_xy: np.ndarray, ego_yaw: float
                  ) -> list[Agent]:
        from pyquaternion import Quaternion
        boxes = self.det.get(token)
        if boxes is None:
            self.missing += 1
            return []

        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        R = np.array([[c, -s], [s, c]])           # world -> ego

        kept = []                                  # world-frame, pre-id
        for b in boxes:
            if float(b.get('detection_score', 1.0)) < self.score_thr:
                continue
            world = np.asarray(b['translation'][:2], dtype=np.float64)
            p = R @ (world - ego_xy)
            if np.linalg.norm(p) > self.max_range:
                continue
            v = np.asarray(list(b.get('velocity') or (0.0, 0.0)), dtype=np.float64)
            if np.isnan(v).any():
                v = np.zeros(2)
            kept.append((b, world, v, p))

        ids = self._ids_for(token, kept)

        out = []
        for (b, _world, v, p), tid in zip(kept, ids):
            w_, l_, h_ = b['size']
            yaw_w = Quaternion(b['rotation']).yaw_pitch_roll[0]
            out.append(Agent(track_id=int(tid),
                             xy=p, yaw=float(yaw_w - ego_yaw),
                             lwh=np.array([l_, w_, h_], dtype=np.float64),
                             vxy=R @ v,
                             score=float(b.get('detection_score', 1.0)),
                             label=0))
        return out

    def _ids_for(self, token: str, kept: list, dt: float = 0.5) -> list:
        """Stable track ids for this frame's boxes, cached per token.

        Cached because `agents_at` is called repeatedly for the same token (the
        planner, the metrics and the counterfactual all ask), and association
        must not depend on how many times it was asked.
        """
        if token in self._token_ids:
            return self._token_ids[token]

        # REAL IDENTITY WINS, and nothing here should overrule it. Sparse4D v3
        # propagates ids through its temporal instance bank, which is the
        # assumption `TrackCovarianceTracker` was built on; the nearest-neighbour
        # fallback below exists only because the BEVFormer detection submission
        # has no identity to read. Associating on top of a real tracker would
        # discard a measured AMOTA 0.627 and replace it with a 3 m gate.
        if all(b.get('tracking_id') is not None for b, _w, _v, _p in kept) and kept:
            ids = [abs(hash(b['tracking_id'])) % 1_000_000
                   for b, _w, _v, _p in kept]
            self._token_ids[token] = ids
            self.used_real_ids = True
            return ids

        if not self.associate:
            ids = [b.get('tracking_id') or abs(hash(
                (token, round(float(w[0]), 1), round(float(w[1]), 1)))) % 100000
                for b, w, _v, _p in kept]
            ids = [int(i) if str(i).isdigit() else abs(hash(i)) % 100000
                   for i in ids]
            self._token_ids[token] = ids
            return ids

        # predict every live track forward one step under constant velocity
        pred = {tid: (t['xy'] + t['vxy'] * dt, t['cls'])
                for tid, t in self._tracks.items()}
        taken: set = set()
        ids: list = [None] * len(kept)

        # greedy, most-confident detection first: a high-score box should get
        # first claim on a track rather than losing it to a marginal neighbour
        order = sorted(range(len(kept)),
                       key=lambda i: -float(kept[i][0].get('detection_score', 0.0)))
        for i in order:
            b, world, _v, _p = kept[i]
            cls = b.get('detection_name')
            best, bd = None, self.ASSOC_GATE_M
            for tid, (xy, tcls) in pred.items():
                if tid in taken or tcls != cls:
                    continue
                d = float(np.linalg.norm(xy - world))
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best = self._next_tid
                self._next_tid += 1
            else:
                taken.add(best)
            ids[i] = best
            self._tracks[best] = {'xy': world,
                                  'vxy': np.asarray(kept[i][2], float),
                                  'cls': cls}

        # drop tracks not seen this frame, so a stale one cannot claim a box
        # several seconds later at a position it drifted to
        for tid in [t for t in self._tracks if t not in ids]:
            del self._tracks[tid]

        self._token_ids[token] = ids
        return ids


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
