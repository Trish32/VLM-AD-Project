"""Closed-loop driving metrics: safety, route completion, comfort, prediction, latency.

Every tuning decision in this stack so far has been made against an unmeasured
objective — the only scored quantity was "does a red light produce STOP", which
covers a small fraction of frames. This module is the missing yardstick.

Definitions are spelled out rather than assumed, because most of these names mean
slightly different things in different papers and a metric you cannot reproduce is
worse than none:

  safety            collision = true OBB overlap between the ego footprint and an
                    agent footprint (separating-axis test, not a disc
                    approximation — discs over-report by ~30% on long vehicles).
                    Also min clearance and time-to-collision violations.

  route completion  arc length of the reference route actually covered, measured
                    by projecting each ego pose onto the route polyline. Using
                    projection rather than nearest-waypoint means a car that
                    stops early scores the distance it truly made good, and one
                    that drifts sideways does not earn credit for it.

  comfort           longitudinal accel, lateral accel (v^2 * curvature) and jerk.
                    Reported as RMS plus threshold violations; RMS alone hides a
                    single violent manoeuvre inside an otherwise smooth run.

  prediction        ADE/FDE of each agent forecast against what the agent actually
                    did, which is only computable in closed loop because the
                    future is in the log.

  latency           per-stage wall clock, reported p50/p95. Means hide the tail,
                    and the tail is what breaks a control loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import ego_footprint_corners

# Comfort thresholds. ISO 2631 territory for ride comfort; the lateral figure is
# the one passengers notice first.
COMFORT_LONG_ACCEL = 2.0      # m/s^2
COMFORT_LAT_ACCEL = 3.0       # m/s^2
COMFORT_JERK = 2.0            # m/s^3
TTC_THRESHOLD = 1.5           # s — below this counts as a near-miss


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def obb_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two convex quads overlap, by the separating-axis theorem.

    Both footprints are rectangles, so it suffices to test the four edge normals.
    Exact where a circumscribed-disc test is not: two 4.6 x 1.8 m cars sitting
    side by side 2.2 m apart do not touch, but their circumscribed discs (r~2.5 m)
    do — a disc test would report a collision that never happened.
    """
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            n = np.linalg.norm(axis)
            if n < 1e-12:
                continue
            axis /= n
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() - 1e-9 or pb.max() < pa.min() - 1e-9:
                return False          # a separating axis exists -> disjoint
    return True


def polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate gap between two convex quads (0.0 when overlapping).

    Point-to-edge minimum over both directions. Good to well under the 0.1 m that
    any clearance threshold cares about, and dependency-free.
    """
    if obb_overlap(a, b):
        return 0.0

    def _pt_seg(p, s0, s1):
        d = s1 - s0
        L2 = float(d @ d)
        t = 0.0 if L2 < 1e-12 else float(np.clip((p - s0) @ d / L2, 0.0, 1.0))
        return float(np.linalg.norm(p - (s0 + t * d)))

    best = np.inf
    for src, dst in ((a, b), (b, a)):
        for p in src:
            for i in range(len(dst)):
                best = min(best, _pt_seg(p, dst[i], dst[(i + 1) % len(dst)]))
    return float(best)


# ---------------------------------------------------------------------------
# Rollout record
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One closed-loop step. Everything the metrics need, nothing derived."""

    t: float
    ego_xy: np.ndarray                    # (2,) world
    ego_yaw: float
    ego_v: float
    accel: float                          # commanded m/s^2
    steer: float                          # commanded rad
    decision: str = ""
    agent_boxes: list = field(default_factory=list)   # [(xy, yaw, l, w, track_id)]
    predictions: dict = field(default_factory=dict)   # track_id -> (T,2) world
    latency_ms: dict = field(default_factory=dict)    # stage -> ms
    emergency: bool = False
    off_route_m: float = 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _ego_poly(rec: StepRecord, length: float, width: float) -> np.ndarray:
    return ego_footprint_corners(rec.ego_xy, rec.ego_yaw, length, width)


def _agent_poly(box) -> np.ndarray:
    xy, yaw, l, w = box[0], box[1], box[2], box[3]
    return ego_footprint_corners(np.asarray(xy, float), float(yaw),
                                 float(l), float(w))


# Contact behind this bearing from the ego's heading is a rear impact.
REAR_ARC_RAD = 2.0944          # +-120 deg from forward


def _ego_at_fault(rec, box) -> bool:
    """Did the EGO cause this contact, or was it run into?

    An undifferentiated collision count makes every defensive layer look
    harmful: braking invites a rear-end, the rear-end scores identically to
    driving into a wall, and anything trained on that learns not to brake. The
    same confusion shows up in evaluation -- caution raised the count here while
    completion collapsed, which reads as "slowing is dangerous" when it means
    "slowing gets you hit from behind".

    Attribution is geometric and deliberately simple: contact inside the ego's
    forward arc with the ego closing is the ego's doing; contact behind it, or
    with the ego not closing, is not. Struck-while-stationary is never the ego's
    fault -- there is no trajectory it could have chosen instead.
    """
    rel = np.asarray(box[0], dtype=np.float64) - np.asarray(rec.ego_xy, float)
    rng = float(np.linalg.norm(rel))
    if rng < 1e-6:
        return True                       # coincident: no geometry to reason on
    heading = np.array([np.cos(rec.ego_yaw), np.sin(rec.ego_yaw)])
    bearing = float(np.arccos(np.clip(heading @ (rel / rng), -1.0, 1.0)))
    if bearing > REAR_ARC_RAD:
        return False                      # struck from behind
    return float(rec.ego_v) > 0.5         # forward contact only counts if moving


def safety_metrics(records: list[StepRecord], ego_length: float = 4.6,
                   ego_width: float = 1.8) -> dict:
    """Collisions, clearance and near-misses over a rollout."""
    collisions, clearances, ttc_violations = [], [], 0
    for k, rec in enumerate(records):
        ego = _ego_poly(rec, ego_length, ego_width)
        step_min = np.inf
        for box in rec.agent_boxes:
            ap = _agent_poly(box)
            d = polygon_distance(ego, ap)
            step_min = min(step_min, d)
            if d <= 0.0:
                collisions.append({'step': k, 't': rec.t,
                                   'track_id': box[4] if len(box) > 4 else None,
                                   'ego_at_fault': _ego_at_fault(rec, box)})
            # TTC along the closing direction, only meaningful when approaching.
            rel = np.asarray(box[0], float) - rec.ego_xy
            rng = float(np.linalg.norm(rel))
            if rng > 1e-3 and rec.ego_v > 0.1:
                heading = np.array([np.cos(rec.ego_yaw), np.sin(rec.ego_yaw)])
                closing = float(rec.ego_v * (heading @ (rel / rng)))
                if closing > 0.1 and (d / closing) < TTC_THRESHOLD:
                    ttc_violations += 1
        if np.isfinite(step_min):
            clearances.append(step_min)

    return {
        'collision': len(collisions) > 0,
        'n_collision_steps': len(collisions),
        # Split by attribution: an undifferentiated count makes every defensive
        # layer look harmful, because braking invites a rear-end that scores the
        # same as driving into a wall.
        'n_collision_steps_ego_fault':
            sum(1 for c in collisions if c.get('ego_at_fault')),
        'n_collision_steps_other_fault':
            sum(1 for c in collisions if not c.get('ego_at_fault')),
        'collisions': collisions[:20],
        'min_clearance_m': float(min(clearances)) if clearances else None,
        'mean_clearance_m': float(np.mean(clearances)) if clearances else None,
        'ttc_violations': ttc_violations,
        'emergency_brakes': sum(r.emergency for r in records),
    }


# Shorter than this and the logged "route" is parking jitter, not a route.
MIN_ROUTE_M = 5.0


def route_completion(records: list[StepRecord], route: np.ndarray) -> dict:
    """Fraction of the reference route's arc length actually made good.

    Progress is the furthest projection onto the polyline, not the final pose's
    projection — a vehicle that overshoots then reverses should not lose credit
    for ground it covered. Lateral deviation is tracked separately so "completed
    the route" cannot be claimed by a run that cut every corner.

    A scene where the ego never drove has no route to complete, and scoring one
    anyway produces nonsense: nuScenes scene-0553 is a parked car whose logged
    "route" is 4 cm of GPS jitter, so a guard of `total > 0` let the ego's own
    jitter score 94.3% completion while emergency-braking on all 24 steps. Any
    route shorter than MIN_ROUTE_M is reported as undefined rather than scored.
    """
    route = np.asarray(route, dtype=np.float64)
    if len(route) < 2:
        return {'completion': None, 'route_length_m': 0.0}

    seg = np.diff(route, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])

    best_s, devs = 0.0, []
    for rec in records:
        p = rec.ego_xy
        s_here, d_here = 0.0, np.inf
        for i in range(len(seg)):
            L2 = float(seg[i] @ seg[i])
            t = 0.0 if L2 < 1e-12 else float(np.clip((p - route[i]) @ seg[i] / L2,
                                                     0.0, 1.0))
            proj = route[i] + t * seg[i]
            d = float(np.linalg.norm(p - proj))
            if d < d_here:
                d_here, s_here = d, float(cum[i] + t * seg_len[i])
        best_s = max(best_s, s_here)
        devs.append(d_here)

    return {
        'completion': float(best_s / total) if total >= MIN_ROUTE_M else None,
        'degenerate_route': total < MIN_ROUTE_M,
        'progress_m': float(best_s),
        'route_length_m': total,
        'mean_lateral_dev_m': float(np.mean(devs)) if devs else None,
        'max_lateral_dev_m': float(np.max(devs)) if devs else None,
    }


def comfort_metrics(records: list[StepRecord], dt: float) -> dict:
    """Longitudinal / lateral acceleration and jerk, with violation counts."""
    if len(records) < 3:
        return {'insufficient_data': True}

    v = np.array([r.ego_v for r in records])
    a_long = np.diff(v) / dt
    jerk = np.diff(a_long) / dt

    # Lateral accel from the realised path curvature (Menger), not the command —
    # what the occupants feel is the trajectory, not the steering request.
    xy = np.array([r.ego_xy for r in records])
    a_lat = []
    for i in range(1, len(xy) - 1):
        a, b, c = xy[i - 1], xy[i], xy[i + 1]
        ab, bc, ca = (np.linalg.norm(b - a), np.linalg.norm(c - b),
                      np.linalg.norm(a - c))
        denom = ab * bc * ca
        if denom < 1e-6:
            a_lat.append(0.0)
            continue
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        kappa = abs(2.0 * cross) / denom
        a_lat.append(float(v[i] ** 2 * kappa))
    a_lat = np.array(a_lat) if a_lat else np.zeros(1)

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0

    return {
        'long_accel_rms': rms(a_long), 'long_accel_max': float(np.abs(a_long).max()),
        'lat_accel_rms': rms(a_lat), 'lat_accel_max': float(a_lat.max()),
        'jerk_rms': rms(jerk), 'jerk_max': float(np.abs(jerk).max()),
        'long_violations': int((np.abs(a_long) > COMFORT_LONG_ACCEL).sum()),
        'lat_violations': int((a_lat > COMFORT_LAT_ACCEL).sum()),
        'jerk_violations': int((np.abs(jerk) > COMFORT_JERK).sum()),
    }


def prediction_metrics(records: list[StepRecord], dt: float) -> dict:
    """ADE / FDE of agent forecasts against what those agents actually did.

    Only computable in closed loop, where the realised future is in the log. A
    prediction is scored only where the agent is still tracked at the horizon, so
    an agent leaving the scene does not count as a prediction error.
    """
    ades, fdes, scored, skipped = [], [], 0, 0
    by_t = [{b[4]: np.asarray(b[0], float)
             for b in r.agent_boxes if len(b) > 4} for r in records]

    for k, rec in enumerate(records):
        for tid, traj in rec.predictions.items():
            traj = np.asarray(traj, dtype=np.float64)
            errs = []
            for h in range(len(traj)):
                j = k + h + 1
                if j >= len(records) or tid not in by_t[j]:
                    break
                errs.append(float(np.linalg.norm(traj[h] - by_t[j][tid])))
            if not errs:
                skipped += 1
                continue
            ades.append(float(np.mean(errs)))
            fdes.append(errs[-1])
            scored += 1

    return {
        'minADE_m': float(np.mean(ades)) if ades else None,
        'minFDE_m': float(np.mean(fdes)) if fdes else None,
        'n_scored': scored, 'n_unscoreable': skipped,
    }


def latency_metrics(records: list[StepRecord]) -> dict:
    """Per-stage p50/p95. The tail is what breaks a control loop, not the mean."""
    stages: dict[str, list[float]] = {}
    for rec in records:
        for k, v in rec.latency_ms.items():
            stages.setdefault(k, []).append(float(v))
    out = {}
    for k, v in stages.items():
        arr = np.array(v)
        out[k] = {'p50_ms': float(np.percentile(arr, 50)),
                  'p95_ms': float(np.percentile(arr, 95)),
                  'max_ms': float(arr.max()), 'n': len(arr)}
    if stages:
        totals = np.array([sum(r.latency_ms.values()) for r in records
                           if r.latency_ms])
        if len(totals):
            out['total'] = {'p50_ms': float(np.percentile(totals, 50)),
                            'p95_ms': float(np.percentile(totals, 95)),
                            'max_ms': float(totals.max()),
                            'hz_at_p50': float(1000.0 / np.percentile(totals, 50))}
    return out


def evaluate(records: list[StepRecord], route: np.ndarray, dt: float,
             ego_length: float = 4.6, ego_width: float = 1.8) -> dict:
    """All five metric families over one rollout."""
    return {
        'n_steps': len(records),
        'duration_s': len(records) * dt,
        'safety': safety_metrics(records, ego_length, ego_width),
        'route': route_completion(records, route),
        'comfort': comfort_metrics(records, dt),
        'prediction': prediction_metrics(records, dt),
        'latency': latency_metrics(records),
    }


def format_report(m: dict) -> str:
    """Human-readable summary."""
    s, r, c = m['safety'], m['route'], m['comfort']
    p, lat = m['prediction'], m['latency']
    L = [f"steps {m['n_steps']}  ({m['duration_s']:.1f} s)", '',
         'SAFETY', f"  collision           : {'YES' if s['collision'] else 'no'}"
                   f" ({s['n_collision_steps']} steps)",
         f"  min clearance       : "
         + (f"{s['min_clearance_m']:.2f} m" if s['min_clearance_m'] is not None else 'n/a'),
         f"  TTC < {TTC_THRESHOLD}s violations : {s['ttc_violations']}",
         f"  emergency brakes    : {s['emergency_brakes']}", '',
         'ROUTE']
    if r.get('completion') is not None:
        L += [f"  completion          : {r['completion']:.1%} "
              f"({r['progress_m']:.1f} / {r['route_length_m']:.1f} m)",
              f"  lateral dev mean/max: {r['mean_lateral_dev_m']:.2f} / "
              f"{r['max_lateral_dev_m']:.2f} m"]
    L += ['', 'COMFORT']
    if not c.get('insufficient_data'):
        L += [f"  long accel rms/max  : {c['long_accel_rms']:.2f} / "
              f"{c['long_accel_max']:.2f} m/s^2  ({c['long_violations']} over)",
              f"  lat  accel rms/max  : {c['lat_accel_rms']:.2f} / "
              f"{c['lat_accel_max']:.2f} m/s^2  ({c['lat_violations']} over)",
              f"  jerk rms/max        : {c['jerk_rms']:.2f} / "
              f"{c['jerk_max']:.2f} m/s^3  ({c['jerk_violations']} over)"]
    L += ['', 'PREDICTION']
    if p.get('minADE_m') is not None:
        L += [f"  minADE / minFDE     : {p['minADE_m']:.2f} / {p['minFDE_m']:.2f} m"
              f"  (n={p['n_scored']}, {p['n_unscoreable']} unscoreable)"]
    else:
        L += ['  (no scoreable forecasts)']
    L += ['', 'LATENCY']
    for k, v in lat.items():
        L.append(f"  {k:<18s}: p50 {v['p50_ms']:7.1f}  p95 {v['p95_ms']:7.1f} ms"
                 + (f"   -> {v['hz_at_p50']:.2f} Hz" if 'hz_at_p50' in v else ''))
    return '\n'.join(L)
