"""The conservative unknown-space prior, re-measured under the fitted covariance.

1aec3d4 found the prior inert: brakes, clearance and risk identical at every
prior from 0.00 to 0.10, because 0 of 6 anchor waypoints ever land in an
unknown cell. The occlusion is LATERAL -- behind buildings and parked cars --
while the six fixed anchors run straight down the observed corridor.

The covariance recalibration is a reason to re-check but not a reason to expect
a different answer, and the distinction matters. The prior combines with the
agent term as independent survival, so raising agent risk raises the total
either way; what decides whether the PRIOR contributes is purely geometric --
does a waypoint land in an unknown cell. Covariance cannot change that.

So this measures two things and keeps them separate:
  * the sweep, which should stay flat, and
  * `unknown_hit`, the fraction of planned waypoints in unknown cells, which is
    the quantity that actually explains why.

An arm forcing detector-grade covariance onto the GT world is included. It is
not a realistic configuration -- these are annotations -- but it raises ambient
risk sharply, and if the prior were merely being swamped rather than structurally
idle, that arm would show it.

Usage:
    PYTHONPATH=. python e2e_pipeline/occlusion_prior_sweep.py
"""
from __future__ import annotations

import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import (
    ClosedLoopRunner, FlashOccWorldModel, LoopConfig,
    diffusiondrive_anchor_planner)
from e2e_pipeline.planner.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.uncertainty import TrackCovarianceTracker
from e2e_pipeline.planner.world_model import ReactiveWorldModel

A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
PRIORS = (0.00, 0.02, 0.05, 0.10)


def run(nusc, prior, tracker_factory=None, n=10):
    B = E = 0
    clears, risks, hits, unk_frac = [], [], [], []
    for i in range(n):
        # veto OFF: the hard veto and the soft prior are mutually exclusive, so
        # the prior can only be measured with the veto disabled.
        w = FlashOccWorldModel(nusc, scene_idx=i, veto_unknown=False)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                         calibrate_risk=True, unknown_prior=prior,
                         veto_unknown=False)
        r = ClosedLoopRunner(
            w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
            safety=SafetyFilter(limits=FeasibilityLimits(max_risk=0.10)),
            tracker=tracker_factory() if tracker_factory else None,
            latent_model=ReactiveWorldModel())
        recs, m = r.run(command=2)
        s = m['safety']
        B += s['emergency_brakes']
        E += s['n_collision_steps_ego_fault']
        if s['min_clearance_m'] is not None:
            clears.append(s['min_clearance_m'])
        risks += [x for x in r.risk_pred if x is not None]

        # the geometric question, measured directly rather than inferred
        for k, rec in enumerate(recs):
            if rec.planned_traj is None or not len(rec.planned_traj):
                continue
            fs = w.freespace_at(k * cfg.dt, rec.ego_xy, rec.ego_yaw)
            if fs is None or fs.unknown is None:
                continue
            unk_frac.append(float(fs.unknown.mean()))
            hits.append(float(np.mean(fs.unknown_at(np.asarray(rec.planned_traj,
                                                               float)))))
    return {
        'brakes': B, 'ego': E,
        'clear': float(np.mean(clears)) if clears else float('nan'),
        'risk': float(np.mean(risks)) if risks else float('nan'),
        'risk_p90': float(np.percentile(risks, 90)) if risks else float('nan'),
        'unknown_map': float(np.mean(unk_frac)) if unk_frac else float('nan'),
        'unknown_hit': float(np.mean(hits)) if hits else float('nan'),
    }


def _table(nusc, label, tracker_factory):
    print(f'\n  === {label} ===')
    print(f'  {"prior":>7}{"brakes":>8}{"EGO":>5}{"clear":>8}{"mean risk":>11}'
          f'{"risk p90":>10}{"map unknown":>13}{"waypoints in unknown":>22}')
    for p in PRIORS:
        r = run(nusc, p, tracker_factory)
        print(f'  {p:>7.2f}{r["brakes"]:>8}{r["ego"]:>5}{r["clear"]:>7.2f}m'
              f'{r["risk"]:>11.4f}{r["risk_p90"]:>10.4f}'
              f'{r["unknown_map"]:>13.1%}{r["unknown_hit"]:>22.1%}')


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    _table(nusc, 'GT-declared covariance (0.1 m / 0.2 m/s)', None)
    _table(nusc, 'detector-grade covariance forced on (stress arm)',
           lambda: TrackCovarianceTracker(calibrated_noise=True))


if __name__ == '__main__':
    main()
