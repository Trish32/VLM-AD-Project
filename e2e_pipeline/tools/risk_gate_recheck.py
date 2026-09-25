"""Is max_risk = 0.10 still the right gate under the measured covariance?

0.10 was chosen in §18 against a risk model whose agent covariance was wrong in
both terms: position under-estimated 1.3-2.0x, and velocity under-estimated
twelvefold (0.2 m/s declared against 2.46 m/s measured). A threshold is only
meaningful relative to the distribution it thresholds, so recalibrating the
inputs invalidates the number even if nothing else changed.

TWO EGO MODES, AND THE SECOND IS THE ONE TO READ. §19 established that every
collision in this project traces to divergence from the logged trajectory rather
than to a planning error, so a collision count from the free-running sim mostly
measures deviation. `follow_logged_ego` pins the ego to the log, which drives
divergence to zero and leaves the gate's effect on the DECISION visible without
that confound. The free-running arm is kept because it is the one that shows
whether the gate is over-braking.

Usage:
    PYTHONPATH=. python e2e_pipeline/risk_gate_recheck.py
"""
from __future__ import annotations

import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import (
    ClosedLoopRunner, LivePerceptionWorldModel, LoopConfig,
    diffusiondrive_anchor_planner)
from e2e_pipeline.planner.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.uncertainty import TrackCovarianceTracker
from e2e_pipeline.planner.world_model import ReactiveWorldModel

A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
THRESHOLDS = (0.05, 0.10, 0.20, 0.40, 1.01)


def run(nusc, max_risk, pinned=False, tracker_factory=None, n=10):
    E = O = B = 0
    comps, clears, jerks, divs, risks = [], [], [], [], []
    for i in range(n):
        w = LivePerceptionWorldModel(nusc, scene_idx=i)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                         calibrate_risk=True, follow_logged_ego=pinned)
        r = ClosedLoopRunner(
            w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
            safety=SafetyFilter(limits=FeasibilityLimits(max_risk=max_risk)),
            tracker=tracker_factory() if tracker_factory else None,
            latent_model=ReactiveWorldModel())
        recs, m = r.run(command=2)
        s = m['safety']
        E += s['n_collision_steps_ego_fault']
        O += s['n_collision_steps_other_fault']
        B += s['emergency_brakes']
        if s['min_clearance_m'] is not None:
            clears.append(s['min_clearance_m'])
        if m['route'].get('completion') is not None:
            comps.append(m['route']['completion'])
        c = m['comfort']
        if not c.get('insufficient_data'):
            jerks.append(c['jerk_rms'])
        divs += getattr(w, 'divergence', [])
        risks += [x for x in r.risk_pred if x is not None]
    return {
        'ego': E, 'other': O, 'brakes': B,
        'clear': float(np.mean(clears)) if clears else float('nan'),
        'compl': float(np.mean(comps)) if comps else float('nan'),
        'jerk': float(np.mean(jerks)) if jerks else float('nan'),
        'div': float(np.mean(divs)) if divs else float('nan'),
        'risk_p50': float(np.median(risks)) if risks else float('nan'),
        'risk_p90': float(np.percentile(risks, 90)) if risks else float('nan'),
    }


def _table(nusc, pinned):
    print(f'\n  === {"PINNED to logged ego" if pinned else "FREE-RUNNING sim"} ===')
    print(f'  {"max_risk":>9}{"EGO":>5}{"other":>7}{"brakes":>8}{"clear":>8}'
          f'{"compl":>8}{"jerk":>7}{"diverg":>8}{"risk p50":>10}{"risk p90":>10}')
    for mr in THRESHOLDS:
        r = run(nusc, mr, pinned=pinned)
        print(f'  {mr:>9.2f}{r["ego"]:>5}{r["other"]:>7}{r["brakes"]:>8}'
              f'{r["clear"]:>7.2f}m{r["compl"]:>8.1%}{r["jerk"]:>7.2f}'
              f'{r["div"]:>7.1f}m{r["risk_p50"]:>10.4f}{r["risk_p90"]:>10.4f}')


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    _table(nusc, pinned=True)
    _table(nusc, pinned=False)

    # Does the gate's behaviour depend on the covariance fix, or would 0.10 have
    # looked the same before? Same threshold, old noise model.
    print('\n  === covariance A/B at max_risk = 0.10, free-running ===')
    print(f'  {"covariance":>22}{"EGO":>5}{"other":>7}{"brakes":>8}{"clear":>8}'
          f'{"compl":>8}{"risk p50":>10}{"risk p90":>10}')
    arms = {
        'old (0.1 m / 0.2 m/s)': lambda: TrackCovarianceTracker(
            calibrated_noise=False, pos_noise=0.1, vel_noise=0.2),
        'measured': lambda: TrackCovarianceTracker(calibrated_noise=True),
    }
    for name, mk in arms.items():
        r = run(nusc, 0.10, pinned=False, tracker_factory=mk)
        print(f'  {name:>22}{r["ego"]:>5}{r["other"]:>7}{r["brakes"]:>8}'
              f'{r["clear"]:>7.2f}m{r["compl"]:>8.1%}'
              f'{r["risk_p50"]:>10.4f}{r["risk_p90"]:>10.4f}')


if __name__ == '__main__':
    main()
