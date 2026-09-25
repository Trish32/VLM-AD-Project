"""Ablation: does wiring the calibration into the decision path change anything?

The Platt map was fitted (2a812f4) and validated across disjoint scenes
(abf16d8), then sat unused -- the risk model still emitted raw, 6x-inflated
probabilities and the filter still thresholded them. This measures what
connecting it does.

Usage:
    PYTHONPATH=. python e2e_pipeline/calibration_ablation.py
"""
import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import (ClosedLoopRunner, LoopConfig,
                                      ReactiveGTWorldModel,
                                      diffusiondrive_anchor_planner)
from e2e_pipeline.planner.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.planner.world_model import ReactiveWorldModel

A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'


def run(nusc, label, max_risk, calibrate, budget=0.0):
    E = O = B = 0
    comps, clears, jerks, eces = [], [], [], []
    for i in range(10):
        w = ReactiveGTWorldModel(nusc, scene_idx=i)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                         calibrate_risk=calibrate, risk_budget=budget)
        r = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                             safety=SafetyFilter(limits=FeasibilityLimits(max_risk=max_risk)),
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
        rep = r.calibration_report()
        if rep['n'] and np.isfinite(rep['ece']):
            eces.append(rep['ece'])
    ece = np.mean(eces) if eces else float('nan')
    print(f'  {label:36s}{E:>5}{O:>7}{np.mean(clears):>7.2f}m{B:>8}'
          f'{np.mean(comps):>8.1%}{np.mean(jerks):>7.2f}{ece:>8.3f}')


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    print(f'  {"config":36s}{"EGO":>5}{"other":>7}{"clear":>8}{"brakes":>8}'
          f'{"compl":>8}{"jerk":>7}{"ECE":>8}')
    run(nusc, 'raw risk, max_risk=0.60 (current)', 0.60, False)
    run(nusc, 'raw risk, max_risk=0.05 (original)', 0.05, False)
    # Calibrated risk is ~6x smaller, so the equivalent thresholds are lower.
    for mr in (0.05, 0.10, 0.20):
        run(nusc, f'CALIBRATED, max_risk={mr}', mr, True)
    run(nusc, 'CALIBRATED + budget eps=0.10', 0.85, True, 0.10)


if __name__ == '__main__':
    main()
