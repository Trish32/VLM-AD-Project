"""Sweep the safety filter's max_risk against fault-split safety.

Reproduces the finding in RESULT.md: the risk gate was the binding
constraint on the whole pipeline, and relaxing it improves every metric
including the collisions it exists to prevent.

Usage:
    PYTHONPATH=. python e2e_pipeline/risk_sweep.py
"""
import numpy as np, os
from nuscenes import NuScenes
from e2e_pipeline.closed_loop import (ReactiveGTWorldModel, LoopConfig,
                                      ClosedLoopRunner, diffusiondrive_anchor_planner)
from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter

nusc = NuScenes(version='v1.0-mini',
                dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'), verbose=False)
A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
print('  max_risk  brakes  EGO-FAULT  other  completion  clearance')
for mr in (0.05, 0.15, 0.30, 0.60, 1.01):
    B = E = O = 0
    comps, clears = [], []
    for i in range(10):
        w = ReactiveGTWorldModel(nusc, scene_idx=i)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed())
        filt = SafetyFilter(limits=FeasibilityLimits(max_risk=mr))
        recs, m = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt),
                                   cfg, safety=filt).run(command=2)
        s = m['safety']
        B += s['emergency_brakes']
        E += s['n_collision_steps_ego_fault']
        O += s['n_collision_steps_other_fault']
        if m['route'].get('completion') is not None:
            comps.append(m['route']['completion'])
        mc = s['min_clearance_m']
        if mc is not None:
            clears.append(mc)
    print(f'   {mr:5.2f}      {B:3d}       {E:3d}     {O:3d}      {np.mean(comps):5.1%}      {np.mean(clears):.2f}m')
