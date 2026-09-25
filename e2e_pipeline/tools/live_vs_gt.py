"""Closed-loop comparison: GT oracle vs live detector output.

Usage:
    PYTHONPATH=. python e2e_pipeline/live_vs_gt.py
"""
import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import (ClosedLoopRunner, GTWorldModel,
                                      LivePerceptionWorldModel, LoopConfig,
                                      diffusiondrive_anchor_planner)
from e2e_pipeline.planner.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.planner.world_model import ReactiveWorldModel

A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'


def run(nusc, label, WorldCls, **kw):
    E = O = B = 0
    comps, clears, jerks, divs, nag = [], [], [], [], []
    for i in range(10):
        w = WorldCls(nusc, scene_idx=i, **kw)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                         calibrate_risk=True)
        r = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                             safety=SafetyFilter(limits=FeasibilityLimits(max_risk=0.10)),
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
        nag += [len(rec.agent_boxes) for rec in recs]
        divs += getattr(w, 'divergence', [])
    d = f'{np.mean(divs):.1f}m' if divs else '-'
    print(f'  {label:22s}{E:>5}{O:>7}{np.mean(clears):>7.2f}m{B:>8}'
          f'{np.mean(comps):>8.1%}{np.mean(jerks):>7.2f}{np.mean(nag):>8.1f}{d:>9}')


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    print(f'  {"world model":22s}{"EGO":>5}{"other":>7}{"clear":>8}{"brakes":>8}'
          f'{"compl":>8}{"jerk":>7}{"agents":>8}{"diverg":>9}')
    run(nusc, 'GT oracle', GTWorldModel)
    run(nusc, 'LIVE detector @0.25', LivePerceptionWorldModel)
    run(nusc, 'LIVE detector @0.40', LivePerceptionWorldModel, score_thr=0.40)


if __name__ == '__main__':
    main()
