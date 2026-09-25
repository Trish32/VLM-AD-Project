"""Re-baseline every safety layer against an unjammed pipeline.

All earlier measurements used max_risk=0.05, which made the ego brake constantly
and sit stationary -- so every layer was evaluated against a pipeline that did
not drive. This re-runs them at 0.60 with separated safety / progress / comfort
/ intervention reporting.

Usage:
    PYTHONPATH=. python e2e_pipeline/rebaseline.py
"""
import numpy as np, os
from nuscenes import NuScenes
from e2e_pipeline.closed_loop import (
    ReactiveGTWorldModel, LoopConfig, ClosedLoopRunner,
    diffusiondrive_anchor_planner)
from e2e_pipeline.planner.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.planner.world_model import ReactiveWorldModel

nusc = NuScenes(version='v1.0-mini',
                dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'), verbose=False)
A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
MR = 0.60

CONFIGS = [
    ('baseline',      dict()),
    ('+verifier',     dict(use_verifier=True)),
    ('+ttc-gate',     dict(use_structured=True)),
    ('+shadow',       dict(use_shadow=True)),
    ('+world-model',  dict(use_world_model=True)),
    ('+all',          dict(use_verifier=True, use_structured=True,
                           use_shadow=True, use_world_model=True)),
]

print(f'  max_risk={MR}, reactive agents, 10 scenes x 20 steps')
print(f'  {"config":14s}{"EGO-FLT":>8}{"other":>7}{"clear":>8}{"brakes":>8}{"compl":>8}{"jerk":>7}{"interv":>8}')
for name, kw in CONFIGS:
    E = O = B = 0
    clears, comps, jerks, interv = [], [], [], 0
    for i in range(10):
        w = ReactiveGTWorldModel(nusc, scene_idx=i)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(), **kw)
        filt = SafetyFilter(limits=FeasibilityLimits(max_risk=MR))
        r = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                             safety=filt, latent_model=ReactiveWorldModel())
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
        interv += r.verifier_fired + r.structured_fired + sum(r.shadow_fired.values())
    print(f'  {name:14s}{E:>8}{O:>7}{np.mean(clears):>7.2f}m{B:>8}{np.mean(comps):>7.1%}'
          f'{np.mean(jerks):>7.2f}{interv:>8}')
