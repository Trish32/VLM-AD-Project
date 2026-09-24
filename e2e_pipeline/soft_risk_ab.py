import numpy as np, os
from nuscenes import NuScenes
from e2e_pipeline.closed_loop import (ReactiveGTWorldModel, LoopConfig,
                                      ClosedLoopRunner, diffusiondrive_anchor_planner)
from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.uncertainty import RiskModel
from e2e_pipeline.world_model import ReactiveWorldModel
from e2e_pipeline.calibration import (PlattCalibrator, risk_speed_scale,
                                      scale_trajectory, EMERGENCY_RISK)

nusc = NuScenes(version='v1.0-mini',
                dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'), verbose=False)
A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
cal = PlattCalibrator(a=0.457, b=-2.333, fitted=True, n_positive=11)   # from 2a812f4
orig = SafetyFilter.__call__


def run(label, max_risk, soft):
    def cap(self, c, sc, s=None, r=None):
        res = orig(self, c, sc, s, r)
        if soft and not res.emergency and len(res.trajectory):
            raw = RiskModel(sc.ego).evaluate(np.asarray(res.trajectory, float),
                                             sc.agents, dt=0.5).total
            res.trajectory = scale_trajectory(res.trajectory,
                                              risk_speed_scale(float(cal(raw))))
        return res
    SafetyFilter.__call__ = cap
    E = O = B = 0
    comps, clears, jerks = [], [], []
    for i in range(10):
        w = ReactiveGTWorldModel(nusc, scene_idx=i)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed())
        recs, m = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                                   safety=SafetyFilter(limits=FeasibilityLimits(max_risk=max_risk)),
                                   latent_model=ReactiveWorldModel()).run(command=2)
        s = m['safety']
        E += s['n_collision_steps_ego_fault']; O += s['n_collision_steps_other_fault']
        B += s['emergency_brakes']
        if s['min_clearance_m'] is not None: clears.append(s['min_clearance_m'])
        if m['route'].get('completion') is not None: comps.append(m['route']['completion'])
        c = m['comfort']
        if not c.get('insufficient_data'): jerks.append(c['jerk_rms'])
    SafetyFilter.__call__ = orig
    print(f'  {label:34s}{E:>6}{O:>7}{np.mean(clears):>7.2f}m{B:>8}{np.mean(comps):>8.1%}{np.mean(jerks):>7.2f}')


print(f'  {"config":34s}{"EGO":>6}{"other":>7}{"clear":>8}{"brakes":>8}{"compl":>8}{"jerk":>7}')
run('hard veto @0.05 (original)', 0.05, False)
run('hard veto @0.60 (tuned)', 0.60, False)
run(f'SOFT + emergency-only @{EMERGENCY_RISK}', EMERGENCY_RISK, True)
