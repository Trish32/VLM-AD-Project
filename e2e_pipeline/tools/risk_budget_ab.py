import numpy as np, os
from nuscenes import NuScenes
from e2e_pipeline.closed_loop import (
    ReactiveGTWorldModel, LoopConfig, ClosedLoopRunner,
    diffusiondrive_anchor_planner)
from e2e_pipeline.planner.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.uncertainty import RiskModel
from e2e_pipeline.planner.world_model import ReactiveWorldModel
from e2e_pipeline.calibration.calibration import (
    PlattCalibrator, risk_speed_scale, scale_trajectory,
    scale_for_risk_budget)

nusc = NuScenes(version='v1.0-mini',
                dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'), verbose=False)
A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
cal = PlattCalibrator(a=0.457, b=-2.333, fitted=True, n_positive=11)
orig = SafetyFilter.__call__


def run(label, mode, budget=None):
    def cap(self, c, sc, s=None, r=None):
        res = orig(self, c, sc, s, r)
        if mode and not res.emergency and len(res.trajectory):
            rm = RiskModel(sc.ego)
            if mode == 'curve':
                raw = rm.evaluate(np.asarray(res.trajectory, float), sc.agents, dt=0.5).total
                sc_ = risk_speed_scale(float(cal(raw)))
            else:
                sc_ = scale_for_risk_budget(
                    res.trajectory, sc, budget, cal,
                    lambda t: rm.evaluate(np.asarray(t, float), sc.agents, dt=0.5).total)
            res.trajectory = scale_trajectory(res.trajectory, sc_)
        return res
    SafetyFilter.__call__ = cap
    E = O = B = 0; comps, clears, jerks = [], [], []
    for i in range(10):
        w = ReactiveGTWorldModel(nusc, scene_idx=i)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed())
        recs, m = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                                   safety=SafetyFilter(limits=FeasibilityLimits(max_risk=0.85)),
                                   latent_model=ReactiveWorldModel()).run(command=2)
        s = m['safety']
        E += s['n_collision_steps_ego_fault']; O += s['n_collision_steps_other_fault']
        B += s['emergency_brakes']
        if s['min_clearance_m'] is not None: clears.append(s['min_clearance_m'])
        if m['route'].get('completion') is not None: comps.append(m['route']['completion'])
        c = m['comfort']
        if not c.get('insufficient_data'): jerks.append(c['jerk_rms'])
    SafetyFilter.__call__ = orig
    print(f'  {label:32s}{E:>5}{O:>7}{np.mean(clears):>7.2f}m{B:>8}{np.mean(comps):>8.1%}{np.mean(jerks):>7.2f}')


print(f'  {"config":32s}{"EGO":>5}{"other":>7}{"clear":>8}{"brakes":>8}{"compl":>8}{"jerk":>7}')
run('no response (filter only)', None)
run('hand-set curve', 'curve')
for eps in (0.02, 0.05, 0.10, 0.20):
    run(f'risk budget eps={eps:.2f}', 'budget', eps)
