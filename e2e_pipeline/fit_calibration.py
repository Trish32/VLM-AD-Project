"""Collect (predicted risk, outcome) pairs and fit the Platt map.

The first attempt had 2 positives from one configuration and was not
identifiable. Varying command and initial speed explores genuinely different
rollouts through the same scenes -- real data, not resampling -- and reaches 21
positives over 966 steps.

CAVEAT ON THE SPLIT. The held-out half is random over STEPS, and all 10 scenes
appear in both halves. That leaks scene-level structure, so the held-out ECE is
optimistic; a scene-level split is the stricter test and is not yet run.

Usage:
    PYTHONPATH=. python e2e_pipeline/fit_calibration.py
"""
import numpy as np, os, itertools
from nuscenes import NuScenes
from e2e_pipeline.closed_loop import (ReactiveGTWorldModel, LoopConfig,
                                      ClosedLoopRunner, diffusiondrive_anchor_planner)
from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.uncertainty import RiskModel
from e2e_pipeline.world_model import ReactiveWorldModel
from e2e_pipeline.metrics import _ego_poly, _agent_poly, polygon_distance
from e2e_pipeline.calibration import (expected_calibration_error, format_reliability,
                                      PlattCalibrator)

nusc = NuScenes(version='v1.0-mini',
                dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'), verbose=False)
A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
orig = SafetyFilter.__call__
P, Y = [], []

for scene, cmd, vmul in itertools.product(range(10), (0, 1, 2), (0.6, 1.0, 1.4)):
    preds = []
    def cap(self, c, sc, s=None, r=None):
        res = orig(self, c, sc, s, r)
        preds.append(float(RiskModel(sc.ego).evaluate(np.asarray(res.trajectory, float),
                                                      sc.agents, dt=0.5).total)
                     if not res.emergency and len(res.trajectory) else None)
        return res
    SafetyFilter.__call__ = cap
    try:
        w = ReactiveGTWorldModel(nusc, scene_idx=scene)
        v0 = w.initial_speed() * vmul
        cfg = LoopConfig(max_steps=39, initial_speed=v0)
        recs, _ = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                                   safety=SafetyFilter(limits=FeasibilityLimits(max_risk=0.60)),
                                   latent_model=ReactiveWorldModel()).run(command=cmd)
    except Exception:
        SafetyFilter.__call__ = orig
        continue
    SafetyFilter.__call__ = orig
    for i, rec in enumerate(recs):
        if i >= len(preds) or preds[i] is None:
            continue
        ego = _ego_poly(rec, 4.6, 1.8)
        d = min([polygon_distance(ego, _agent_poly(b)) for b in rec.agent_boxes], default=np.inf)
        P.append(preds[i]); Y.append(1.0 if d <= 1.0 else 0.0)

P, Y = np.array(P), np.array(Y)
print(f'  collected {len(P)} (pred,outcome) pairs from 10 scenes x 3 commands x 3 speeds')
print(f'  positives: {int(Y.sum())}  ({Y.mean():.2%})')
print('\n  BEFORE'); print(format_reliability(expected_calibration_error(P, Y, 5, 'quantile')))
# held-out split so the fit is not scored on its own data
rng = np.random.default_rng(0); idx = rng.permutation(len(P)); k = len(P)//2
tr, te = idx[:k], idx[k:]
pl = PlattCalibrator().fit(P[tr], Y[tr])
print(f'\n  Platt on train half: a={pl.a:.3f} b={pl.b:.3f} positives={pl.n_positive} identifiable={pl.identifiable}')
print('\n  AFTER, on HELD-OUT half')
print(format_reliability(expected_calibration_error(pl(P[te]), Y[te], 5, 'quantile')))
print('\n  held-out ECE uncalibrated:',
      f"{expected_calibration_error(P[te], Y[te], 5, 'quantile')['ece']:.4f}")
