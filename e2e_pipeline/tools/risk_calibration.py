"""Reliability diagram for RiskModel: is its collision probability calibrated?

A probability is only meaningful if events it assigns p actually occur at rate
p. RiskModel computes one analytically from Kalman covariance and a non-central
chi-squared CDF -- a MODEL, never checked against outcomes. This checks it.

Unlike action-conditioned prediction, calibration IS learnable from logs: it
compares predicted rates to observed frequencies, which recorded data contains.
What it needs is positive events, and that is the binding constraint here.

Usage:
    PYTHONPATH=. python e2e_pipeline/risk_calibration.py
"""
import numpy as np, os, collections
from nuscenes import NuScenes
from e2e_pipeline.closed_loop import (
    ReactiveGTWorldModel, LoopConfig, ClosedLoopRunner,
    diffusiondrive_anchor_planner)
from e2e_pipeline.planner.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.uncertainty import RiskModel
from e2e_pipeline.planner.world_model import ReactiveWorldModel

nusc = NuScenes(version='v1.0-mini',
                dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'), verbose=False)
A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'

orig = SafetyFilter.__call__
pairs = []          # (predicted_risk_of_chosen, scene, step_index)

def cap(self, c, sc, s=None, r=None):
    res = orig(self, c, sc, s, r)
    if not res.emergency and res.chosen_index is not None:
        p = RiskModel(sc.ego).evaluate(np.asarray(res.trajectory, float),
                                       sc.agents, dt=0.5)
        pairs.append(float(p.total))
    else:
        pairs.append(None)
    return res

SafetyFilter.__call__ = cap
outcomes = []
for i in range(10):
    w = ReactiveGTWorldModel(nusc, scene_idx=i)
    cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed())
    filt = SafetyFilter(limits=FeasibilityLimits(max_risk=0.60))
    recs, m = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                               safety=filt, latent_model=ReactiveWorldModel()).run(command=2)
    from e2e_pipeline.metrics.metrics import _ego_poly, _agent_poly, polygon_distance
    for rec in recs:
        ego = _ego_poly(rec, 4.6, 1.8)
        d = min([polygon_distance(ego, _agent_poly(b)) for b in rec.agent_boxes], default=np.inf)
        outcomes.append(d)
SafetyFilter.__call__ = orig

pred = [p for p in pairs if p is not None]
n = min(len(pred), len(outcomes))
pred, out = np.array(pairs[:n], dtype=object), np.array(outcomes[:n])
mask = np.array([p is not None for p in pred])
pr = np.array([p for p in pred if p is not None], float)
ob = out[mask]

print(f'  predictions with a chosen plan: {len(pr)}/{n}')
print(f'  predicted risk:  mean {pr.mean():.4f}  p50 {np.percentile(pr,50):.4f}  p95 {np.percentile(pr,95):.4f}  max {pr.max():.4f}')
for thr, lab in ((0.0, 'collision (d<=0)'), (0.5, 'near-miss (d<0.5m)'), (1.0, 'close (d<1.0m)')):
    pos = int((ob <= thr).sum())
    print(f'  observed {lab:20s}: {pos:4d}/{len(ob)} = {pos/max(len(ob),1):.1%}')
print('\n  RELIABILITY (predicted vs observed near-miss rate, d<1.0m):')
bins = [0, .01, .05, .15, .40, 1.01]
for lo, hi in zip(bins, bins[1:]):
    m_ = (pr >= lo) & (pr < hi)
    if m_.sum():
        print(f'   p in [{lo:.2f},{hi:.2f}): n={m_.sum():4d}  mean pred {pr[m_].mean():.3f}  observed {(ob[m_]<=1.0).mean():.3f}')
