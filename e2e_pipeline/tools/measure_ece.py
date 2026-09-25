import numpy as np, os
from nuscenes import NuScenes
from e2e_pipeline.closed_loop import (
    ReactiveGTWorldModel, LoopConfig, ClosedLoopRunner,
    diffusiondrive_anchor_planner)
from e2e_pipeline.planner.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.uncertainty import RiskModel
from e2e_pipeline.planner.world_model import ReactiveWorldModel
from e2e_pipeline.metrics.metrics import _ego_poly, _agent_poly, polygon_distance
from e2e_pipeline.calibration.calibration import (
    expected_calibration_error, format_reliability, PlattCalibrator,
    TemperatureCalibrator)

nusc = NuScenes(version='v1.0-mini',
                dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'), verbose=False)
A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
orig = SafetyFilter.__call__
preds = []

def cap(self, c, sc, s=None, r=None):
    res = orig(self, c, sc, s, r)
    preds.append(float(RiskModel(sc.ego).evaluate(np.asarray(res.trajectory, float),
                                                  sc.agents, dt=0.5).total)
                 if not res.emergency and len(res.trajectory) else None)
    return res

SafetyFilter.__call__ = cap
labels = []
for i in range(10):
    w = ReactiveGTWorldModel(nusc, scene_idx=i)
    cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed())
    recs, _ = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                               safety=SafetyFilter(limits=FeasibilityLimits(max_risk=0.60)),
                               latent_model=ReactiveWorldModel()).run(command=2)
    for rec in recs:
        ego = _ego_poly(rec, 4.6, 1.8)
        d = min([polygon_distance(ego, _agent_poly(b)) for b in rec.agent_boxes], default=np.inf)
        labels.append(1.0 if d <= 1.0 else 0.0)
SafetyFilter.__call__ = orig

n = min(len(preds), len(labels))
p = np.array([x for x in preds[:n] if x is not None])
y = np.array([labels[:n][i] for i, x in enumerate(preds[:n]) if x is not None])

print('  BEFORE calibration')
print(format_reliability(expected_calibration_error(p, y, n_bins=5, strategy='quantile')))
pl = PlattCalibrator().fit(p, y)
tc = TemperatureCalibrator().fit(p, y)
print(f'\n  Platt fit: a={pl.a:.3f} b={pl.b:.3f}  positives={pl.n_positive}  identifiable={pl.identifiable}')
print(f'  Temperature fit: T={tc.temperature:.3f}  identifiable={tc.identifiable}')
print('\n  AFTER Platt')
print(format_reliability(expected_calibration_error(pl(p), y, n_bins=5, strategy='quantile')))
