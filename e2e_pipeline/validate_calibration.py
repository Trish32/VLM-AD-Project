import numpy as np, os, itertools
from nuscenes import NuScenes
from e2e_pipeline.closed_loop import (ReactiveGTWorldModel, LoopConfig,
                                      ClosedLoopRunner, diffusiondrive_anchor_planner)
from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.uncertainty import RiskModel
from e2e_pipeline.world_model import ReactiveWorldModel
from e2e_pipeline.metrics import _ego_poly, _agent_poly, polygon_distance
from e2e_pipeline.calibration import (expected_calibration_error, PlattCalibrator)

nusc = NuScenes(version='v1.0-mini',
                dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'), verbose=False)
A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
orig = SafetyFilter.__call__
DATA = {}          # scene -> (preds, labels)

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
        cfg = LoopConfig(max_steps=39, initial_speed=w.initial_speed() * vmul)
        recs, _ = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                                   safety=SafetyFilter(limits=FeasibilityLimits(max_risk=0.60)),
                                   latent_model=ReactiveWorldModel()).run(command=cmd)
    except Exception:
        SafetyFilter.__call__ = orig; continue
    SafetyFilter.__call__ = orig
    P, Y = DATA.setdefault(scene, ([], []))
    for i, rec in enumerate(recs):
        if i >= len(preds) or preds[i] is None: continue
        ego = _ego_poly(rec, 4.6, 1.8)
        d = min([polygon_distance(ego, _agent_poly(b)) for b in rec.agent_boxes], default=np.inf)
        P.append(preds[i]); Y.append(1.0 if d <= 1.0 else 0.0)

print('  per-scene positives:', {k: int(np.sum(v[1])) for k, v in sorted(DATA.items())})

def pack(scenes):
    P = np.concatenate([DATA[s][0] for s in scenes if s in DATA])
    Y = np.concatenate([DATA[s][1] for s in scenes if s in DATA])
    return P, Y

# scene-level split, both directions so neither half is cherry-picked
for tr_s, te_s, tag in ((range(0, 5), range(5, 10), 'train 0-4 / test 5-9'),
                        (range(5, 10), range(0, 5), 'train 5-9 / test 0-4')):
    Ptr, Ytr = pack(list(tr_s)); Pte, Yte = pack(list(te_s))
    pl = PlattCalibrator().fit(Ptr, Ytr)
    e_un = expected_calibration_error(Pte, Yte, 5, 'quantile')
    e_cal = expected_calibration_error(pl(Pte), Yte, 5, 'quantile')
    print(f'\n  {tag}')
    print(f'    train positives {int(Ytr.sum())} (identifiable={pl.identifiable})  a={pl.a:.3f} b={pl.b:.3f}')
    print(f'    test  positives {int(Yte.sum())}  n={len(Yte)}')
    print(f'    ECE  uncalibrated {e_un["ece"]:.4f}  ->  Platt {e_cal["ece"]:.4f}')
    print(f'    MCE  uncalibrated {e_un["mce"]:.4f}  ->  Platt {e_cal["mce"]:.4f}')
