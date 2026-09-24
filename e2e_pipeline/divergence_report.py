"""Run the four divergence-aware metrics, GT vs live detector.

Usage:
    PYTHONPATH=. python e2e_pipeline/divergence_report.py
"""
import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import (ClosedLoopRunner, GTWorldModel,
                                      LivePerceptionWorldModel, LoopConfig,
                                      diffusiondrive_anchor_planner)
from e2e_pipeline.divergence_metrics import (StepObs, bucketed_collision_rate,
                                             counterfactual_safety,
                                             progress_per_divergence,
                                             recovery_rate)
from e2e_pipeline.metrics import _agent_poly, _ego_poly, polygon_distance
from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.uncertainty import RiskModel
from e2e_pipeline.world_model import ReactiveWorldModel

A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
DT = 0.5


def gather(nusc, WorldCls):
    allobs, divs_per_scene, per_scene = [], [], []
    for i in range(10):
        w = WorldCls(nusc, scene_idx=i)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                         calibrate_risk=True)
        r = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                             safety=SafetyFilter(limits=FeasibilityLimits(max_risk=0.10)),
                             latent_model=ReactiveWorldModel())
        recs, _ = r.run(command=2)
        divs, prev = [], None
        for k, rec in enumerate(recs):
            fr = w.samples[min(k, len(w.samples) - 1)]
            div = float(np.linalg.norm(np.asarray(rec.ego_xy, float) - fr['xy']))
            divs.append(div)
            ego = _ego_poly(rec, 4.6, 1.8)
            dmin = min([polygon_distance(ego, _agent_poly(b)) for b in rec.agent_boxes],
                       default=np.inf)
            prog = 0.0 if prev is None else float(np.linalg.norm(
                np.asarray(rec.ego_xy, float) - prev))
            prev = np.asarray(rec.ego_xy, float)
            lv = (float(np.linalg.norm(w.samples[k + 1]['xy'] - fr['xy']) / DT)
                  if k + 1 < len(w.samples) else 0.0)

            # counterfactual: same world, two trajectories
            agents = w.agents_at(k * DT, fr['xy'], fr['yaw'])
            rm = RiskModel(type(rec).__mro__ and __import__(
                'e2e_pipeline.scene', fromlist=['EgoState']).EgoState(speed=rec.ego_v),
                calibrator=r.calibrator)
            logged_fut = []
            for h in range(1, 7):
                j = min(k + h, len(w.samples) - 1)
                d = w.samples[j]['xy'] - fr['xy']
                c, s = np.cos(-fr['yaw']), np.sin(-fr['yaw'])
                logged_fut.append([c * d[0] - s * d[1], s * d[0] + c * d[1]])
            if rec.planned_traj is None or not len(rec.planned_traj):
                continue            # emergency brake: no plan to compare
            plan = np.asarray(rec.planned_traj, float)
            try:
                rp = float(rm.evaluate(plan, agents, dt=DT).total)
                rl = float(rm.evaluate(np.asarray(logged_fut, float), agents, dt=DT).total)
            except Exception:
                rp = rl = 0.0
            allobs.append(StepObs(divergence=div, collided=dmin <= 0.0,
                                  progress_m=prog, ego_v=rec.ego_v, logged_v=lv,
                                  risk_planner=rp, risk_logged=rl))
        divs_per_scene.append(divs)
        per_scene.append([o for o in allobs[-len(divs):]])
    return allobs, divs_per_scene, per_scene


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    res = {}
    for label, cls in (('GT', GTWorldModel), ('LIVE', LivePerceptionWorldModel)):
        res[label] = gather(nusc, cls)

    print('  (1) COLLISION RATE WITHIN MATCHED DIVERGENCE BUCKETS')
    print(f'  {"bucket":>12}{"GT n":>7}{"GT rate":>10}{"LIVE n":>8}{"LIVE rate":>11}')
    gb = bucketed_collision_rate(res['GT'][0])
    lb = bucketed_collision_rate(res['LIVE'][0])
    for g, l in zip(gb, lb):
        hi = 'inf' if not np.isfinite(g['hi']) else f"{g['hi']:.0f}"
        print(f"  {f'{g[chr(39)+chr(39)] if False else g['lo']:.0f}-{hi} m':>12}"
              f"{g['n']:>7}{g['collision_rate']:>10.1%}{l['n']:>8}{l['collision_rate']:>11.1%}")

    print('\n  (2) RECOVERY RATE (divergence > 3 m, recovered below 1.5 m)')
    for label in ('GT', 'LIVE'):
        agg = [recovery_rate(d) for d in res[label][1]]
        ex = sum(a['excursions'] for a in agg)
        rr = [a['recovery_rate'] for a in agg if np.isfinite(a['recovery_rate'])]
        worst = max(a['longest_unrecovered'] for a in agg)
        print(f'   {label:5s} excursions {ex:>3}   recovery rate '
              f'{np.mean(rr) if rr else float("nan"):.0%}   longest unrecovered '
              f'{worst} steps')

    print('\n  (3) COUNTERFACTUAL SAFETY (plan vs logged ego, same world)')
    for label in ('GT', 'LIVE'):
        c = counterfactual_safety(res[label][0])
        print(f"   {label:5s} n={c['n']:>4}  mean Δrisk {c['mean_delta']:+.4f}  "
              f"worse {c['worse_rate']:.0%}  much-worse {c['much_worse_rate']:.0%}  "
              f"better {c['better_rate']:.0%}")

    print('\n  (4) PROGRESS PER UNIT DIVERGENCE')
    for label in ('GT', 'LIVE'):
        from e2e_pipeline.divergence_metrics import aggregate_scenes
        p = aggregate_scenes(res[label][2])
        print(f"   {label:5s} progress {p['progress_m']:>6.0f} m / divergence "
              f"{p['divergence_m']:>6.0f} m  =  {p['progress_per_div']:>5.1f}  "
              f"stalled {p['stalled_fraction']:.0%}  speed vs logged {p['speed_ratio']:.0%}")


if __name__ == '__main__':
    main()
