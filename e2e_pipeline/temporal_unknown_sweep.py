"""The four occlusion tests again, under a temporal `unknown` and a graded gate.

Three arms, changing one thing at a time so a difference is attributable:

  raycast + 2-valued   the shape all previous runs used: unknown is the
                       geometric shadow and hitting it is a hard veto
  temporal + 2-valued  unknown becomes "not observed within N frames", still
                       vetoed -- isolates the DEFINITION change
  temporal + 3-valued  and the clearance gate splits obstacle / free / unknown,
                       with unknown admissible under a stopping constraint and
                       a depth penalty -- isolates the GATE change

Each runs the same four priors and answers the same three questions: does the
occlusion prior trigger, do brakes change, does risk change.

Diversity and occupancy indicators are measured PRE-GATE, on the candidate set
as proposed. Measured post-gate they describe the filter, not the planner --
which is exactly how "0 of 6 waypoints land in unknown" came to be reported for
three investigations while the planner was in fact proposing occlusion-entering
candidates on 20% of steps.

Commands are derived per step (`command=None`), since the hardcoded 'straight'
was itself suppressing 97% of the candidate geometry.

Usage:
    PYTHONPATH=. python e2e_pipeline/temporal_unknown_sweep.py
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import (ClosedLoopRunner, FlashOccWorldModel,
                                      LoopConfig, diffusiondrive_anchor_planner)
from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.scene import EgoState, SceneRepresentation
from e2e_pipeline.world_model import ReactiveWorldModel

A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'
PRIORS = (0.00, 0.02, 0.05, 0.10)

ARMS = (
    ('raycast + 2-valued', 'raycast', False),
    ('temporal + 2-valued', 'temporal', False),
    ('temporal + 3-valued', 'temporal', True),
)


def run(nusc, prior, occlusion, three, n=10):
    B = E = 0
    clears, risks, unk_map = [], [], []
    cand_tot = cand_unk = 0
    verdicts = Counter()
    sel_hit = []
    for i in range(n):
        w = FlashOccWorldModel(nusc, scene_idx=i, veto_unknown=False,
                               occlusion=occlusion)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                         calibrate_risk=True, unknown_prior=prior,
                         veto_unknown=False)
        filt = SafetyFilter(limits=FeasibilityLimits(
            max_risk=0.10, three_valued_unknown=three))
        planner = diffusiondrive_anchor_planner(A, cfg.dt)
        r = ClosedLoopRunner(w, planner, cfg, safety=filt,
                             latent_model=ReactiveWorldModel())
        recs, m = r.run(command=None)
        s = m['safety']
        B += s['emergency_brakes']
        E += s['n_collision_steps_ego_fault']
        if s['min_clearance_m'] is not None:
            clears.append(s['min_clearance_m'])
        risks += [x for x in r.risk_pred if x is not None]

        # -- PRE-GATE indicators ------------------------------------------
        for k in range(min(20, len(w.samples) - 1)):
            fr = w.samples[k]
            fs = w.freespace_at(k * cfg.dt, fr['xy'], fr['yaw'])
            if fs is None or fs.unknown is None:
                continue
            unk_map.append(float(fs.unknown.mean()))
            sc = SceneRepresentation(
                agents=w.agents_at(k * cfg.dt, fr['xy'], fr['yaw']),
                freespace=fs, ego=EgoState(speed=w.initial_speed()),
                timestamp=k * cfg.dt)
            cands, _ = planner(sc, w.command_at(k * cfg.dt))
            c = np.asarray(cands, float)
            cand_tot += len(c)
            for t in c:
                per = fs.unknown_at(t)
                if per.any():
                    cand_unk += 1
                    if three:
                        entry, depth = filt.unknown_geometry(t, per)
                        verdicts[filt.unknown_feasibility(
                            entry, depth, sc.ego.speed)] += 1

        for k, rec in enumerate(recs):
            if rec.planned_traj is None or not len(rec.planned_traj):
                continue
            fs = w.freespace_at(k * cfg.dt, rec.ego_xy, rec.ego_yaw)
            if fs is None or fs.unknown is None:
                continue
            sel_hit.append(float(np.mean(
                fs.unknown_at(np.asarray(rec.planned_traj, float)))))

    return {
        'brakes': B, 'ego': E,
        'clear': float(np.mean(clears)) if clears else float('nan'),
        'risk': float(np.mean(risks)) if risks else float('nan'),
        'risk_p90': float(np.percentile(risks, 90)) if risks else float('nan'),
        'unk_map': float(np.mean(unk_map)) if unk_map else float('nan'),
        'cand_unk': cand_unk / max(cand_tot, 1),
        'sel_hit': float(np.mean(sel_hit)) if sel_hit else float('nan'),
        'verdicts': verdicts,
    }


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)
    for label, occ, three in ARMS:
        print(f'\n  === {label} ===')
        print(f'  {"prior":>7}{"brakes":>8}{"EGO":>5}{"clear":>8}{"mean risk":>11}'
              f'{"risk p90":>10}{"map unk":>9}{"cand->unk":>11}{"sel->unk":>10}')
        last = None
        for p in PRIORS:
            r = run(nusc, p, occ, three)
            last = r
            print(f'  {p:>7.2f}{r["brakes"]:>8}{r["ego"]:>5}{r["clear"]:>7.2f}m'
                  f'{r["risk"]:>11.4f}{r["risk_p90"]:>10.4f}{r["unk_map"]:>9.1%}'
                  f'{r["cand_unk"]:>11.1%}{r["sel_hit"]:>10.2%}')
        if last and last['verdicts']:
            tot = sum(last['verdicts'].values())
            print('    unknown verdicts on occlusion-entering candidates: '
                  + '  '.join(f'{k} {v / tot:.0%}'
                              for k, v in last['verdicts'].most_common()))


if __name__ == '__main__':
    main()
