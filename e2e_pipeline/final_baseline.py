"""End-to-end re-baseline under the current defaults.

EVERY NUMBER IN THIS LOG PREDATES FOUR DEFAULT CHANGES. Since the last full run
the stack has changed in ways that interact:

  * the drive command is honoured instead of hardcoded 'straight' (§27)
  * per-object covariance is fitted, and detector worlds use it (§26)
  * live detections are associated across frames instead of re-seeded (§26)
  * SafetyFilter.w_risk is 1.0 instead of 10.0 (§29)

Measuring them one at a time was right for attribution and is wrong for a
headline: the combination has never been run. This is the canonical
configuration, reported across the five metric families, for both perception
sources and both ego modes.

BOTH EGO MODES, AND THE PINNED ONE IS THE SAFETY NUMBER. §19 established that a
collision count under a free-running ego mostly measures deviation from the
recording. Pinned removes that confound and is the number to read for safety;
free-running is the number to read for whether the policy drives.

Usage:
    PYTHONPATH=. python e2e_pipeline/final_baseline.py
"""
from __future__ import annotations

import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import (ClosedLoopRunner, GTWorldModel,
                                      LivePerceptionWorldModel, LoopConfig,
                                      diffusiondrive_anchor_planner)
from e2e_pipeline.divergence_metrics import (bucketed_collision_rate,
                                             recovery_rate)
from e2e_pipeline.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.world_model import ReactiveWorldModel

A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'


def run(nusc, WorldCls, pinned, command, n=10):
    E = O = B = 0
    comps, clears, jerks, divs, risks, lat = [], [], [], [], [], []
    ade, fde = [], []
    per_scene_div = []
    for i in range(n):
        w = WorldCls(nusc, scene_idx=i)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                         calibrate_risk=True, follow_logged_ego=pinned)
        r = ClosedLoopRunner(
            w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
            safety=SafetyFilter(limits=FeasibilityLimits(max_risk=0.10)),
            latent_model=ReactiveWorldModel())
        recs, m = r.run(command=command)
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
        p = m.get('prediction', {})
        if p.get('ade') is not None:
            ade.append(p['ade'])
        if p.get('fde') is not None:
            fde.append(p['fde'])
        risks += [x for x in r.risk_pred if x is not None]
        lat += [sum(rec.latency_ms.values()) for rec in recs if rec.latency_ms]
        d = [float(np.linalg.norm(np.asarray(rec.ego_xy, float)
                                  - w.samples[min(k, len(w.samples) - 1)]['xy']))
             for k, rec in enumerate(recs)]
        divs += d
        per_scene_div.append(d)
    ece = float('nan')
    return {
        'ego': E, 'other': O, 'brakes': B,
        'clear': float(np.mean(clears)) if clears else float('nan'),
        'compl': float(np.mean(comps)) if comps else float('nan'),
        'jerk': float(np.mean(jerks)) if jerks else float('nan'),
        'ade': float(np.mean(ade)) if ade else float('nan'),
        'fde': float(np.mean(fde)) if fde else float('nan'),
        'div': float(np.mean(divs)) if divs else float('nan'),
        'risk': float(np.mean(risks)) if risks else float('nan'),
        'lat_p50': float(np.percentile(lat, 50)) if lat else float('nan'),
        'lat_p95': float(np.percentile(lat, 95)) if lat else float('nan'),
        'per_scene_div': per_scene_div, 'ece': ece,
    }


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)

    print('  CANONICAL BASELINE -- current defaults, derived commands\n')
    hdr = (f'  {"config":>26}{"EGO":>5}{"other":>7}{"brakes":>8}{"clear":>8}'
           f'{"compl":>8}{"jerk":>7}{"diverg":>9}{"risk":>8}'
           f'{"lat p50":>9}{"lat p95":>9}')
    print(hdr)
    out = {}
    for label, cls in (('GT', GTWorldModel), ('LIVE', LivePerceptionWorldModel)):
        for mode, pinned in (('pinned', True), ('free', False)):
            r = run(nusc, cls, pinned, None)
            out[(label, mode)] = r
            print(f'  {f"{label}, {mode}, cmd derived":>26}{r["ego"]:>5}'
                  f'{r["other"]:>7}{r["brakes"]:>8}{r["clear"]:>7.2f}m'
                  f'{r["compl"]:>8.1%}{r["jerk"]:>7.2f}{r["div"]:>8.1f}m'
                  f'{r["risk"]:>8.4f}{r["lat_p50"]:>8.1f}m{r["lat_p95"]:>8.1f}m')

    # the old default, for the delta
    print()
    for label, cls in (('GT', GTWorldModel), ('LIVE', LivePerceptionWorldModel)):
        r = run(nusc, cls, False, 2)
        out[(label, 'old')] = r
        print(f'  {f"{label}, free, cmd=straight":>26}{r["ego"]:>5}'
              f'{r["other"]:>7}{r["brakes"]:>8}{r["clear"]:>7.2f}m'
              f'{r["compl"]:>8.1%}{r["jerk"]:>7.2f}{r["div"]:>8.1f}m'
              f'{r["risk"]:>8.4f}{r["lat_p50"]:>8.1f}m{r["lat_p95"]:>8.1f}m')

    print('\n  RECOVERY FROM DIVERGENCE (free-running, derived commands)')
    for label in ('GT', 'LIVE'):
        agg = [recovery_rate(d) for d in out[(label, 'free')]['per_scene_div']]
        ex = sum(a['excursions'] for a in agg)
        rr = [a['recovery_rate'] for a in agg if np.isfinite(a['recovery_rate'])]
        worst = max(a['longest_unrecovered'] for a in agg)
        print(f'   {label:5s} excursions past 3 m {ex:>3}   recovery rate '
              f'{(np.mean(rr) if rr else float("nan")):.0%}   longest '
              f'unrecovered {worst} steps')


if __name__ == '__main__':
    main()
