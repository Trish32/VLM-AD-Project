"""Additive vs multiplicative ranking, and whether lambda reorders past 11.

TWO SYMPTOMS OF ONE CAUSE. Calibration maps every raw risk into
[0.0884, 0.1328] -- a band 4.4 points wide. In a weighted sum that means the
risk term can only move the total by w_risk * 0.044, while clearance spreads
0.893 across candidates. So:

  * W_RISK is unidentifiable below lambda ~ 11, because that is where
    lambda * 0.081 finally exceeds the 0.893 clearance spread. Every previous
    sweep stopped at 9.5 and concluded "inert" from inside the region where no
    answer was possible.
  * Wiring the calibrator into the critic CUT risk spread 0.516 -> 0.081.
    Calibrating the numbers, which is correct, destroyed their influence on the
    ranking, which is not.

Multiplying rather than adding removes the coupling: `clearance * (1 - risk)`
makes risk a fraction of a plan's value rather than a fixed subtraction, so a
narrow calibrated range still expresses itself proportionally.

Usage:
    PYTHONPATH=. python e2e_pipeline/ranking_form_ab.py
"""
from __future__ import annotations

import os

import numpy as np
from nuscenes import NuScenes

from e2e_pipeline.closed_loop import (
    ClosedLoopRunner, LivePerceptionWorldModel, LoopConfig,
    diffusiondrive_anchor_planner)
from e2e_pipeline.planner.safety_filter import FeasibilityLimits, SafetyFilter
from e2e_pipeline.planner.world_model import ReactiveWorldModel

A = 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'


def run(nusc, multiplicative, w_risk=10.0, n=10):
    E = O = B = 0
    comps, clears, jerks, risks = [], [], [], []
    for i in range(n):
        w = LivePerceptionWorldModel(nusc, scene_idx=i)
        cfg = LoopConfig(max_steps=20, initial_speed=w.initial_speed(),
                         calibrate_risk=True)
        filt = SafetyFilter(limits=FeasibilityLimits(max_risk=0.10),
                            w_risk=w_risk, multiplicative=multiplicative)
        r = ClosedLoopRunner(w, diffusiondrive_anchor_planner(A, cfg.dt), cfg,
                             safety=filt, latent_model=ReactiveWorldModel())
        _, m = r.run(command=None)
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
        risks += [x for x in r.risk_pred if x is not None]
    return {
        'ego': E, 'other': O, 'brakes': B,
        'clear': float(np.mean(clears)) if clears else float('nan'),
        'compl': float(np.mean(comps)) if comps else float('nan'),
        'jerk': float(np.mean(jerks)) if jerks else float('nan'),
        'risk': float(np.mean(risks)) if risks else float('nan'),
    }


def main():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.expanduser('~/Downloads/nuScenes_miniV1.0'),
                    verbose=False)

    print('  === ranking form, w_risk = 10 ===')
    print(f'  {"form":>16}{"EGO":>5}{"other":>7}{"brakes":>8}{"clear":>8}'
          f'{"compl":>8}{"jerk":>7}{"mean risk":>11}')
    for label, mult in (('additive', False), ('multiplicative', True)):
        r = run(nusc, mult)
        print(f'  {label:>16}{r["ego"]:>5}{r["other"]:>7}{r["brakes"]:>8}'
              f'{r["clear"]:>7.2f}m{r["compl"]:>8.1%}{r["jerk"]:>7.2f}'
              f'{r["risk"]:>11.4f}')

    # Does w_risk reorder anything past the point where lambda * risk-spread
    # overtakes the clearance spread? Additive only -- the multiplicative form
    # has no w_risk to sweep, which is the point.
    print('\n  === additive: does w_risk bite past ~11? ===')
    print(f'  {"w_risk":>8}{"lam*0.081":>11}{"EGO":>5}{"other":>7}{"brakes":>8}'
          f'{"clear":>8}{"compl":>8}{"mean risk":>11}')
    for wr in (6.0, 9.0, 11.0, 13.0, 15.0, 25.0):
        r = run(nusc, False, w_risk=wr)
        mark = '  <- exceeds clearance spread 0.893' if wr * 0.081 > 0.893 else ''
        print(f'  {wr:>8.1f}{wr * 0.081:>11.3f}{r["ego"]:>5}{r["other"]:>7}'
              f'{r["brakes"]:>8}{r["clear"]:>7.2f}m{r["compl"]:>8.1%}'
              f'{r["risk"]:>11.4f}{mark}')


if __name__ == '__main__':
    main()
