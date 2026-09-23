#!/usr/bin/env python3
"""Does a detector-induced decision difference survive into the trajectory?

`compare_detectors.py` measures where four detectors disagree about the DECISION.
That is one categorical token, and a difference there is only interesting if it
changes what the car actually does. This runs the rest of the pipeline --
decision -> DrivingIntent -> DiffusionDrive anchors -> trajectory -- and measures
the divergence in metres.

Why it can only shrink the difference, never grow it: the arms share a frame, a
command (`straight`), and an anchor set, so two arms that reach the same decision
produce byte-identical trajectories, and two that differ produce trajectories
that differ only through `target_speed_mps`. The planner pins the first waypoint
to what is reachable from the current speed, so even a PROCEED/STOP disagreement
cannot separate the paths by more than one step of braking authority at t=0.5 s.
Reporting metres rather than decision-agreement is the point: it converts a
categorical disagreement into the units a planner is judged in.

DECISION -> TARGET SPEED. The VLM emits a categorical decision, the planner wants
m/s. The mapping is a fraction of the ego's CURRENT logged speed, so the same
decision means something different at 3 m/s and 15 m/s:

    PROCEED    1.0 x v0      SLOW_DOWN  0.5 x v0
    YIELD      0.3 x v0      STOP       0.0

These fractions are a stated convention, not a measured quantity, and every arm
is subject to the identical mapping -- so they cannot favour an arm, but they do
set the overall scale of the reported divergences.

Usage:
    conda run -n simple_bev_vldrive python tools/compare_planning.py \
        --decisions eval_results/detector_comparison.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from nuscenes import NuScenes

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from e2e_pipeline.scene import EgoState, SceneRepresentation
from e2e_pipeline.vlm_planner import DrivingIntent, intent_conditioned_planner

from dataroot import default_dataroot
from score_decisions import ego_speed_profile

SPEED_FRACTION = {'PROCEED': 1.0, 'SLOW_DOWN': 0.5, 'YIELD': 0.3, 'STOP': 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--decisions',
                    default=str(ROOT / 'eval_results' / 'detector_comparison.json'))
    ap.add_argument('--anchors',
                    default=str(REPO / 'diffusiondrive_planner' / 'data' / 'kmeans'
                                / 'kmeans_plan_6.npy'))
    ap.add_argument('--dataroot', default=default_dataroot())
    ap.add_argument('--dt', type=float, default=0.5)
    ap.add_argument('--out',
                    default=str(ROOT / 'eval_results' / 'planning_comparison.json'))
    args = ap.parse_args()

    data = json.loads(Path(args.decisions).read_text())
    arms, frames = data['arms'], data['frames']
    plan = intent_conditioned_planner(args.anchors, dt=args.dt)

    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)
    speed_of = {}
    for si in range(len(nusc.scene)):
        toks, sp = ego_speed_profile(nusc, si)
        speed_of.update(dict(zip(toks, sp)))

    rows = []
    for fr in frames:
        tok = fr['token']
        v0 = float(speed_of.get(tok, 0.0))
        scene = SceneRepresentation(agents=[], freespace=None,
                                    ego=EgoState(speed=v0))
        trajs = {}
        for lab in arms:
            dec = fr['decisions'][lab]
            intent = DrivingIntent(
                command='straight',
                target_speed_mps=SPEED_FRACTION.get(dec, 1.0) * v0)
            cands, _scores = plan(scene, intent)
            # Compare the whole candidate SET, not one pick: the safety filter
            # chooses among them downstream, so the set is what the VLM actually
            # hands over.
            trajs[lab] = np.asarray(cands, dtype=np.float64)
        rows.append({'token': tok, 'v0': v0, 'trajs': trajs,
                     'decisions': fr['decisions'], 'light': fr['light']})

    n = len(rows)
    print(f'\n{"=" * 72}\n{n} frames, {len(arms)} arms -> DiffusionDrive '
          f'trajectories\n{"=" * 72}')
    print(f'anchors: {args.anchors}')
    print(f'mean ego speed {np.mean([r["v0"] for r in rows]):.2f} m/s\n')

    print('PAIRWISE TRAJECTORY DIVERGENCE  (mean over frames and candidates)')
    print(f'  {"pair":<40s}{"endpoint":>11s}{"mean|path|":>12s}{"frames diff":>13s}')
    summary = {}
    for a, b in itertools.combinations(arms, 2):
        ends, paths, ndiff = [], [], 0
        for r in rows:
            ta, tb = r['trajs'][a], r['trajs'][b]
            e = float(np.mean(np.linalg.norm(ta[:, -1] - tb[:, -1], axis=-1)))
            p = float(np.mean(np.linalg.norm(ta - tb, axis=-1)))
            ends.append(e)
            paths.append(p)
            if r['decisions'][a] != r['decisions'][b]:
                ndiff += 1
        summary[f'{a} vs {b}'] = {'endpoint_m': float(np.mean(ends)),
                                  'path_m': float(np.mean(paths)),
                                  'frames_decision_differs': ndiff}
        print(f'  {a + " vs " + b:<40s}{np.mean(ends):>10.3f}m'
              f'{np.mean(paths):>11.3f}m{ndiff:>10d}/{n}')

    # On frames where two arms DO disagree, how far apart do the paths get? The
    # all-frames mean above is diluted by the frames where they agree exactly.
    print('\nON DISAGREEING FRAMES ONLY')
    any_printed = False
    for a, b in itertools.combinations(arms, 2):
        sel = [r for r in rows if r['decisions'][a] != r['decisions'][b]]
        if not sel:
            continue
        any_printed = True
        e = np.mean([np.mean(np.linalg.norm(r['trajs'][a][:, -1]
                                            - r['trajs'][b][:, -1], axis=-1))
                     for r in sel])
        print(f'  {a + " vs " + b:<40s}{e:>10.3f}m  over {len(sel)} frames')
    if not any_printed:
        print('  none -- every arm produced the same decision on every frame.')
        print('  Trajectory divergence is exactly 0 by construction, and this')
        print('  experiment has measured that the detector never reached the plan.')

    Path(args.out).write_text(json.dumps(
        {'n_frames': n, 'arms': arms, 'speed_fraction': SPEED_FRACTION,
         'anchors': args.anchors, 'pairwise': summary}, indent=1))
    print(f'\n[INFO] saved -> {args.out}')


if __name__ == '__main__':
    main()
