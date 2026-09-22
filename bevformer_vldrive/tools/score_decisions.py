#!/usr/bin/env python3
"""Was the decision right? Scored against what the human driver actually did.

Until now the only scored quantity in this stack was "does a red light produce
STOP", which covers a handful of frames. Everything else -- prompt wording, the
detection-text format, hysteresis constants, the risk threshold -- was tuned
against an unmeasured objective. This is the missing yardstick.

THE REFERENCE, AND WHAT IT IS NOT
---------------------------------
nuScenes has no decision labels, so the reference comes from the ego's own
future: differentiate `ego_pose` to get the speed profile over the next 3 s and
read off what the driver did.

    stopped, or decelerating hard      -> STOP
    decelerating moderately            -> SLOW_DOWN
    holding speed or accelerating      -> PROCEED

This is a PROXY and it is wrong in specific, knowable ways:

  * A human slows for reasons no sensor sees -- a familiar junction, a
    passenger, a phone. The pipeline is marked wrong for not hallucinating that.
  * SLOW_DOWN and PROCEED are not crisply separable. A driver easing off 1 m/s
    over 3 s is doing something a reasonable planner might call either.
  * It rewards imitation, not safety. A human who should have braked and did
    not makes PROCEED the "correct" answer.
  * Stationary frames are trivially STOP and would inflate accuracy, so frames
    where the ego never moves across the whole window are excluded by default.

So treat the headline number as a regression detector -- did this change make
agreement worse? -- rather than as a measure of driving quality. The confusion
matrix is more informative than the accuracy, and the ASYMMETRIC errors are the
ones that matter: predicting PROCEED where the human stopped is a different kind
of wrong from predicting STOP where they drove on.

Usage:
    python tools/score_decisions.py --log tools/reasoning_decisions/*.jsonl
    python tools/score_decisions.py --scenes 0 4 --live     # query the VLM now
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from dataroot import default_dataroot

DECISIONS = ('PROCEED', 'SLOW_DOWN', 'YIELD', 'STOP')
# Ordered by how much caution each implies; used for the severity analysis.
SEVERITY = {'PROCEED': 0, 'SLOW_DOWN': 1, 'YIELD': 2, 'STOP': 3}

# Thresholds on the ego's realised behaviour over the horizon.
STOP_SPEED = 0.8        # m/s — below this the ego is stopped
HARD_DECEL = -1.5       # m/s^2 — mean over the horizon
SOFT_DECEL = -0.4       # m/s^2


def ego_speed_profile(nusc, scene_idx: int) -> tuple[list[str], np.ndarray]:
    """(sample_tokens, speeds) for one scene, from logged ego poses."""
    sc = nusc.scene[scene_idx]
    toks, xs = [], []
    tok = sc['first_sample_token']
    while tok:
        s = nusc.get('sample', tok)
        ep = nusc.get('ego_pose',
                      nusc.get('sample_data', s['data']['LIDAR_TOP'])['ego_pose_token'])
        toks.append(tok)
        xs.append(ep['translation'][:2])
        tok = s['next']
    xs = np.asarray(xs, dtype=np.float64)
    step = np.linalg.norm(np.diff(xs, axis=0), axis=1) / 0.5
    speeds = np.concatenate([step[:1], step])          # pad first frame
    return toks, speeds


def reference_decision(speeds: np.ndarray, k: int, horizon: int = 6
                       ) -> tuple[str | None, dict]:
    """What the driver did over the next `horizon` steps, as a decision label.

    Returns (label, detail) or (None, detail) when the window is unusable --
    truncated at the end of the scene, or the ego never moves at all, which
    would be a free STOP and inflate any accuracy computed over it.
    """
    end = k + horizon + 1
    if end > len(speeds):
        return None, {'reason': 'window truncated'}
    win = speeds[k:end]
    v0, v_end = float(win[0]), float(win[-1])
    accel = (v_end - v0) / (horizon * 0.5)

    if v0 < STOP_SPEED and win.max() < STOP_SPEED:
        return None, {'reason': 'ego stationary throughout', 'v0': v0}

    detail = {'v0': round(v0, 2), 'v_end': round(v_end, 2),
              'accel': round(accel, 2), 'v_min': round(float(win.min()), 2)}
    if win.min() < STOP_SPEED or accel < HARD_DECEL:
        return 'STOP', detail
    if accel < SOFT_DECEL:
        return 'SLOW_DOWN', detail
    return 'PROCEED', detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', nargs='+',
                    default=[str(TOOLS_DIR / 'reasoning_decisions' / '*.jsonl')],
                    help='decision JSONL(s) written by vis_infer/make_composite_gif')
    ap.add_argument('--dataroot', default=default_dataroot())
    ap.add_argument('--horizon', type=int, default=6, help='steps (0.5 s each)')
    ap.add_argument('--include-stationary', action='store_true',
                    help='score frames where the ego never moves (inflates accuracy)')
    ap.add_argument('--out', default=str(ROOT / 'eval_results' / 'decision_scores.json'))
    args = ap.parse_args()

    files: list[str] = []
    for pat in args.log:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f'no decision logs matched {args.log}')

    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)
    tok_to_scene = {}
    profiles = {}
    for i in range(len(nusc.scene)):
        toks, sp = ego_speed_profile(nusc, i)
        profiles[i] = (toks, sp)
        for j, t in enumerate(toks):
            tok_to_scene[t] = (i, j)

    rows, skipped = [], Counter()
    for f in files:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tok = r.get('token') or r.get('sample_token')
            pred = r.get('decision', 'UNKNOWN')
            if tok not in tok_to_scene:
                skipped['token not in mini'] += 1
                continue
            if pred not in DECISIONS:
                skipped[f'unparseable: {pred}'] += 1
                continue
            si, k = tok_to_scene[tok]
            ref, detail = reference_decision(profiles[si][1], k, args.horizon)
            if ref is None:
                if not (args.include_stationary
                        and detail.get('reason') == 'ego stationary throughout'):
                    skipped[detail['reason']] += 1
                    continue
                ref = 'STOP'
            rows.append({'token': tok, 'scene': nusc.scene[si]['name'], 'frame': k,
                         'pred': pred, 'ref': ref, 'light': r.get('light', 'none'),
                         **detail})

    if not rows:
        raise SystemExit(f'nothing scoreable. skipped: {dict(skipped)}')

    n = len(rows)
    correct = sum(r['pred'] == r['ref'] for r in rows)
    print(f'\n{n} scoreable decisions from {len(files)} log(s)')
    if skipped:
        print(f'skipped: {dict(skipped)}')
    print(f'\n=== agreement with the human: {correct}/{n} ({correct/n:.1%}) ===')

    print('\nconfusion  (rows = human did, cols = pipeline said):')
    ref_order = [d for d in DECISIONS if any(r['ref'] == d for r in rows)]
    pred_order = [d for d in DECISIONS if any(r['pred'] == d for r in rows)]
    print('              ' + ''.join(f'{p:>11s}' for p in pred_order))
    for ref in ref_order:
        cells = [sum(r['ref'] == ref and r['pred'] == p for r in rows)
                 for p in pred_order]
        print(f'  {ref:<12s}' + ''.join(f'{c:>11d}' for c in cells))

    # The asymmetry is the point: under-reacting and over-reacting are not the
    # same failure, and a single accuracy number hides which one you have.
    under = sum(SEVERITY[r['pred']] < SEVERITY[r['ref']] for r in rows)
    over = sum(SEVERITY[r['pred']] > SEVERITY[r['ref']] for r in rows)
    print(f'\nunder-reacted (less caution than the human): {under}/{n} ({under/n:.1%})')
    print(f'over-reacted  (more caution than the human): {over}/{n} ({over/n:.1%})')
    proceed_on_stop = sum(r['pred'] == 'PROCEED' and r['ref'] == 'STOP' for r in rows)
    print(f'  of which PROCEED where the human stopped: {proceed_on_stop}'
          f'   <- the one that matters')

    lit = [r for r in rows if r['light'] in ('red', 'yellow')]
    if lit:
        stopped = sum(r['pred'] in ('STOP', 'SLOW_DOWN') for r in lit)
        print(f'\nframes where the pipeline read red/yellow: {len(lit)}, '
              f'of which it slowed or stopped: {stopped}/{len(lit)}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {'n': n, 'accuracy': correct / n, 'under_reacted': under,
         'over_reacted': over, 'proceed_on_human_stop': proceed_on_stop,
         'skipped': dict(skipped), 'horizon_steps': args.horizon,
         'thresholds': {'stop_speed': STOP_SPEED, 'hard_decel': HARD_DECEL,
                        'soft_decel': SOFT_DECEL},
         'frames': rows}, indent=1))
    print(f'\n[INFO] saved -> {args.out}')
    print('\nThe reference is the human\'s realised speed profile, a PROXY: it '
          'rewards imitation,\nnot safety, and cannot see why a driver slowed. '
          'Read it as a regression detector.')


if __name__ == '__main__':
    main()
