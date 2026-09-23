#!/usr/bin/env python3
"""Does a better 3-D detector produce a better driving decision?

`ablate_gt_detections.py` answered the endpoints: BEVFormer's own detections
versus perfect ground truth. This fills in the middle with two real detectors, so
the question stops being "does detection quality matter at all" and becomes "what
is the shape of the curve".

    detector                mAP (mini_val)   boxes/frame
    BEVFormer-Tiny          0.163            136
    BEVFusion (robust)      0.468             98
    BEVFusion (MIT, det)    0.578            151
    ground truth            1.000             --

WHAT IS HELD CONSTANT
---------------------
Everything except the detection text: same frames, same camera image, same
stage-1 light state, same prompt template, same decoding parameters. The text is
produced by `_format_detection_rows` for every arm, so a difference measures
detection quality rather than layout -- the same load-bearing detail as the GT
ablation.

The BEV raster is NOT sent. The shipped pipeline includes it, so this is a
deliberate deviation, for two reasons. Rendering it identically across detectors
would mean re-encoding BEVFusion's boxes into BEVFormer's `reg_preds` tensor
layout including the SECOND yaw convention -- a conversion that has produced
silent 90-degree errors in this repo more than once. And holding it at
BEVFormer's version for all four arms would feed every arm the SAME picture of a
DIFFERENT detector's scene, which is worse than omitting it. The row sweep
already established the text dominates the raster badly enough to suppress
camera light-reading entirely, so this drops the weak channel and varies the
strong one.

STAGE 1 RUNS ONCE PER FRAME, NOT ONCE PER ARM
---------------------------------------------
Light state is read from the camera with no detection text in context, so it
cannot depend on which detector produced the boxes. Computing it per arm would
burn 4x the calls and introduce sampling noise on a quantity that is
detector-invariant by construction. It is computed once and shared.

This has a consequence worth predicting before looking: on any frame where the
light is decisive, all four arms must agree. Divergence can only appear where the
light is absent or permissive. The GT ablation saw exactly this -- 6/6 agreement
with a light present, 33% without one -- so results are reported split both ways.

Usage:
    conda run -n simple_bev_vldrive python tools/compare_detectors.py \
        --arm "bevformer=/path/results_mini_val.json" \
        --arm "bevfusion-robust=../bevfusion_vldrive/BEVFusion_robust_vl/eval_out/results_nusc.json" \
        --arm "bevfusion-mit=../bevfusion_vldrive/bevfusion_vl/eval_out_det/results_nusc.json" \
        --gt --max-frames 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name
from pyquaternion import Quaternion

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from vis_infer import (_format_detection_rows, _parse_response,
                       _query_ollama_streaming, build_prompt, front_camera_b64,
                       light_crop_b64, query_light)

from dataroot import default_dataroot
from score_decisions import ego_speed_profile, reference_decision

GT_ARM = 'ground-truth'


def load_results(path: str) -> dict[str, list]:
    """nuScenes submission JSON -> {sample_token: [box dict]}."""
    with open(path) as f:
        return json.load(f)['results']


def _ego_rotation(nusc, sample_token: str):
    """(R, t) taking a GLOBAL point to the ego frame via R.T @ (p - t)."""
    sample = nusc.get('sample', sample_token)
    ep = nusc.get('ego_pose', nusc.get('sample_data',
                                       sample['data']['LIDAR_TOP'])['ego_pose_token'])
    return Quaternion(ep['rotation']).rotation_matrix, np.asarray(ep['translation'])


def boxes_to_items(nusc, sample_token: str, boxes: list) -> list[tuple]:
    """Submission-format boxes -> ego-frame (class, x, y, vx, vy, score) tuples.

    Detection results are GLOBAL frame, exactly like nuScenes annotations, so the
    rotation here is the same one `gt_detection_items` applies. Velocity is also
    global and needs the same rotation; skipping that is the Sparse4D temporal bug
    in miniature.
    """
    R, t = _ego_rotation(nusc, sample_token)
    items = []
    for b in boxes:
        name = b.get('detection_name')
        if name is None:
            continue
        p = R.T @ (np.asarray(b['translation']) - t)
        v = np.asarray(list(b.get('velocity') or (0.0, 0.0)) + [0.0])
        if np.isnan(v).any():
            v = np.zeros(3)
        v_ego = R.T @ v
        items.append((name, float(p[0]), float(p[1]),
                      float(v_ego[0]), float(v_ego[1]),
                      float(b.get('detection_score', 1.0))))
    return items


def gt_items(nusc, sample_token: str) -> list[tuple]:
    """Ground-truth annotations in the same ego-frame tuple format."""
    R, t = _ego_rotation(nusc, sample_token)
    items = []
    for ann_tok in nusc.get('sample', sample_token)['anns']:
        ann = nusc.get('sample_annotation', ann_tok)
        name = category_to_detection_name(ann['category_name'])
        if name is None:
            continue
        p = R.T @ (np.asarray(ann['translation']) - t)
        v = nusc.box_velocity(ann_tok)
        if np.isnan(v).any():
            v = np.zeros(3)
        v_ego = R.T @ v
        items.append((name, float(p[0]), float(p[1]),
                      float(v_ego[0]), float(v_ego[1]), 1.0))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', action='append', default=[],
                    metavar='LABEL=PATH',
                    help='detector arm; repeatable. PATH is a nuScenes results JSON')
    ap.add_argument('--gt', action='store_true',
                    help='add a ground-truth arm (the ceiling)')
    ap.add_argument('--dataroot', default=default_dataroot())
    ap.add_argument('--max-frames', type=int, default=0, help='0 = all shared frames')
    ap.add_argument('--max-rows', type=int, default=5)
    ap.add_argument('--score-thr', type=float, default=0.25,
                    help='drop boxes below this detection score')
    ap.add_argument('--ollama-model', default='qwen2.5vl:7b')
    ap.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    ap.add_argument('--ollama-timeout', type=int, default=180)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise-floor', action='store_true', default=True,
                    help='re-run the FIRST arm over the same frames and report its '
                         'self-agreement. Any arm-to-arm gap smaller than this floor '
                         'is sampling noise, not detector quality.')
    ap.add_argument('--no-noise-floor', dest='noise_floor', action='store_false')
    ap.add_argument('--out', default=str(ROOT / 'eval_results' / 'detector_comparison.json'))
    args = ap.parse_args()

    arms: dict[str, dict] = {}
    for spec in args.arm:
        if '=' not in spec:
            raise SystemExit(f'--arm needs LABEL=PATH, got {spec!r}')
        label, path = spec.split('=', 1)
        arms[label] = load_results(path)
    if args.gt:
        arms[GT_ARM] = None                     # generated per frame
    if len(arms) < 2:
        raise SystemExit('need at least two arms to compare')

    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)

    # Only frames every detector actually reported on -- an arm missing a token
    # would otherwise be silently scored on a different frame set.
    shared = None
    for label, res in arms.items():
        if res is None:
            continue
        toks = set(res)
        shared = toks if shared is None else (shared & toks)
    shared = sorted(shared)

    # Order by (scene, frame) so the printout reads in time order, and attach the
    # human reference decision from the ego's realised speed profile.
    order, ref_of = {}, {}
    for si in range(len(nusc.scene)):
        toks, speeds = ego_speed_profile(nusc, si)
        for k, t in enumerate(toks):
            order[t] = (si, k)
            ref, _ = reference_decision(speeds, k)
            ref_of[t] = ref
    shared = [t for t in shared if t in order]
    shared.sort(key=lambda t: order[t])
    if args.max_frames:
        shared = shared[:args.max_frames]

    print(f'[INFO] {len(arms)} arms x {len(shared)} shared frames '
          f'= {len(arms) * len(shared)} decision calls'
          + (f' (+{len(shared)} noise-floor)' if args.noise_floor else ''))
    print(f'[INFO] arms: {", ".join(arms)}')

    def decide(cam_b64, det_text, light) -> str:
        raw = _query_ollama_streaming(
            [cam_b64], args.ollama_model, args.ollama_url, args.ollama_timeout,
            on_update=lambda _t: None,
            # prev=None deliberately: carrying the previous decision chains frames
            # within an arm, so one flip at frame 0 cascades and the frames stop
            # being independent observations. The shipped pipeline keeps it on.
            prompt=build_prompt(det_text, with_front_cam=True,
                                light_state=light, prev=None),
            temperature=args.temperature, seed=args.seed)
        _, decision, _ = _parse_response(raw)
        return decision

    rows, t0 = [], time.time()
    for i, tok in enumerate(shared):
        cam = front_camera_b64(nusc, tok, args.dataroot)
        crop = light_crop_b64(nusc, tok, args.dataroot)
        # Detector-invariant by construction: no detection text is in context.
        light = query_light(cam, args.ollama_model, args.ollama_url,
                            args.ollama_timeout, crop_b64=crop)

        row = {'token': tok, 'scene': nusc.scene[order[tok][0]]['name'],
               'frame': order[tok][1], 'light': light, 'ref': ref_of.get(tok),
               'decisions': {}, 'n_boxes': {}}

        for label, res in arms.items():
            items = (gt_items(nusc, tok) if res is None
                     else boxes_to_items(nusc, tok, res.get(tok, [])))
            items = [it for it in items if it[5] >= args.score_thr]
            row['n_boxes'][label] = len(items)
            det = _format_detection_rows(items, max_rows=args.max_rows)
            row['decisions'][label] = decide(cam, det, light)

        if args.noise_floor:
            first = next(iter(arms))
            res = arms[first]
            items = (gt_items(nusc, tok) if res is None
                     else boxes_to_items(nusc, tok, res.get(tok, [])))
            items = [it for it in items if it[5] >= args.score_thr]
            row['decisions']['__replica__'] = decide(
                cam, _format_detection_rows(items, max_rows=args.max_rows), light)

        rows.append(row)
        el = time.time() - t0
        print(f'  [{i + 1}/{len(shared)}] {row["scene"]} #{row["frame"]:02d} '
              f'light={light:6s} ' +
              '  '.join(f'{k}={v}' for k, v in row['decisions'].items() if k != '__replica__')
              + f'   ({el / (i + 1):.1f}s/frame)')

    report(rows, list(arms), args)


def report(rows, labels, args) -> None:
    n = len(rows)
    if not n:
        raise SystemExit('no frames scored')
    print(f'\n{"=" * 72}\n{n} frames, {len(labels)} arms\n{"=" * 72}')

    floor = None
    if any('__replica__' in r['decisions'] for r in rows):
        first = labels[0]
        same = sum(r['decisions'][first] == r['decisions']['__replica__'] for r in rows)
        floor = same / n
        print(f'\nNOISE FLOOR  {first} vs itself: {same}/{n} ({floor:.1%})')
        if floor < 1.0:
            print('  Decoding is not deterministic on these frames. Any arm-to-arm')
            print('  agreement above this floor is indistinguishable from noise.')

    print('\nDECISION DISTRIBUTION')
    for lab in labels:
        c = Counter(r['decisions'][lab] for r in rows)
        print(f'  {lab:22s} ' + '  '.join(f'{k} {v}' for k, v in c.most_common()))

    print('\nPAIRWISE AGREEMENT')
    print('  ' + ' ' * 22 + ''.join(f'{l[:11]:>13s}' for l in labels))
    for a in labels:
        cells = []
        for b in labels:
            agree = sum(r['decisions'][a] == r['decisions'][b] for r in rows)
            cells.append(f'{agree / n:>12.1%} ')
        print(f'  {a:22s}' + ''.join(cells))

    # The prediction stated in the module docstring: with a decisive light, the
    # detector cannot matter, because it is not in stage 1's context at all.
    lit = [r for r in rows if r['light'] in ('red', 'yellow')]
    unlit = [r for r in rows if r['light'] not in ('red', 'yellow')]
    print(f'\nSPLIT BY LIGHT   decisive (red/yellow): {len(lit)}   '
          f'absent/green: {len(unlit)}')
    for name, subset in (('decisive light', lit), ('no decisive light', unlit)):
        if not subset:
            continue
        full = sum(len({r['decisions'][l] for l in labels}) == 1 for r in subset)
        print(f'  {name:20s} all {len(labels)} arms identical: '
              f'{full}/{len(subset)} ({full / len(subset):.1%})')

    scored = [r for r in rows if r['ref']]
    if scored:
        print(f'\nAGREEMENT WITH THE HUMAN  ({len(scored)} frames with a usable '
              f'reference)')
        for lab in labels:
            ok = sum(r['decisions'][lab] == r['ref'] for r in scored)
            print(f'  {lab:22s} {ok}/{len(scored)} ({ok / len(scored):.1%})')
        print('  Reference is the ego\'s realised speed profile -- a proxy that '
              'rewards\n  imitation, not safety. Read as a regression detector.')

    print('\nMEAN BOXES ABOVE THRESHOLD')
    for lab in labels:
        print(f'  {lab:22s} {np.mean([r["n_boxes"][lab] for r in rows]):6.1f}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {'n_frames': n, 'arms': labels, 'noise_floor': floor,
         'temperature': args.temperature, 'seed': args.seed,
         'score_thr': args.score_thr, 'max_rows': args.max_rows,
         'bev_raster_sent': False, 'prev_context': False,
         'frames': rows}, indent=1))
    print(f'\n[INFO] saved -> {out}')


if __name__ == '__main__':
    main()
