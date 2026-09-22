#!/usr/bin/env python3
"""Does 3-D detection quality bind the driving decision?

The end-to-end scorer showed every failure sitting in stage-1 light reading, with
zero stage-2 reasoning faults — which raises the question of whether a better
detector (BEVFusion, say) would move the decision at all. Swapping detectors is a
week of work; this answers it in an afternoon.

Method: hold EVERYTHING constant except the detection text. Same BEV canvas, same
camera image, same stage-1 light state, same prompt template, same frame. One arm
gets BEVFormer's predictions, the other gets nuScenes ground-truth boxes rendered
through the identical formatter. If the decisions do not diverge, detector quality
is not the ceiling and porting a better one buys nothing here.

Sharing `_format_detection_rows` between both arms is load-bearing: if the two
texts differed in layout, this would measure formatting rather than detection.

Usage:
    conda run -n simple_bev_vldrive python tools/ablate_gt_detections.py \
        --scenes 4 --max-frames 6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from nuscenes.eval.detection.utils import category_to_detection_name
from pyquaternion import Quaternion

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from data import NuScenesMiniLoader
from eval import _build_remap
from model import BEVFormerTiny
from vis_infer import (DecisionSmoother, _encode_image, _format_detection_rows,
                       _get_device, _parse_response, _query_ollama_streaming,
                       build_prompt, detection_items, detections_to_text,
                       front_camera_b64, query_light)
from visualizer import build_scene_canvas

from dataroot import default_dataroot


def gt_detections_to_text(nusc, sample_token: str, max_rows: int = 5,
                          forward_only: bool = True) -> str:
    """Ground-truth boxes in the same format the predicted ones use.

    nuScenes annotations live in the GLOBAL frame, so each box is rotated into the
    ego frame here (+x forward, +y left) — the same frame `detections_to_text`
    produces from the LiDAR frame. Velocity comes from `box_velocity`, which is
    also global and needs the same rotation; forgetting that is the Sparse4D
    temporal bug in miniature.

    GT has no confidence, so rows report conf 1.00. That is a real (small)
    asymmetry between the arms and is noted in the output.
    """
    sample = nusc.get('sample', sample_token)
    ep = nusc.get('ego_pose', nusc.get('sample_data',
                                       sample['data']['LIDAR_TOP'])['ego_pose_token'])
    R = Quaternion(ep['rotation']).rotation_matrix
    t = np.asarray(ep['translation'])

    items = []
    for ann_tok in sample['anns']:
        ann = nusc.get('sample_annotation', ann_tok)
        name = category_to_detection_name(ann['category_name'])
        if name is None:
            continue
        p_ego = R.T @ (np.asarray(ann['translation']) - t)
        v_glob = nusc.box_velocity(ann_tok)
        if np.isnan(v_glob).any():
            v_glob = np.zeros(3)
        v_ego = R.T @ v_glob
        items.append((name, float(p_ego[0]), float(p_ego[1]),
                      float(v_ego[0]), float(v_ego[1]), 1.0))
    return _format_detection_rows(items, max_rows=max_rows,
                                  forward_only=forward_only)


def gt_detection_items(nusc, sample_token: str) -> list[tuple]:
    """Ground-truth boxes as ego-frame (class, x, y, vx, vy, score) tuples."""
    sample = nusc.get('sample', sample_token)
    ep = nusc.get('ego_pose', nusc.get('sample_data',
                                       sample['data']['LIDAR_TOP'])['ego_pose_token'])
    R = Quaternion(ep['rotation']).rotation_matrix
    t = np.asarray(ep['translation'])
    items = []
    for ann_tok in sample['anns']:
        ann = nusc.get('sample_annotation', ann_tok)
        name = category_to_detection_name(ann['category_name'])
        if name is None:
            continue
        p_ego = R.T @ (np.asarray(ann['translation']) - t)
        v = nusc.box_velocity(ann_tok)
        if np.isnan(v).any():
            v = np.zeros(3)
        v_ego = R.T @ v
        items.append((name, float(p_ego[0]), float(p_ego[1]),
                      float(v_ego[0]), float(v_ego[1]), 1.0))
    return items


def swap_fields(pred_items, gt_items, velocity=False, conf=False,
                max_dist: float = 3.0):
    """Predicted positions with selected fields replaced from the nearest GT box.

    Isolates WHICH detector output drives the decision. The full-GT arm showed
    predicted detections tripling PROCEED against ground truth, and the diffs
    pointed at velocity (10.9 vs 6.0 m/s closing) rather than position (within a
    metre) — consistent with eval, where trans_err beats the paper but vel_err is
    ~0.90 against 0.657. This arm tests that directly.

    Matching is nearest same-class within `max_dist`. An unmatched prediction is a
    likely false positive with no GT counterpart, so it keeps its own values; the
    match rate is reported so an arm built mostly from unmatched boxes is visible
    rather than silently meaningless.
    """
    out, matched = [], 0
    for (name, x, y, vx, vy, score) in pred_items:
        best = None
        for (gn, gx, gy, gvx, gvy, _) in gt_items:
            if gn != name:
                continue
            d = math.hypot(gx - x, gy - y)
            if d <= max_dist and (best is None or d < best[0]):
                best = (d, gvx, gvy)
        if best is not None:
            matched += 1
        nvx, nvy = (best[1], best[2]) if (best and velocity) else (vx, vy)
        out.append((name, x, y, nvx, nvy, 1.0 if conf else score))
    return out, matched, len(pred_items)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=default_dataroot())
    ap.add_argument('--checkpoint',
                    default='model/checkpoints/bevformer_tiny_fp16_epoch_24.pth')
    ap.add_argument('--scenes', type=int, nargs='+', default=[4])
    ap.add_argument('--max-frames', type=int, default=6)
    ap.add_argument('--score-thr', type=float, default=0.25)
    ap.add_argument('--cam-width', type=int, default=640)
    ap.add_argument('--ollama-url', default='http://localhost:11434')
    ap.add_argument('--ollama-model', default='qwen2.5vl:7b')
    ap.add_argument('--ollama-timeout', type=int, default=180)
    ap.add_argument('--arms', nargs='+',
                    default=['pred', 'gt_vel', 'gt_conf', 'gt'],
                    choices=['pred', 'gt_vel', 'gt_conf', 'gt_vel_conf', 'gt'],
                    help='pred=baseline; gt_vel=predicted positions with GT '
                         'velocities; gt_conf=predicted values at conf 1.00; '
                         'gt=full ground truth')
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='MUST stay 0 for ablations; 0.1 makes arms incomparable')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--hysteresis', action='store_true',
                    help='Enable the decision smoother. OFF by default here: the '
                         'latch propagates frame 0 across a whole scene, so every '
                         'frame stops being an independent measurement and the '
                         'effective sample size collapses to the scene count.')
    ap.add_argument('--skip-determinism-check', action='store_true')
    ap.add_argument('--determinism-frames', type=int, default=3,
                    help='Frames to probe for determinism. One frame x 3 trials is '
                         'too weak: at temperature 0.1 per-frame divergence is '
                         'roughly 1-in-3, so a single-frame check passes by luck '
                         'about 44%% of the time (observed). Divergence is also '
                         'frame-dependent, so breadth beats depth here.')
    ap.add_argument('--determinism-trials', type=int, default=3)
    ap.add_argument('--noise-floor', dest='noise_floor', action='store_true',
                    default=True,
                    help='Re-run the FIRST arm a second time over the same frames '
                         'and report its self-agreement. This is the only valid '
                         'check: a pre-flight probe on other frames cannot certify '
                         'the comparison frames, because each prompt carries the '
                         'PREVIOUS decision, so one flip at frame 0 cascades. Any '
                         'arm-to-arm difference smaller than this floor is noise.')
    ap.add_argument('--no-noise-floor', dest='noise_floor', action='store_false')
    ap.add_argument('--prev-context', dest='prev_context', action='store_true',
                    default=False,
                    help='Carry the previous decision into each prompt. OFF for '
                         'ablations: it chains frames within an arm, so a '
                         'difference at frame 0 propagates and the frames stop '
                         'being independent observations. The shipped pipeline '
                         'keeps it ON; this deliberately differs to buy statistics.')
    ap.add_argument('--out', default=str(ROOT / 'eval_results' / 'gt_detection_ablation.json'))
    args = ap.parse_args()

    device = _get_device()
    model = BEVFormerTiny().to(device).eval()
    ck = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(_build_remap(ck.get('state_dict', ck)), strict=False)
    loader = NuScenesMiniLoader(args.dataroot)
    nusc = loader.nusc
    tmp = Path(args.out).parent
    tmp.mkdir(parents=True, exist_ok=True)

    def decide(images, det, light, prev):
        raw = _query_ollama_streaming(
            images, args.ollama_model, args.ollama_url, args.ollama_timeout,
            on_update=lambda _t: None,
            prompt=build_prompt(det, with_front_cam=len(images) > 1,
                                light_state=light, prev=prev),
            temperature=args.temperature, seed=args.seed)
        _, decision, _ = _parse_response(raw)
        return decision

    ARMS = {
        'pred':        dict(velocity=False, conf=False, full_gt=False),
        'gt_vel':      dict(velocity=True,  conf=False, full_gt=False),
        'gt_conf':     dict(velocity=False, conf=True,  full_gt=False),
        'gt_vel_conf': dict(velocity=True,  conf=True,  full_gt=False),
        'gt':          dict(velocity=False, conf=False, full_gt=True),
    }
    arms = [a for a in args.arms if a in ARMS]

    # ---- determinism gate -------------------------------------------------
    # Run one real frame twice through the SAME arm before measuring anything.
    # If the two answers differ, arm-to-arm differences are not attributable and
    # the run is aborted rather than producing numbers that look like findings.
    # This exists because an earlier version silently did exactly that: an arm
    # compared against itself agreed 0/6 across two runs, and a four-arm study
    # was built on top before anyone checked.
    if not args.skip_determinism_check:
        sidx0 = args.scenes[0]
        probes, checked = [], 0
        with torch.no_grad():
            for sample in loader.iter_scene(scene_idx=sidx0):
                if checked >= args.determinism_frames:
                    break
                res = model(sample['imgs'], sample['img_metas'], prev_bev=None)
                tok = sample['img_metas'][0]['sample_token']
                sd0 = nusc.get('sample_data',
                               nusc.get('sample', tok)['data']['LIDAR_TOP'])
                ep = nusc.get('ego_pose', sd0['ego_pose_token'])
                l2e0 = Quaternion(nusc.get('calibrated_sensor',
                                           sd0['calibrated_sensor_token'])['rotation']
                                  ).yaw_pitch_roll[0]
                canvas = build_scene_canvas(
                    res, ep, None,
                    patch_origin=(ep['translation'][0], ep['translation'][1]),
                    patch_range=150.0, canvas_size=512,
                    score_thr=args.score_thr, lidar2ego_yaw=l2e0)
                bp = tmp / '_det_check.png'
                cv2.imwrite(str(bp), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                cam0 = front_camera_b64(nusc, tok, args.dataroot,
                                        max_width=args.cam_width)
                imgs0 = [_encode_image(str(bp))] + ([cam0] if cam0 else [])
                det0 = _format_detection_rows(
                    detection_items(res['cls_logits'][0].cpu(),
                                    res['reg_preds'][0].cpu(),
                                    res['ref_pts'][0].cpu(),
                                    score_thr=args.score_thr, lidar2ego_yaw=l2e0))
                trials = [decide(imgs0, det0, 'none', None)
                          for _ in range(args.determinism_trials)]
                probes.append((checked, trials))
                checked += 1

        bad = [(f, t) for f, t in probes if len(set(t)) != 1]
        n_calls = sum(len(t) for _, t in probes)
        if bad:
            detail = '\n'.join(f'    frame {f}: {t}' for f, t in bad)
            raise SystemExit(
                f'\nABORT: decision stage is not deterministic at '
                f'temperature={args.temperature}.\n{detail}\n'
                f'Arm differences would be indistinguishable from sampling noise — '
                f'an arm compared against ITSELF once agreed 0/6 across two runs.\n'
                f'Use --temperature 0, or --skip-determinism-check to override.')
        print(f'[determinism] {n_calls} calls over {len(probes)} frames all '
              f'reproducible at temperature={args.temperature} -> arms comparable')

    rows, match_stats = [], [0, 0]
    for sidx in args.scenes:
        sc = nusc.scene[sidx]
        l2e = Quaternion(nusc.get('calibrated_sensor',
                                  nusc.get('sample_data',
                                           nusc.get('sample', sc['first_sample_token'])
                                           ['data']['LIDAR_TOP'])
                                  ['calibrated_sensor_token'])['rotation']
                         ).yaw_pitch_roll[0]
        print(f'\n=== scene {sidx}: {sc["name"]} ===')
        # Every arm needs its OWN smoother and decision history, or they leak
        # into each other through the prev-decision context and stop being
        # independent measurements.
        sm = {a: DecisionSmoother(
                  release_frames=2 if args.hysteresis else 0,
                  stop_latch=2 if args.hysteresis else 0) for a in arms}
        prev = {a: None for a in arms}
        prev_bev, n = None, 0

        with torch.no_grad():
            for sample in loader.iter_scene(scene_idx=sidx):
                if n >= args.max_frames:
                    break
                res = model(sample['imgs'], sample['img_metas'], prev_bev=prev_bev)
                prev_bev = res['bev_feat'].detach()
                tok = sample['img_metas'][0]['sample_token']

                ep = nusc.get('ego_pose',
                              nusc.get('sample_data',
                                       nusc.get('sample', tok)['data']['LIDAR_TOP'])
                              ['ego_pose_token'])
                canvas = build_scene_canvas(
                    res, ep, None,
                    patch_origin=(ep['translation'][0], ep['translation'][1]),
                    patch_range=150.0, canvas_size=512,
                    score_thr=args.score_thr, lidar2ego_yaw=l2e)
                bev_path = tmp / '_ablate_bev.png'
                cv2.imwrite(str(bev_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

                cam = front_camera_b64(nusc, tok, args.dataroot,
                                       max_width=args.cam_width)
                images = [_encode_image(str(bev_path))] + ([cam] if cam else [])
                # Stage 1 runs ONCE and every arm shares it: light state is not
                # under test here and re-querying would inject noise between arms.
                light = query_light(cam, args.ollama_model, args.ollama_url,
                                    args.ollama_timeout) if cam else 'none'

                pred_items = detection_items(res['cls_logits'][0].cpu(),
                                             res['reg_preds'][0].cpu(),
                                             res['ref_pts'][0].cpu(),
                                             score_thr=args.score_thr,
                                             lidar2ego_yaw=l2e)
                gt_items = gt_detection_items(nusc, tok)

                rec = {'scene': sc['name'], 'frame': n, 'token': tok, 'light': light}
                for a in arms:
                    cfg = ARMS[a]
                    if cfg['full_gt']:
                        items = gt_items
                    else:
                        items, m, tot = swap_fields(pred_items, gt_items,
                                                    velocity=cfg['velocity'],
                                                    conf=cfg['conf'])
                        if a == arms[0]:
                            match_stats[0] += m
                            match_stats[1] += tot
                    det = _format_detection_rows(items)
                    rec[f'det_{a}'] = det
                    rec[a] = sm[a].update(decide(images, det, light,
                                             prev[a] if args.prev_context else None))
                    prev[a] = {'decision': rec[a], 'light': light}

                print(f'  f{n:2d} light={light:<6s} ' +
                      '  '.join(f'{a}->{rec[a]:<10s}' for a in arms))
                rows.append(rec)
                n += 1

    # ---- noise floor ------------------------------------------------------
    floor = None
    if args.noise_floor:
        base = arms[0]
        print(f'\n[noise floor] re-running arm \'{base}\' over the same frames')
        repeat = {}
        for sidx in args.scenes:
            sc = nusc.scene[sidx]
            l2e = Quaternion(nusc.get('calibrated_sensor',
                                      nusc.get('sample_data',
                                               nusc.get('sample',
                                                        sc['first_sample_token'])
                                               ['data']['LIDAR_TOP'])
                                      ['calibrated_sensor_token'])['rotation']
                             ).yaw_pitch_roll[0]
            sm2 = DecisionSmoother(release_frames=2 if args.hysteresis else 0,
                                   stop_latch=2 if args.hysteresis else 0)
            prev2, prev_bev2, n2 = None, None, 0
            with torch.no_grad():
                for sample in loader.iter_scene(scene_idx=sidx):
                    if n2 >= args.max_frames:
                        break
                    res = model(sample['imgs'], sample['img_metas'],
                                prev_bev=prev_bev2)
                    prev_bev2 = res['bev_feat'].detach()
                    tok = sample['img_metas'][0]['sample_token']
                    src = next((r for r in rows if r['token'] == tok), None)
                    if src is None:
                        n2 += 1
                        continue
                    sd = nusc.get('sample_data',
                                  nusc.get('sample', tok)['data']['LIDAR_TOP'])
                    ep = nusc.get('ego_pose', sd['ego_pose_token'])
                    canvas = build_scene_canvas(
                        res, ep, None,
                        patch_origin=(ep['translation'][0], ep['translation'][1]),
                        patch_range=150.0, canvas_size=512,
                        score_thr=args.score_thr, lidar2ego_yaw=l2e)
                    bp = tmp / '_nf_bev.png'
                    cv2.imwrite(str(bp), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                    cam = front_camera_b64(nusc, tok, args.dataroot,
                                           max_width=args.cam_width)
                    imgs = [_encode_image(str(bp))] + ([cam] if cam else [])
                    d = sm2.update(decide(imgs, src[f'det_{base}'], src['light'],
                                          prev2 if args.prev_context else None))
                    prev2 = {'decision': d, 'light': src['light']}
                    repeat[tok] = d
                    n2 += 1
        same = sum(repeat.get(r['token']) == r[base] for r in rows)
        floor = same / max(len(rows), 1)
        print(f'[noise floor] arm \'{base}\' agrees with itself on '
              f'{same}/{len(rows)} ({floor:.0%})')
        if floor < 1.0:
            print(f'[noise floor] !! {(1-floor):.0%} of frames flip on a rerun. '
                  f'Arm differences at or below that are NOT attributable.')

    print('\n' + '=' * 70)
    print('DECISION DISTRIBUTION BY ARM')
    print('=' * 70)
    n_scenes = len({r['scene'] for r in rows})
    if args.hysteresis:
        print(f'  [!] hysteresis ON: the latch carries frame 0 across each scene, '
              f'so effective n is {n_scenes} scenes, NOT {len(rows)} frames.')
    else:
        print(f'  n = {len(rows)} frames across {n_scenes} scenes, '
              f'hysteresis off, temperature {args.temperature}')
    base = arms[0]
    for a in arms:
        dist = Counter(r[a] for r in rows)
        agree = sum(r[a] == r['gt'] for r in rows) if 'gt' in arms else None
        line = f'  {a:<12s} ' + '  '.join(f'{k}={v}' for k, v in dist.most_common())
        if agree is not None and a != 'gt':
            line += f'   | agrees with gt on {agree}/{len(rows)}'
        print(line)

    if match_stats[1]:
        print(f'\n  GT match rate for swapped arms: {match_stats[0]}/{match_stats[1]} '
              f'({match_stats[0]/match_stats[1]:.0%}) of predicted boxes had a GT '
              f'counterpart within 3 m; unmatched boxes kept their own values.')

    if 'gt' in arms and 'gt_vel' in arms:
        d_pred = sum(r[base] != r['gt'] for r in rows)
        d_vel = sum(r['gt_vel'] != r['gt'] for r in rows)
        print(f'\n  disagreement with full GT: {base} {d_pred}/{len(rows)} -> '
              f'gt_vel {d_vel}/{len(rows)}')
        if d_pred:
            closed = (d_pred - d_vel) / d_pred
            print(f'  swapping ONLY velocity closes {closed:.0%} of the gap '
                  f'-> velocity is {"the dominant" if closed >= 0.5 else "a partial"} '
                  f'cause')

    Path(args.out).write_text(json.dumps(
        {'n': len(rows), 'arms': arms,
         'distributions': {a: dict(Counter(r[a] for r in rows)) for a in arms},
         'gt_match_rate': (match_stats[0] / match_stats[1]) if match_stats[1] else None,
         'noise_floor_self_agreement': floor,
         'temperature': args.temperature, 'hysteresis': args.hysteresis,
         'frames': rows}, indent=1))
    print(f'\n[INFO] saved -> {args.out}')


if __name__ == '__main__':
    main()
