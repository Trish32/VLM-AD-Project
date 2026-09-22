#!/usr/bin/env python3
"""Score the VLM's traffic-light reading against hand labels.

Consumes the JSONL exported by annotate.html and runs the same stage-1 query the
pipeline uses (`vis_infer.query_light` — camera alone, one question), so the
number produced here is the number the pipeline actually gets.

Reports accuracy overall and bucketed by range, because the open question in the
README is how stage-1 reliability degrades with distance: on the red-light test it
hit at 34 m and 29 m but missed at 32 m, so range alone did not explain the miss.

Two exclusion rules, and the second has a carve-out that matters:

  * `unknown` labels are excluded rather than counted wrong — a frame the
    annotator could not read is not evidence about the model.
  * `governs_ego: false` is excluded ONLY on frames that contain a fixture, where
    it means "that signal governs cross traffic". On a NEGATIVE frame (no fixture
    in view at all) the field is meaningless, and excluding on it would silently
    drop every false-positive test — which is the whole reason negatives exist.

Negatives are scored as their own group and reported as a false-positive rate.
This is not hypothetical: the model has been observed answering "LIGHT: red" on a
payload with no camera attached, and a phantom red forces a spurious STOP.

Usage:
    conda run -n simple_bev_vldrive python tools/score_light_labels.py \
        --labels ~/Downloads/light_labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from nuscenes import NuScenes

from vis_infer import front_camera_b64, light_crop_b64, query_light

from dataroot import default_dataroot

RANGE_BUCKETS = [(0, 15), (15, 25), (25, 35), (35, 50), (50, 1e9)]

# Decisions that count as "stopping" when a red light governs the lane.
STOPPING = {'STOP', 'SLOW_DOWN'}


def run_e2e(rows: list[dict], args, is_negative) -> dict:
    """Run the FULL two-stage pipeline per labelled frame and score the DECISION.

    The light scorer measures perception — can the model read the signal. This
    measures behaviour: given a red light that was correctly read, does the
    pipeline actually stop?  The two come apart in both directions, and neither
    is visible from the other:

      * scene-0757 frame 1: stage 1 MISSED the light, yet the decision was STOP
        anyway (the cones carried it). Right answer, wrong reason.
      * the single-call design read the light correctly in isolation and still
        answered PROCEED once detection text was in context.

    So the interesting output is not one accuracy but a decomposition: light
    correct/incorrect crossed with decision correct/incorrect.

    BEVFormer fuses temporally through `prev_bev`, so frames cannot be visited in
    label order — each scene is walked from its first keyframe with the cache
    carried forward, and the VLM runs only where a label exists.
    """
    import cv2
    import torch
    from pyquaternion import Quaternion

    from data import NuScenesMiniLoader
    from eval import _build_remap
    from model import BEVFormerTiny
    from vis_infer import (_encode_image, _get_device, _parse_response,
                           _query_ollama_streaming, build_prompt,
                           detections_to_text)
    from visualizer import build_scene_canvas

    device = _get_device()
    model = BEVFormerTiny().to(device).eval()
    ck = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(_build_remap(ck.get('state_dict', ck)), strict=False)
    loader = NuScenesMiniLoader(args.dataroot)
    nusc = loader.nusc

    want = {r['sample_token']: r for r in rows}
    scene_idx = {s['name']: i for i, s in enumerate(nusc.scene)}
    tmp_bev = Path(args.out).parent / '_e2e_bev.png'
    tmp_bev.parent.mkdir(parents=True, exist_ok=True)

    out: list[dict] = []
    for scene_name in sorted({r['scene'] for r in rows}):
        idx = scene_idx.get(scene_name)
        if idx is None:
            continue
        sc = nusc.scene[idx]
        l2e = Quaternion(nusc.get('calibrated_sensor',
                                  nusc.get('sample_data',
                                           nusc.get('sample', sc['first_sample_token'])
                                           ['data']['LIDAR_TOP'])
                                  ['calibrated_sensor_token'])['rotation']
                         ).yaw_pitch_roll[0]
        prev_bev = None
        with torch.no_grad():
            for sample in loader.iter_scene(scene_idx=idx):
                res = model(sample['imgs'], sample['img_metas'], prev_bev=prev_bev)
                prev_bev = res['bev_feat'].detach()
                tok = sample['img_metas'][0]['sample_token']
                lab = want.get(tok)
                if lab is None:
                    continue          # still had to run it, for prev_bev continuity

                ep = nusc.get('ego_pose',
                              nusc.get('sample_data',
                                       nusc.get('sample', tok)['data']['LIDAR_TOP'])
                              ['ego_pose_token'])
                canvas = build_scene_canvas(
                    res, ep, None,
                    patch_origin=(ep['translation'][0], ep['translation'][1]),
                    patch_range=150.0, canvas_size=512, score_thr=0.25,
                    lidar2ego_yaw=l2e)
                cv2.imwrite(str(tmp_bev), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

                det = detections_to_text(res['cls_logits'][0].cpu(),
                                         res['reg_preds'][0].cpu(),
                                         res['ref_pts'][0].cpu(),
                                         score_thr=0.25, lidar2ego_yaw=l2e)
                cam = front_camera_b64(nusc, tok, args.dataroot,
                                       max_width=args.cam_width)
                pred_light = query_light(cam, args.ollama_model, args.ollama_url,
                                         args.ollama_timeout) if cam else 'none'
                images = [_encode_image(str(tmp_bev))] + ([cam] if cam else [])
                raw = _query_ollama_streaming(
                    images, args.ollama_model, args.ollama_url, args.ollama_timeout,
                    on_update=lambda _t: None,
                    prompt=build_prompt(det, with_front_cam=cam is not None,
                                        light_state=pred_light))
                _, decision, _ = _parse_response(raw)

                gt = lab['light']
                rec = {**lab, 'pred_light': pred_light, 'decision': decision,
                       'light_ok': pred_light == gt,
                       'negative': is_negative(lab)}
                out.append(rec)
                print(f'  {scene_name} f{lab["frame"]:3d}  gt={gt:<6s} '
                      f'light={pred_light:<6s} -> {decision:<10s} '
                      f'{"" if rec["light_ok"] else "(light MISS)"}')
    return {'frames': out}


def report_e2e(frames: list[dict]) -> dict:
    """Decompose behaviour into stage-1 and stage-2 failure modes."""
    reds = [f for f in frames if f['light'] == 'red']
    clear = [f for f in frames if f['light'] in ('green', 'none')]

    print('\n' + '=' * 60)
    print('END-TO-END: does the decision follow the true light state?')
    print('=' * 60)

    summary: dict = {}
    if reds:
        stopped = [f for f in reds if f['decision'] in STOPPING]
        hard = [f for f in reds if f['decision'] == 'STOP']
        print(f'\nTrue RED ({len(reds)} frames) — the safety-critical case:')
        print(f'  stopped (STOP or SLOW_DOWN) : {len(stopped)}/{len(reds)} '
              f'({len(stopped)/len(reds):.1%})')
        print(f'  full STOP                   : {len(hard)}/{len(reds)} '
              f'({len(hard)/len(reds):.1%})')
        # The decomposition that matters: was a failure perception or reasoning?
        lit_ok = [f for f in reds if f['light_ok']]
        stage2_fail = [f for f in lit_ok if f['decision'] not in STOPPING]
        stage1_fail = [f for f in reds if not f['light_ok']]
        rescued = [f for f in stage1_fail if f['decision'] in STOPPING]
        print(f'  light read correctly        : {len(lit_ok)}/{len(reds)}')
        print(f'    of those, failed to stop  : {len(stage2_fail)}  <- stage-2 fault')
        print(f'  light MISSED                : {len(stage1_fail)}')
        print(f'    but stopped anyway        : {len(rescued)}  '
              f'<- right answer, wrong reason')
        summary['red'] = {
            'n': len(reds), 'stopped': len(stopped), 'full_stop': len(hard),
            'light_ok': len(lit_ok), 'stage2_failures': len(stage2_fail),
            'stage1_failures': len(stage1_fail), 'rescued_by_other_cues': len(rescued),
        }

    if clear:
        spurious = [f for f in clear if f['decision'] == 'STOP']
        print(f'\nTrue GREEN/NONE ({len(clear)} frames) — over-conservatism:')
        print(f'  spurious full STOP : {len(spurious)}/{len(clear)} '
              f'({len(spurious)/len(clear):.1%})')
        phantom = [f for f in clear if f['pred_light'] == 'red']
        if phantom:
            print(f'  phantom RED read   : {len(phantom)}  '
                  f'<- each forces a spurious stop')
        summary['clear'] = {'n': len(clear), 'spurious_stop': len(spurious),
                            'phantom_red': len(phantom)}

    dist = Counter(f['decision'] for f in frames)
    print('\ndecision distribution:', dict(dist.most_common()))
    summary['decision_distribution'] = dict(dist)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', required=True, help='light_labels.jsonl from annotate.html')
    ap.add_argument('--dataroot', default=default_dataroot())
    ap.add_argument('--ollama-url', default='http://localhost:11434')
    ap.add_argument('--ollama-model', default='qwen2.5vl:7b')
    ap.add_argument('--ollama-timeout', type=int, default=120)
    ap.add_argument('--cam-width', type=int, default=640)
    ap.add_argument('--no-light-crop', dest='light_crop', action='store_false',
                    default=True,
                    help='Score WITHOUT the map-projected light crop, for A/B')
    ap.add_argument('--mode', choices=['light', 'e2e', 'both'], default='light',
                    help="'light' scores stage 1 alone (fast, ~1 s/frame). 'e2e' "
                         "runs BEVFormer plus both VLM stages and scores the "
                         "DECISION (~10-15 s/frame). 'both' does each in turn.")
    ap.add_argument('--checkpoint',
                    default='model/checkpoints/bevformer_tiny_fp16_epoch_24.pth')
    ap.add_argument('--limit', type=int, default=0,
                    help='Score only the first N labels (0 = all); e2e is slow')
    ap.add_argument('--out', default=str(ROOT / 'eval_results' / 'light_scores.json'))
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.labels).read_text().splitlines() if l.strip()]
    print(f'[INFO] {len(rows)} labelled frames')

    # A frame with no fixture in view is a NEGATIVE: `governs_ego` is meaningless
    # there, and excluding it on that basis would drop every false-positive test —
    # which is the entire reason the negatives exist. Only cross-traffic signals on
    # frames that DO contain a fixture get excluded.
    def is_negative(r: dict) -> bool:
        return bool(r.get('negative')) or r.get('range_m') is None

    scored = [r for r in rows
              if r['light'] != 'unknown'
              and (is_negative(r) or r.get('governs_ego', True))]
    skipped_unknown = sum(r['light'] == 'unknown' for r in rows)
    skipped_cross = sum(r['light'] != 'unknown' and not is_negative(r)
                        and not r.get('governs_ego', True) for r in rows)
    n_neg = sum(is_negative(r) for r in scored)
    print(f'[INFO] scoring {len(scored)} ({len(scored) - n_neg} with a light, '
          f'{n_neg} negatives)  '
          f'(excluded {skipped_unknown} unknown, {skipped_cross} cross-traffic)')
    if not scored:
        raise SystemExit('nothing to score')
    if args.limit:
        scored = scored[:args.limit]
        print(f'[INFO] --limit {args.limit}: scoring {len(scored)}')

    e2e_summary, e2e_frames = None, []
    if args.mode in ('e2e', 'both'):
        print('\n[INFO] end-to-end pass (BEVFormer + both VLM stages)')
        e2e_frames = run_e2e(scored, args, is_negative)['frames']
        e2e_summary = report_e2e(e2e_frames)
        if args.mode == 'e2e':
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({'mode': 'e2e', 'n_scored': len(e2e_frames),
                                       'e2e': e2e_summary,
                                       'frames': e2e_frames}, indent=1))
            print(f'\n[INFO] saved -> {out}')
            return

    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)

    results, confusion = [], Counter()
    by_bucket: dict[tuple, list[bool]] = defaultdict(list)

    for i, r in enumerate(scored, 1):
        cam = front_camera_b64(nusc, r['sample_token'], args.dataroot,
                               max_width=args.cam_width)
        if cam is None:
            continue
        crop = (light_crop_b64(nusc, r['sample_token'], args.dataroot)
                if args.light_crop else None)
        pred = query_light(cam, args.ollama_model, args.ollama_url,
                           args.ollama_timeout, crop_b64=crop)
        ok = pred == r['light']
        confusion[(r['light'], pred)] += 1
        # Negatives have no range to bucket by; they are scored as their own group.
        if not is_negative(r):
            for lo, hi in RANGE_BUCKETS:
                if lo <= r['range_m'] < hi:
                    by_bucket[(lo, hi)].append(ok)
                    break
        results.append({**r, 'pred': pred, 'correct': ok,
                        'negative': is_negative(r)})
        rng = '  n/a ' if is_negative(r) else f'{r["range_m"]:5.1f}'
        print(f'  [{i:3d}/{len(scored)}] {r["scene"]} f{r["frame"]:3d} '
              f'{rng} m  gt={r["light"]:<6s} pred={pred:<6s} '
              f'{"OK" if ok else "MISS"}')

    n = len(results)
    acc = sum(x['correct'] for x in results) / max(n, 1)
    print(f'\n=== accuracy {acc:.3f}  ({sum(x["correct"] for x in results)}/{n}) ===')

    print('\nby range:')
    for (lo, hi), v in sorted(by_bucket.items()):
        if v:
            hi_s = '+' if hi > 1e8 else f'-{hi:.0f}'
            print(f'  {lo:3.0f}{hi_s:>4s} m : {sum(v)/len(v):.3f}  (n={len(v)})')

    print('\nconfusion (gt -> pred):')
    for (gt, pred), c in confusion.most_common():
        mark = '' if gt == pred else '   <-'
        print(f'  {gt:<7s} -> {pred:<7s} {c:4d}{mark}')

    # The failure that matters most for safety: a real red read as anything else.
    red_missed = sum(c for (gt, pred), c in confusion.items()
                     if gt == 'red' and pred != 'red')
    red_total = sum(c for (gt, _), c in confusion.items() if gt == 'red')
    if red_total:
        print(f'\nRED recall: {(red_total - red_missed)/red_total:.3f} '
              f'({red_total - red_missed}/{red_total}) — misses here are the '
              f'safety-relevant ones')

    # False positives: frames with no fixture in view where a light was reported.
    negs = [x for x in results if x['negative']]
    fp = [x for x in negs if x['pred'] != 'none']
    if negs:
        print(f'\nFALSE POSITIVES on {len(negs)} light-free frames: '
              f'{len(fp)} ({len(fp)/len(negs):.1%})')
        for gt_pred, c in Counter(x['pred'] for x in fp).most_common():
            print(f'    reported {gt_pred:<7s} {c:4d}')
        if any(x['pred'] == 'red' for x in fp):
            n_red = sum(x['pred'] == 'red' for x in fp)
            print(f'    ^ {n_red} phantom REDs would each force a spurious STOP')
    # Labels that disagree with the map's own view of the scene are worth a look:
    # either the map is missing a fixture or the frame was mislabelled.
    map_miss = [x for x in negs if x['light'] != 'none']
    if map_miss:
        print(f'\n{len(map_miss)} frames marked "no fixture in view" were labelled '
              f'with a light — map likely incomplete there:')
        for x in map_miss[:10]:
            print(f'    {x["scene"]} f{x["frame"]} -> {x["light"]}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'n_labelled': len(rows), 'n_scored': n, 'accuracy': acc,
        'excluded_unknown': skipped_unknown, 'excluded_not_governing': skipped_cross,
        'by_range': {f'{lo}-{hi}': {'acc': sum(v) / len(v), 'n': len(v)}
                     for (lo, hi), v in by_bucket.items() if v},
        'negatives': {
            'n': sum(x['negative'] for x in results),
            'false_positives': sum(x['negative'] and x['pred'] != 'none'
                                   for x in results),
            'phantom_red': sum(x['negative'] and x['pred'] == 'red'
                               for x in results),
        },
        'confusion': {f'{gt}->{pred}': c for (gt, pred), c in confusion.items()},
        'red_recall': (red_total - red_missed) / red_total if red_total else None,
        'e2e': e2e_summary,
        'e2e_frames': e2e_frames,
        'frames': results,
    }, indent=1))
    print(f'\n[INFO] saved -> {out}')


if __name__ == '__main__':
    main()
