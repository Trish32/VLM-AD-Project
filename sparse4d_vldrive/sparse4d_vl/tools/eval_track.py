#!/usr/bin/env python3
"""
Evaluate Sparse4D-v3 detection-AND-tracking on nuScenes mini (AMOTA / AMOTP).

The model assigns persistent track IDs via the instance bank (ID propagation on
the temporal cache — get_instance_id), surfaced as `track_ids` in the decoder
output.  This script writes a nuScenes *tracking* submission and runs the
official TrackingEval (AMOTA, AMOTP, MOTA, ...).

Only the 7 nuScenes tracking classes are submitted (bicycle, bus, car,
motorcycle, pedestrian, trailer, truck); barrier / traffic_cone /
construction_vehicle are detection-only and excluded.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python sparse4d_vl/tools/eval_track.py \
        --checkpoint sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth \
        --eval-set mini_val
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory
from nuscenes.utils.splits import create_splits_scenes

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sparse4d_vl.data.loader import NuScenesSparse4DLoader
from sparse4d_vl.model.sparse4d_v3 import Sparse4Dv3
from sparse4d_vl.model.detection3d import CLASS_NAMES
from sparse4d_vl.model.checkpoint import load_checkpoint
from sparse4d_vl.tools.eval import _boxes_lidar_to_global, _split_tokens

# nuScenes tracking challenge classes (subset of the 10 detection classes)
TRACKING_NAMES = {
    'bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck',
}


def run_inference(model, loader, max_frames):
    """Run the model scene-by-scene, collecting tracking detections per sample."""
    all_results: dict[str, list] = {}
    total = 0
    with torch.no_grad():
        for scene_idx in range(len(loader.nusc.scene)):
            model.reset_state()                      # new scene → reset cache + IDs
            for frame in loader.iter_scene(scene_idx):
                if max_frames is not None and total >= max_frames:
                    break
                metas = frame['img_metas']
                token = metas['sample_token']
                ego2g     = metas['ego2global']
                lidar2ego = metas.get('lidar2ego', np.eye(4, dtype=np.float32))

                out = model(frame['imgs'].float(), metas)
                dets = out['detections'][0]
                boxes  = dets['boxes_3d'].cpu().numpy()
                scores = dets['scores_3d'].cpu().numpy()
                labels = dets['labels_3d'].cpu().numpy()
                tids   = dets.get('track_ids')
                tids   = tids.cpu().numpy() if tids is not None else None

                sample_dets = []
                if boxes.shape[0] > 0:
                    glob = _boxes_lidar_to_global(boxes, ego2g, lidar2ego)
                    for i, (d, sc, lb) in enumerate(zip(glob, scores, labels)):
                        name = CLASS_NAMES[int(lb)]
                        if name not in TRACKING_NAMES:
                            continue
                        tid = int(tids[i]) if tids is not None else -1
                        if tid < 0:
                            continue                 # no track identity → skip
                        d.update({
                            'sample_token':   token,
                            'tracking_name':  name,
                            'tracking_score': float(sc),
                            # globally-unique string id (scene-prefixed)
                            'tracking_id':    f'{scene_idx}_{tid}',
                        })
                        sample_dets.append(d)
                all_results[token] = sample_dets
                total += 1
            if max_frames is not None and total >= max_frames:
                break
    print(f'  inference: {total} frames')
    return all_results


def main():
    p = argparse.ArgumentParser(description='Sparse4D-v3 tracking (AMOTA) eval')
    p.add_argument('--checkpoint', default='sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth')
    p.add_argument('--dataroot',   default='/Users/trish/Downloads/nuScenes_miniV1.0')
    p.add_argument('--version',    default='v1.0-mini')
    p.add_argument('--eval-set',   default='mini_val')
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--out-dir',    default='sparse4d_track_outputs')
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[Sparse4D track] eval_set={args.eval_set}')

    loader = NuScenesSparse4DLoader(args.dataroot, version=args.version)
    model = Sparse4Dv3(pretrained_backbone=False)
    model.eval()
    if args.checkpoint and Path(args.checkpoint).exists():
        load_checkpoint(model, args.checkpoint, version='v3')
    else:
        print('[ckpt] no checkpoint — random weights')

    print('\n[1/2] inference (with track-ID propagation)...')
    results = run_inference(model, loader, args.max_frames)
    n_tracks = len({d['tracking_id'] for v in results.values() for d in v})
    print(f'       unique tracks: {n_tracks}')

    # Filter to the requested split
    split_toks = _split_tokens(loader.nusc, args.eval_set)
    subset = {k: results.get(k, []) for k in split_toks}
    submission = {
        'meta': {'use_camera': True, 'use_lidar': False, 'use_radar': False,
                 'use_map': False, 'use_external': False},
        'results': subset,
    }

    print('\n[2/2] TrackingEval (AMOTA)...')
    with tempfile.TemporaryDirectory() as tmp:
        res_path = os.path.join(tmp, 'tracking_result.json')
        with open(res_path, 'w') as f:
            json.dump(submission, f)
        cfg = config_factory('tracking_nips_2019')
        ev = TrackingEval(config=cfg, result_path=res_path, eval_set=args.eval_set,
                          output_dir=tmp, nusc_version=args.version,
                          nusc_dataroot=args.dataroot, verbose=False)
        metrics = ev.main(render_curves=False)

    amota = metrics['amota']; amotp = metrics['amotp']
    print(f'\n{"="*50}')
    print(f'  AMOTA : {amota:.4f}')
    print(f'  AMOTP : {amotp:.4f}')
    print(f'  MOTA  : {metrics.get("mota", float("nan")):.4f}')
    print(f'  recall: {metrics.get("recall", float("nan")):.4f}')
    print(f'{"="*50}')

    # Per-class AMOTA
    lm = metrics.get('label_metrics', {}).get('amota', {})
    if lm:
        print(f'\n{"Class":<16}{"AMOTA":>8}')
        print('-' * 24)
        for c in sorted(lm):
            v = lm[c]
            print(f'  {c:<14}{(v if v is not None else float("nan")):>8.4f}')

    with open(out_dir / 'track_summary.json', 'w') as f:
        json.dump({'amota': amota, 'amotp': amotp,
                   'mota': metrics.get('mota'), 'recall': metrics.get('recall')}, f, indent=2)
    print(f'\nSaved → {out_dir}/track_summary.json')


if __name__ == '__main__':
    main()
