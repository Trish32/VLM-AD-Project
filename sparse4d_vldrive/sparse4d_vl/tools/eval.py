#!/usr/bin/env python3
"""
Evaluate Sparse4D v1 / v2 on nuScenes mini.

Reports:
  • nuScenes mAP + NDS  (via devkit NuScenesEval)
  • Per-class AP table

Usage:
    python sparse4d_vl/tools/eval.py \
        --version v2 \
        --checkpoint sparse4d_vl/model/checkpoints/sparse4dv2_r50_HInf_256x704.pth \
        --dataroot /Users/trish/Downloads/nuScenes_miniV1.0 \
        [--eval-set mini_val] [--max-frames N]

Coordinate system:
  Model output: ego/lidar frame  →  converted to global frame for submission to the nuScenes devkit(expects the global frame).
  Sizes: exp() applied (model stores log-space w/l/h).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.utils.splits import create_splits_scenes

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sparse4d_vl.data.loader import NuScenesSparse4DLoader
from sparse4d_vl.model.sparse4d_v2 import Sparse4Dv1, Sparse4Dv2
from sparse4d_vl.model.sparse4d_v3 import Sparse4Dv3
from sparse4d_vl.model.detection3d import CLASS_NAMES
from sparse4d_vl.model.checkpoint import load_checkpoint

# ---------------------------------------------------------------------------
# nuScenes category / attribute helpers
# ---------------------------------------------------------------------------
# The nuScenes metric requires an attribute string (e.g. vehicle.parked) per box; 
# the model doesn't predict attributes, so a per-class default is used. 
# Attributes barely affect mAP but are required for a valid submission.
ATTR_DEFAULT = {
    'car':                  'vehicle.parked',
    'truck':                'vehicle.parked',
    'construction_vehicle': 'vehicle.parked',
    'bus':                  'vehicle.parked',
    'trailer':              'vehicle.parked',
    'motorcycle':           'cycle.without_rider',
    'bicycle':              'cycle.without_rider',
    'pedestrian':           'pedestrian.standing',
    'barrier':              '',
    'traffic_cone':         '',
}


# ---------------------------------------------------------------------------
# Coordinate conversion: ego frame → global frame
# ---------------------------------------------------------------------------

def _boxes_lidar_to_global(
    boxes:      np.ndarray,   # (K, 9)  [x,y,z, w,l,h, yaw, vx,vy]  in lidar sensor frame
    ego2global: np.ndarray,   # (4, 4)  ego → global
    lidar2ego:  np.ndarray,   # (4, 4)  lidar sensor → ego
) -> list[dict]:
    """Convert lidar-frame boxes to nuScenes submission format (global frame).

    Prediction positions are in the LIDAR SENSOR frame (the frame the model was
    trained in, matching the checkpoint's K-means anchors and lidar2img projection).
    Chain: lidar_sensor → ego → global.
    """
    lidar2global = ego2global @ lidar2ego   # # forward chain (4, 4)
    R_l2g = lidar2global[:3, :3]           # (3, 3)
    t_l2g = lidar2global[:3, 3]            # (3,)

    # Lidar→global heading angle (yaw contribution from lidar2global rotation around Z)
    lidar_yaw_in_global = math.atan2(float(R_l2g[1, 0]), float(R_l2g[0, 0]))

    dets = []
    for box in boxes:
        x, y, z        = box[0], box[1], box[2]
        # Model anchor dims are [length, width, height] (slot 3 = extent along
        # heading).  nuScenes submission wants size = [width, length, height],
        # so swap the first two (reference: nus_box_dims = dims[..., [1,0,2]]).
        l, w, h        = box[3], box[4], box[5]  # The decoder emitted [length, width, height] (slot-3 = heading extent)
        yaw_lidar      = box[6]
        vx_lidar, vy_lidar = box[7], box[8]

        # Position: lidar frame → global
        pos_lidar  = np.array([x, y, z])
        # GT loader applied R.T @ (x - t) (global→lidar), 
        # eval applies R @ x + t (lidar→global) — predictions go the opposite direction
        pos_global = R_l2g @ pos_lidar + t_l2g  # lidar → global (forward!)

        # Yaw: add lidar→global heading offset
        yaw_global = yaw_lidar + lidar_yaw_in_global

        q = Quaternion(axis=[0, 0, 1], angle=float(yaw_global))

        # Velocity: rotate from lidar frame to global
        v_lidar  = np.array([vx_lidar, vy_lidar])
        v_global = R_l2g[:2, :2] @ v_lidar

        dets.append({
            'translation': pos_global.tolist(),
            'size':        [float(w), float(l), float(h)],  # But nuScenes submission wants size=[width, length, height]
            'rotation':    [q.w, q.x, q.y, q.z],
            'velocity':    v_global.tolist(),
        })

    return dets


# ---------------------------------------------------------------------------
# Inference over all scenes
# ---------------------------------------------------------------------------

def run_inference(
    model:      Sparse4Dv1 | Sparse4Dv2,
    loader:     NuScenesSparse4DLoader,
    version:    str,
    max_frames: int | None,
) -> dict:
    """
    Run model over all nuScenes mini scenes and collect detections per sample.

    Returns
    -------
    all_results : {sample_token: [det_dict, ...]}
    """
    all_results: dict[str, list] = {}
    total_frames = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        for scene_idx in range(len(loader.nusc.scene)):
            # Resetting state per scen
            if hasattr(model, 'reset_state'):
                model.reset_state()

            for frame in loader.iter_scene(scene_idx):
                if max_frames is not None and total_frames >= max_frames:
                    break

                imgs      = frame['imgs'].float()
                img_metas = frame['img_metas']
                token     = img_metas['sample_token']
                ego2g      = img_metas['ego2global']         # (4, 4) numpy
                lidar2ego  = img_metas.get('lidar2ego', np.eye(4, dtype=np.float32))

                out  = model(imgs, img_metas)                # Run the model
                dets = out['detections'][0]                  # batch element 0

                boxes_3d  = dets['boxes_3d'].cpu().numpy()   # (K, 9) metric
                scores_3d = dets['scores_3d'].cpu().numpy()  # (K,)
                labels_3d = dets['labels_3d'].cpu().numpy()  # (K,) int

                sample_dets = []
                if boxes_3d.shape[0] > 0:
                    # Convert each box to global fram
                    global_dets = _boxes_lidar_to_global(boxes_3d, ego2g, lidar2ego)
                    for det, score, label in zip(global_dets, scores_3d, labels_3d):
                        cls_name = CLASS_NAMES[int(label)]
                        det.update({
                            'sample_token':      token,
                            'detection_name':    cls_name,
                            'detection_score':   float(score),
                            'attribute_name':    ATTR_DEFAULT[cls_name],
                        })
                        sample_dets.append(det)

                all_results[token] = sample_dets
                total_frames += 1

            if max_frames is not None and total_frames >= max_frames:
                break

    elapsed = time.perf_counter() - t_start
    fps = total_frames / elapsed if elapsed > 0 else 0
    print(f'  Inference: {total_frames} frames in {elapsed:.1f}s  ({fps:.1f} fps)')
    return all_results


# ---------------------------------------------------------------------------
# nuScenes evaluation
# ---------------------------------------------------------------------------

def _split_tokens(nusc: NuScenes, split_name: str) -> set:
    """Return all sample tokens belonging to a nuScenes split."""
    scene_names = set(create_splits_scenes()[split_name])
    tokens = set()
    for scene in nusc.scene:
        if scene['name'] not in scene_names:
            continue
        tok = scene['first_sample_token']
        while tok:
            tokens.add(tok)
            tok = nusc.get('sample', tok)['next']
    return tokens


def _run_nusc_eval(
    nusc:       NuScenes,
    results:    dict,
    split:      str,
    output_dir: str,
) -> tuple:
    """Write a submission JSON for one split and run NuScenesEval."""
    split_toks = _split_tokens(nusc, split)
    subset = {k: results.get(k, []) for k in split_toks}

    submission = {
        'meta': {
            'use_camera': True, 'use_lidar': False,
            'use_radar': False, 'use_map': False, 'use_external': False,
        },
        'results': subset,
    }
    result_path = os.path.join(output_dir, f'results_{split}.json')
    with open(result_path, 'w') as f:
        json.dump(submission, f)

    try:
        nusc_eval = NuScenesEval(
            nusc,
            config=config_factory('detection_cvpr_2019'),
            result_path=result_path,
            eval_set=split,
            output_dir=output_dir,
            verbose=False,
        )
        metrics = nusc_eval.main(plot_examples=0, render_curves=False)
    except Exception as e:
        # Devkit raises if all predictions are empty (e.g. --max-frames test run)
        print(f'    [{split}] NuScenesEval error (all predictions empty?): {e}')
        metrics = {
            'mean_ap': 0.0, 'nd_score': 0.0,
            'mean_dist_aps': {c: 0.0 for c in CLASS_NAMES},
        }  # Empty-prediction is caught and returns zeros instead of crashing
    return metrics, len(split_toks)


def evaluate(
    nusc:       NuScenes,
    results:    dict,
    eval_set:   str,
    output_dir: str,
) -> tuple:
    """
    Run NuScenesEval.

    eval_set='mini_all' evaluates both mini_train (8 scenes) and mini_val
    (2 scenes) covering all 10 mini scenes, then returns sample-count-weighted
    combined metrics.  Any other eval_set (e.g. 'mini_val') is passed through
    directly.
    """
    if eval_set == 'mini_all':
        splits = ['mini_train', 'mini_val']
        all_metrics, all_n = [], []
        for split in splits:
            m, n = _run_nusc_eval(nusc, results, split, output_dir)
            all_metrics.append(m)
            all_n.append(n)
            print(f'    [{split}]  mAP={m["mean_ap"]:.4f}  NDS={m["nd_score"]:.4f}'
                  f'  ({n} samples)')

        total = sum(all_n)
        combined = {
            'mean_ap':  sum(m['mean_ap']  * n for m, n in zip(all_metrics, all_n)) / total,
            'nd_score': sum(m['nd_score'] * n for m, n in zip(all_metrics, all_n)) / total,
            'mean_dist_aps': {},
        }
        for cls in CLASS_NAMES:
            combined['mean_dist_aps'][cls] = sum(
                m['mean_dist_aps'].get(cls, 0.0) * n
                for m, n in zip(all_metrics, all_n)
            ) / total
        return combined
    else:
        m, _ = _run_nusc_eval(nusc, results, eval_set, output_dir)
        return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Sparse4D nuScenes evaluation')
    p.add_argument('--version',    default='v2', choices=['v1', 'v2', 'v3'])
    p.add_argument('--checkpoint', default='model/checkpoints/sparse4dv2_r50_HInf_256x704.pth',
                   help='Path to .pth checkpoint (skip loading if not given)')
    p.add_argument('--dataroot',   default='/Users/trish/Downloads/nuScenes_miniV1.0')
    p.add_argument('--eval-set',   default='mini_all',
                   help='mini_all (default, all 10 scenes), mini_val, or mini_train')
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--out-dir',    default='sparse4d_eval_outputs')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[Sparse4D eval] version={args.version}  eval_set={args.eval_set}'
          + ('  (mini_train + mini_val combined)' if args.eval_set == 'mini_all' else ''))

    # ---- Build loader ----
    loader = NuScenesSparse4DLoader(args.dataroot)

    # ---- Build model ----
    if args.version == 'v1':
        model = Sparse4Dv1(pretrained_backbone=False)
    elif args.version == 'v3':
        model = Sparse4Dv3(pretrained_backbone=False)
    else:
        model = Sparse4Dv2(pretrained_backbone=False)

    model.eval()

    if args.checkpoint and Path(args.checkpoint).exists():
        load_checkpoint(model, args.checkpoint, version=args.version)
    else:
        print('[ckpt] no checkpoint — evaluating with random weights (sanity check)')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'       device={model.device}  params={n_params:,}')

    # ---- Inference ----
    print('\n[1/2] Running inference...')
    all_results = run_inference(model, loader, args.version, args.max_frames)
    print(f'       samples with detections: '
          f'{sum(1 for v in all_results.values() if v)}/{len(all_results)}')

    # ---- Evaluate ----
    print('\n[2/2] Running NuScenesEval...')
    with tempfile.TemporaryDirectory() as tmp:
        metrics = evaluate(
            loader.nusc, all_results, args.eval_set, tmp
        )

    # ---- Print results ----
    mAP = metrics['mean_ap']
    NDS = metrics['nd_score']
    print(f'\n{"="*50}')
    print(f'  mAP : {mAP:.4f}')
    print(f'  NDS : {NDS:.4f}')
    print(f'{"="*50}')

    # Per-class AP table
    print(f'\n{"Class":<25} {"AP":>6}')
    print('-' * 33)
    for cls_name in CLASS_NAMES:
        ap = metrics['mean_dist_aps'].get(cls_name, float('nan'))
        print(f'  {cls_name:<23} {ap:6.4f}')

    # Save summary JSON
    summary = {'mAP': mAP, 'NDS': NDS, 'per_class_AP': metrics.get('mean_dist_aps', {})}
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary saved to {out_dir}/summary.json')


if __name__ == '__main__':
    main()
