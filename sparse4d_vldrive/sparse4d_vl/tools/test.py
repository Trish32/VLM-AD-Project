#!/usr/bin/env python3
"""
Run Sparse4D inference on all 10 nuScenes mini scenes and report stats.

Unlike eval.py (which requires checkpoint weights and runs NuScenesEval),
test.py is a quick validation run — useful with random or real weights to
check detection counts, scores, and per-class distributions across all scenes.

Usage:
    python sparse4d_vl/tools/test.py --version v2
    python sparse4d_vl/tools/test.py --version v2 \\
        --checkpoint sparse4d_vl/model/checkpoints/sparse4dv2_r50_HInf_256x704.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sparse4d_vl.data.loader import NuScenesSparse4DLoader
from sparse4d_vl.model.sparse4d_v2 import Sparse4Dv1, Sparse4Dv2
from sparse4d_vl.model.detection3d import CLASS_NAMES
from sparse4d_vl.model.checkpoint import load_checkpoint


def run_all_scenes(
    model:      Sparse4Dv1 | Sparse4Dv2,
    loader:     NuScenesSparse4DLoader,
    score_thresh: float,
) -> list[dict]:
    """
    Iterate all 10 mini scenes, collect per-frame stats.

    Returns list of per-scene dicts with frame-level detection info.
    """
    all_scenes = []
    total_frames = 0
    total_dets   = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        for scene_idx, scene_meta in enumerate(loader.nusc.scene):
            scene_name  = scene_meta['name']
            scene_stats = {'scene': scene_name, 'frames': []}
            class_counts = defaultdict(int)

            if hasattr(model, 'reset_state'):
                model.reset_state()

            for frame_idx, frame in enumerate(loader.iter_scene(scene_idx)):
                t0  = time.perf_counter()
                out = model(frame['imgs'].float(), frame['img_metas'])
                ms  = (time.perf_counter() - t0) * 1000

                dets   = out['detections'][0]
                boxes  = dets['boxes_3d']      # (K, 9)
                scores = dets['scores_3d']     # (K,)
                labels = dets['labels_3d']     # (K,)

                keep    = scores >= score_thresh
                n_keep  = int(keep.sum())
                top     = float(scores.max()) if scores.numel() > 0 else 0.0

                for lb in labels[keep].cpu().tolist():
                    class_counts[CLASS_NAMES[lb]] += 1

                scene_stats['frames'].append({
                    'frame':     frame_idx,
                    'dets_raw':  int(boxes.shape[0]),
                    'dets_kept': n_keep,
                    'top_score': round(top, 4),
                    'ms':        round(ms, 1),
                })
                total_dets   += n_keep
                total_frames += 1

            scene_stats['total_dets']   = sum(class_counts.values())
            scene_stats['class_counts'] = dict(class_counts)
            all_scenes.append(scene_stats)

            n_fr = len(scene_stats['frames'])
            avg_ms = sum(f['ms'] for f in scene_stats['frames']) / max(n_fr, 1)
            print(f'  scene {scene_idx:2d} [{scene_name}]  '
                  f'frames={n_fr}  dets={scene_stats["total_dets"]}  '
                  f'avg={avg_ms:.0f}ms/frame')

    elapsed = time.perf_counter() - t_start
    fps = total_frames / elapsed if elapsed > 0 else 0
    print(f'\n  Total: {total_frames} frames  {total_dets} detections'
          f'  {elapsed:.1f}s  ({fps:.1f} fps)')

    return all_scenes


def print_summary(scenes: list[dict], score_thresh: float):
    total_cls = defaultdict(int)
    total_dets = 0
    for sc in scenes:
        for cls, cnt in sc['class_counts'].items():
            total_cls[cls] += cnt
            total_dets += cnt

    print(f'\n{"="*52}')
    print(f'  10 scenes  |  score threshold: {score_thresh}')
    print(f'{"="*52}')
    print(f'  {"Class":<25} {"Detections":>10}')
    print(f'  {"-"*37}')
    for cls in CLASS_NAMES:
        cnt = total_cls.get(cls, 0)
        print(f'  {cls:<25} {cnt:>10}')
    print(f'  {"-"*37}')
    print(f'  {"TOTAL":<25} {total_dets:>10}')
    print(f'{"="*52}')


def parse_args():
    p = argparse.ArgumentParser(description='Sparse4D test — all 10 mini scenes')
    p.add_argument('--version',      default='v2', choices=['v1', 'v2'])
    p.add_argument('--checkpoint',   default='model/checkpoints/sparse4dv2_r50_HInf_256x704.pth')
    p.add_argument('--dataroot',     default='/Users/trish/Downloads/nuScenes_miniV1.0')
    p.add_argument('--score-thresh', type=float, default=0.2)
    p.add_argument('--out-dir',      default='sparse4d_test_outputs')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = NuScenesSparse4DLoader(args.dataroot)

    if args.version == 'v1':
        model = Sparse4Dv1(pretrained_backbone=False)
    else:
        model = Sparse4Dv2(pretrained_backbone=False)
    model.eval()

    if args.checkpoint and Path(args.checkpoint).exists():
        load_checkpoint(model, args.checkpoint, version=args.version)
    else:
        print('[ckpt] no checkpoint — running with random weights')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'[Sparse4D test]  version={args.version}  device={model.device}'
          f'  params={n_params:,}\n')

    scenes = run_all_scenes(model, loader, args.score_thresh)
    print_summary(scenes, args.score_thresh)

    out = out_dir / 'test_results.json'
    with open(out, 'w') as f:
        json.dump(scenes, f, indent=2)
    print(f'\nPer-frame stats saved to {out}')


if __name__ == '__main__':
    main()
