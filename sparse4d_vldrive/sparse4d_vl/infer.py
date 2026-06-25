"""
Sparse4D inference on nuScenes mini — pure PyTorch, MPS-compatible.

Usage
-----
  # v2 (recommended — temporal tracking)
  conda run -n simple_bev_vldrive python sparse4d_vl/infer.py \
        --version v2 \
        --dataroot /Users/trish/Downloads/nuScenes_miniV1.0 \
        --scene 0

  # v1 (single-frame)
  conda run -n simple_bev_vldrive python sparse4d_vl/infer.py \
        --version v1 \
        --dataroot /Users/trish/Downloads/nuScenes_miniV1.0

Output per frame
  Prints frame summary: N predicted boxes with scores.
  Saves a simple bird's-eye-view visualisation to sparse4d_outputs/.

No checkpoint is loaded — this runs with randomly initialised weights to
verify the architecture (shapes, devices, operations) end-to-end.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sparse4d_vl.data.loader import NuScenesSparse4DLoader
from sparse4d_vl.model.sparse4d_v2 import Sparse4Dv1, Sparse4Dv2
from sparse4d_vl.model.sparse4d_v3 import Sparse4Dv3
from sparse4d_vl.model.detection3d import CLASS_NAMES
from sparse4d_vl.model.checkpoint import load_checkpoint
    
# ---------------------------------------------------------------------------
# Simple BEV visualiser
# ---------------------------------------------------------------------------

def _draw_bev(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
              frame_idx: int, out_dir: Path, pc_range: float = 50.0,
              canvas: int = 500):
    """Draw bird's-eye-view box centres on a 500×500 canvas."""
    try:
        import cv2
    except ImportError:
        return   # skip if cv2 unavailable

    img = np.zeros((canvas, canvas, 3), dtype=np.uint8)

    for box, sc, lb in zip(boxes.cpu().numpy(),
                            scores.cpu().numpy(),
                            labels.cpu().numpy()):
        x, y = box[0], box[1]
        # Map metres to pixels: centre = canvas//2, scale = canvas / (2*pc_range)
        px = int(canvas // 2 + x / (2 * pc_range) * canvas)
        py = int(canvas // 2 - y / (2 * pc_range) * canvas)
        if 0 <= px < canvas and 0 <= py < canvas:
            color = _class_color(int(lb))
            cv2.circle(img, (px, py), 4, color, -1)
            cv2.putText(img, f'{sc:.2f}', (px + 5, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    out_path = out_dir / f'bev_{frame_idx:04d}.png'
    cv2.imwrite(str(out_path), img)


def _class_color(label: int) -> tuple[int, int, int]:
    palette = [
        (255, 80, 80),   # car        — red
        (255, 160, 80),  # truck      — orange
        (255, 240, 80),  # constr.    — yellow
        (80, 255, 80),   # bus        — green
        (80, 255, 240),  # trailer    — cyan
        (80, 160, 255),  # barrier    — light blue
        (160, 80, 255),  # motorcycle — violet
        (255, 80, 255),  # bicycle    — magenta
        (255, 255, 255), # pedestrian — white
        (160, 160, 160), # cone       — grey
    ]
    return palette[label % len(palette)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--version',   default='v2', choices=['v1', 'v2', 'v3'],
                   help='Sparse4D version')
    p.add_argument('--checkpoint', default='model/checkpoints/sparse4dv2_r50_HInf_256x704.pth')
    p.add_argument('--dataroot',  default='/Users/trish/Downloads/nuScenes_miniV1.0')
    p.add_argument('--scene',     type=int, default=0, help='Scene index (0-9)')
    p.add_argument('--max-frames',type=int, default=None,
                   help='Stop after this many frames (default: full scene)')
    p.add_argument('--out-dir',   default='sparse4d_outputs')
    p.add_argument('--planning', action='store_true',
                   help='(v3) build SparseDrive motion planner; print motion + ego plan')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f'[Sparse4D] version={args.version}  scene={args.scene}')

    # ---- Data loader ----
    loader = NuScenesSparse4DLoader(args.dataroot)

    # ---- Model ----
    if args.version == 'v1':
        model = Sparse4Dv1(pretrained_backbone=False)
    elif args.version == 'v3':
        model = Sparse4Dv3(pretrained_backbone=False, with_planning=args.planning)
    else:
        model = Sparse4Dv2(pretrained_backbone=False)

    model.eval()
    if args.checkpoint and Path(args.checkpoint).exists():
        load_checkpoint(model, args.checkpoint, version=args.version)
    else:
        print('[ckpt] no checkpoint found — running with random weights')
    device = model.device
    print(f'[Sparse4D] device={device}  parameters={sum(p.numel() for p in model.parameters()):,}')

    # Reset temporal cache for the chosen scene
    if hasattr(model, 'reset_state'):
        model.reset_state()

    # ---- Inference loop ----
    with torch.no_grad():
        for frame_idx, batch in enumerate(loader.iter_scene(args.scene)):
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            imgs      = batch['imgs']        # (1, 6, 3, H, W)
            img_metas = batch['img_metas']

            t0 = time.perf_counter()
            out = model(imgs, img_metas)
            t1 = time.perf_counter()

            dets    = out['detections'][0]   # dict for batch element 0
            boxes   = dets['boxes_3d']       # (K, 9)
            scores  = dets['scores_3d']      # (K,)
            labels  = dets['labels_3d']      # (K,)

            n_det = boxes.shape[0]
            print(
                f'  frame {frame_idx:3d}  '
                f'detections={n_det:3d}  '
                f'top_score={scores.max().item():.3f}  '
                f'{(t1-t0)*1000:.1f} ms'
            )

            # Per-class counts
            if n_det > 0:
                for cls_id, cls_name in enumerate(CLASS_NAMES):
                    cnt = (labels == cls_id).sum().item()
                    if cnt > 0:
                        print(f'    {cls_name}: {cnt}')

            # BEV visualisation
            if n_det > 0:
                _draw_bev(boxes, scores, labels, frame_idx, out_dir)

            # SparseDrive motion + planning printout
            if args.planning and 'ego_traj' in dets:
                cmd_name = {0: 'right', 1: 'straight', 2: 'left'}
                ego = dets['ego_traj']                     # (3, Te, 2)
                logit = dets['ego_logits']                 # (3,)
                sel = int(logit.argmax())
                end = ego[sel, -1]
                print(f'    [plan] cmd={cmd_name[sel]}  ego endpoint=({end[0]:.1f},{end[1]:.1f}) m'
                      f'  (3 modes)')
                if 'trajectories' in dets and dets['trajectories'].shape[0] > 0:
                    tr = dets['trajectories']              # (K_det, modes, T, 2)
                    print(f'    [motion] {tr.shape[0]} agents × {tr.shape[1]} modes × {tr.shape[2]} steps')

    print(f'\n[Sparse4D] outputs saved to {out_dir}/')


if __name__ == '__main__':
    main()
