#!/usr/bin/env python3
"""
Visualize Sparse4D 3D predictions projected onto the 6 nuScenes cameras.

Usage:
    python sparse4d_vl/tools/visualizer.py \
        --version v2 \
        --checkpoint sparse4d_vl/model/checkpoints/sparse4dv2_r50_HInf_256x704.pth \
        --dataroot /Users/trish/Downloads/nuScenes_miniV1.0 \
        --scene 0 [--max-frames 5] [--out-dir sparse4d_viz]

Output:
  For each frame: a 3×2 grid PNG (6 cameras) with projected 3D boxes drawn.
  Box colour encodes object class.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sparse4d_vl.data.loader import NuScenesSparse4DLoader, CAM_NAMES, IMG_H, IMG_W
from sparse4d_vl.model.sparse4d_v2 import Sparse4Dv1, Sparse4Dv2
from sparse4d_vl.model.detection3d import CLASS_NAMES
from sparse4d_vl.model.checkpoint import load_checkpoint

# ---------------------------------------------------------------------------
# Colour palette — one colour per class (BGR for OpenCV)
# ---------------------------------------------------------------------------

_PALETTE_BGR = [
    (80,  80, 255),   # car           — red
    (80, 160, 255),   # truck         — orange
    (80, 240, 255),   # constr.       — yellow
    (80, 255, 80),    # bus           — green
    (240, 255, 80),   # trailer       — cyan
    (255, 160, 80),   # barrier       — light blue
    (255, 80, 160),   # motorcycle    — violet
    (255, 80, 255),   # bicycle       — magenta
    (255, 255, 255),  # pedestrian    — white
    (160, 160, 160),  # traffic_cone  — grey
]


def _class_color(label: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[label % len(_PALETTE_BGR)]


# ---------------------------------------------------------------------------
# 3D box corners in ego/lidar frame
# ---------------------------------------------------------------------------

def _box_corners_ego(box: np.ndarray) -> np.ndarray:
    """
    box : (9,)  [x, y, z, w, l, h, yaw, vx, vy]  metric, ego frame
    Returns: (8, 3) 3D corners in ego frame.
    """
    x, y, z = box[0], box[1], box[2]
    w, l, h = box[3], box[4], box[5]    # width, length, height
    yaw      = box[6]

    # Half-extents
    hw, hl, hh = w / 2, l / 2, h / 2

    # 8 corners: (±l, ±w, ±h) in object frame (before yaw rotation)
    # Convention: x-forward (length), y-left (width), z-up (height)
    corners_obj = np.array([
        [-hl, -hw, -hh], [-hl, -hw,  hh],
        [-hl,  hw, -hh], [-hl,  hw,  hh],
        [ hl, -hw, -hh], [ hl, -hw,  hh],
        [ hl,  hw, -hh], [ hl,  hw,  hh],
    ], dtype=np.float64)   # (8, 3)

    # Rotate by yaw around Z
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    corners_ego = corners_obj @ R.T + np.array([x, y, z])   # (8, 3)
    return corners_ego


# ---------------------------------------------------------------------------
# Project 3D corners onto a camera image
# ---------------------------------------------------------------------------

def _project_corners(
    corners: np.ndarray,      # (8, 3) 3D points in ego frame
    proj_mat: np.ndarray,     # (4, 4) lidar→pixel
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    pts_2d : (8, 2)  pixel coordinates (may be outside image bounds)
    depth  : (8,)    depth values (positive = in front of camera)
    """
    ones = np.ones((8, 1), dtype=np.float64)
    pts_h = np.hstack([corners, ones])       # (8, 4)
    # Project via (3, 4) sub-matrix
    proj_3 = (proj_mat[:3] @ pts_h.T).T      # (8, 3)
    depth  = proj_3[:, 2]
    eps    = 1e-5
    safe   = np.where(np.abs(depth) > eps, depth, eps * np.sign(depth + eps))
    pts_2d = proj_3[:, :2] / safe[:, None]   # (8, 2)
    return pts_2d, depth


# ---------------------------------------------------------------------------
# Draw one 3D box on a camera image
# ---------------------------------------------------------------------------

# Edge indices for a box (connect all 12 edges)
_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 3),   # rear face
    (4, 5), (4, 6), (5, 7), (6, 7),   # front face
    (0, 4), (1, 5), (2, 6), (3, 7),   # lateral edges
]


def _draw_box_on_image(
    img:     np.ndarray,         # (H, W, 3) uint8 BGR
    box:     np.ndarray,         # (9,) metric ego-frame box
    proj:    np.ndarray,         # (4, 4) projection
    color:   tuple[int,int,int],
    label:   str,
    score:   float,
    min_depth: float = 0.1,
) -> np.ndarray:
    """Draw projected 3D box edges and label. Returns modified image."""
    H, W = img.shape[:2]

    corners = _box_corners_ego(box)          # (8, 3)
    pts, depth = _project_corners(corners, proj)

    # Skip if all corners are behind camera
    if (depth <= min_depth).all():
        return img

    img = img.copy()

    for a, b in _EDGES:
        if depth[a] <= min_depth or depth[b] <= min_depth:
            continue
        xa, ya = int(pts[a, 0] + 0.5), int(pts[a, 1] + 0.5)
        xb, yb = int(pts[b, 0] + 0.5), int(pts[b, 1] + 0.5)
        # Only draw if at least one endpoint is roughly in-image (loose clamp)
        if max(xa, xb) < -W or min(xa, xb) > 2 * W:
            continue
        if max(ya, yb) < -H or min(ya, yb) > 2 * H:
            continue
        xa = int(np.clip(xa, -W, 2 * W))
        ya = int(np.clip(ya, -H, 2 * H))
        xb = int(np.clip(xb, -W, 2 * W))
        yb = int(np.clip(yb, -H, 2 * H))
        cv2.line(img, (xa, ya), (xb, yb), color, 2, lineType=cv2.LINE_AA)

    # Draw label at the centroid of visible corners
    vis = pts[depth > min_depth]
    if len(vis) > 0:
        cx, cy = int(vis[:, 0].mean()), int(vis[:, 1].mean())
        if 0 <= cx < W and 0 <= cy < H:
            text = f'{label[:3]} {score:.2f}'
            cv2.putText(img, text, (cx, max(cy - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Visualise one frame: 6-camera grid
# ---------------------------------------------------------------------------

def visualise_frame(
    imgs_np:      np.ndarray,         # (N_cam, 3, H, W)  float32 [0,255] RGB
    projection:   np.ndarray,         # (N_cam, 4, 4)
    boxes_3d:     np.ndarray,         # (K, 9)  metric, ego-frame
    scores:       np.ndarray,         # (K,)
    labels:       np.ndarray,         # (K,)  int
    frame_idx:    int,
    out_dir:      Path,
    score_thresh: float = 0.2,
):
    """Render 6-camera grid with projected boxes, save to out_dir."""
    N_cam = imgs_np.shape[0]

    cam_imgs = []
    for cam_idx in range(N_cam):
        # (3, H, W) float [0,255] RGB → (H, W, 3) uint8 BGR
        img = imgs_np[cam_idx].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        proj = projection[cam_idx]   # (4, 4)

        for k in range(boxes_3d.shape[0]):
            if scores[k] < score_thresh:
                continue
            label_id = int(labels[k])
            img = _draw_box_on_image(
                img, boxes_3d[k], proj,
                color=_class_color(label_id),
                label=CLASS_NAMES[label_id],
                score=float(scores[k]),
            )

        # Camera name overlay
        cam_name = CAM_NAMES[cam_idx].replace('CAM_', '')
        cv2.putText(img, cam_name, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cam_imgs.append(img)

    # Arrange into 3-column × 2-row grid:
    # top row: FRONT_LEFT, FRONT, FRONT_RIGHT
    # bot row: BACK_LEFT,  BACK,  BACK_RIGHT
    # CAM order: FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT
    grid_order = [2, 0, 1, 4, 3, 5]   # top-left to bottom-right
    row0 = np.hstack([cam_imgs[i] for i in grid_order[:3]])
    row1 = np.hstack([cam_imgs[i] for i in grid_order[3:]])
    grid = np.vstack([row0, row1])

    out_path = out_dir / f'frame_{frame_idx:04d}.jpg'
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Sparse4D 3D-box visualizer')
    p.add_argument('--version',    default='v2', choices=['v1', 'v2'])
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--dataroot',   default='/Users/trish/Downloads/nuScenes_miniV1.0')
    p.add_argument('--scene',      type=int, default=0)
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--score-thresh', type=float, default=0.2)
    p.add_argument('--out-dir',    default='sparse4d_viz')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[Sparse4D viz] version={args.version}  scene={args.scene}')

    loader = NuScenesSparse4DLoader(args.dataroot)

    if args.version == 'v1':
        model = Sparse4Dv1(pretrained_backbone=False)
    else:
        model = Sparse4Dv2(pretrained_backbone=False)
    model.eval()

    if args.checkpoint and Path(args.checkpoint).exists():
        load_checkpoint(model, args.checkpoint, version=args.version)
    else:
        print('[ckpt] no checkpoint — using random weights')

    if hasattr(model, 'reset_state'):
        model.reset_state()

    saved = []
    with torch.no_grad():
        for frame_idx, frame in enumerate(loader.iter_scene(args.scene)):
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            imgs_np   = frame['imgs'][0].numpy()           # (N_cam, 3, H, W)
            img_metas = frame['img_metas']
            proj      = img_metas['projection_mat']        # (N_cam, 4, 4)

            t0  = time.perf_counter()
            out = model(frame['imgs'].float(), img_metas)
            ms  = (time.perf_counter() - t0) * 1000

            dets    = out['detections'][0]
            boxes   = dets['boxes_3d'].cpu().numpy()      # (K, 9) metric
            scores  = dets['scores_3d'].cpu().numpy()
            labels  = dets['labels_3d'].cpu().numpy()

            n_vis = int((scores >= args.score_thresh).sum())
            print(f'  frame {frame_idx:3d}  dets={boxes.shape[0]:3d}'
                  f'  visible(>{args.score_thresh})={n_vis}  {ms:.0f}ms')

            path = visualise_frame(
                imgs_np, proj, boxes, scores, labels,
                frame_idx, out_dir, args.score_thresh,
            )
            saved.append(path)

    print(f'\n[Sparse4D viz] saved {len(saved)} frames to {out_dir}/')


if __name__ == '__main__':
    main()
