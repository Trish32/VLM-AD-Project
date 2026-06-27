#!/usr/bin/env python3
"""
BEVFormer-Tiny inference on nuScenes mini v1.0.
Produces per-frame BEV images with predicted 3-D boxes overlaid.

Usage:
    python tools/infer.py \
        --dataroot /Users/trish/Downloads/nuScenes_miniV1.0 \
        --checkpoint model/checkpoints/bevformer_tiny_fp16_epoch_24.pth \
        [--scene 0] [--max-frames 40] [--score-thr 0.25] [--out-dir bev_outputs]

Each output PNG shows:
  - Plasma-coloured BEV feature magnitude heatmap
  - 10 / 20 / 30 / 40 m range rings
  - Ego-vehicle indicator (white triangle, forward = up)
  - Per-class coloured rotated detection boxes with class label and score
  - Class legend (top-right) and frame info bar (bottom)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))   # so eval.py can be imported

from model import BEVFormerTiny
from data import NuScenesMiniLoader
from eval import _build_remap                    # extended checkpoint remapper


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# BGR colours, one per class — chosen for maximum distinguishability
CLASS_COLORS_BGR = [
    (  0, 220,   0),   # car               — green
    (  0, 128, 255),   # truck             — orange
    (200,   0, 200),   # construction_veh  — magenta
    (255,   0, 128),   # bus               — deep pink
    (255, 165,   0),   # trailer           — blue-sky
    (160, 160, 160),   # barrier           — grey
    (255, 255,   0),   # motorcycle        — cyan
    (  0, 255, 255),   # bicycle           — yellow
    (  0,   0, 255),   # pedestrian        — red
    (  0, 200, 200),   # traffic_cone      — teal
]

IMG_SIZE  = 800   # output image resolution (square)
HALF_RANGE = 51.2  # half of [−51.2, 51.2] m


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(model: BEVFormerTiny, path: str) -> None:
    ckpt  = torch.load(path, map_location='cpu')
    raw   = ckpt.get('state_dict', ckpt)
    remap = _build_remap(raw)
    result = model.load_state_dict(remap, strict=False)
    print(f'[ckpt] loaded {len(remap)} keys  |  '
          f'missing {len(result.missing_keys)}  |  '
          f'unexpected {len(result.unexpected_keys)}')


# ---------------------------------------------------------------------------
# Decode model output
# ---------------------------------------------------------------------------

def decode_predictions(
    cls_logits: torch.Tensor,   # (Q, C)
    reg_preds:  torch.Tensor,   # (Q, 10)
    ref_pts:    torch.Tensor,   # (Q, 3)  refined reference points in [0,1]
    score_thr:  float = 0.25,
    max_num:    int   = 200,
) -> list[dict]:
    """NMSFreeCoder-style decode using iteratively refined ref_pts for xyz."""
    scores, labels = cls_logits.sigmoid().max(-1)
    order = scores.argsort(descending=True)[:max_num]

    xr = PC_RANGE[3] - PC_RANGE[0]
    yr = PC_RANGE[4] - PC_RANGE[1]
    zr = PC_RANGE[5] - PC_RANGE[2]

    dets = []
    for idx in order:
        s = float(scores[idx])
        if s < score_thr:
            break
        r = reg_preds[idx].float()
        p = ref_pts[idx].float()

        x = float(p[0]) * xr + PC_RANGE[0]
        y = float(p[1]) * yr + PC_RANGE[1]
        z = float(p[2]) * zr + PC_RANGE[2]
        w = float(np.clip(r[2].exp().item(), 0.1, 20.0))   # log_w at index 2
        l = float(np.clip(r[3].exp().item(), 0.1, 20.0))   # log_l at index 3
        h = float(np.clip(r[5].exp().item(), 0.1, 10.0))   # log_h at index 5
        yaw = float(torch.atan2(r[6], r[7]))

        dets.append({
            'score':  s,
            'label':  int(labels[idx]),
            'cx': x, 'cy': y, 'cz': z,
            'w': w, 'l': l, 'h': h,
            'yaw': yaw,
        })
    return dets


# ---------------------------------------------------------------------------
# BEV coordinate helpers
# ---------------------------------------------------------------------------

def lidar_to_img(cx: float, cy: float) -> tuple[int, int]:
    """
    LiDAR (cx, cy) → image pixel (px, py).
    Convention: forward (+X_lidar) = up (−Y_image), left (+Y_lidar) = left (−X_image).
    Ego vehicle sits at the image centre (IMG_SIZE//2, IMG_SIZE//2).
    """
    px = int((HALF_RANGE - cy) / (2 * HALF_RANGE) * IMG_SIZE)
    py = int((HALF_RANGE - cx) / (2 * HALF_RANGE) * IMG_SIZE)
    return px, py


def box_corners_img(cx_img: int, cy_img: int,
                    w_pix: float, l_pix: float,
                    yaw: float) -> np.ndarray:
    """
    4 corners (int32, shape [4,2]) of a rotated box in image space.
    yaw = LiDAR yaw (0 = forward = up in image).
    w_pix = lateral width in pixels, l_pix = longitudinal length in pixels.
    """
    fwd_x = -np.sin(yaw)    # forward direction in image +x
    fwd_y = -np.cos(yaw)    # forward direction in image +y (down)
    sid_x =  np.cos(yaw)    # right-side direction in image
    sid_y = -np.sin(yaw)

    hl, hw = l_pix / 2.0, w_pix / 2.0
    corners = np.array([
        [cx_img + fwd_x*hl + sid_x*hw, cy_img + fwd_y*hl + sid_y*hw],
        [cx_img + fwd_x*hl - sid_x*hw, cy_img + fwd_y*hl - sid_y*hw],
        [cx_img - fwd_x*hl - sid_x*hw, cy_img - fwd_y*hl - sid_y*hw],
        [cx_img - fwd_x*hl + sid_x*hw, cy_img - fwd_y*hl + sid_y*hw],
    ], dtype=np.float32)
    return corners.astype(np.int32)


def meters_to_pixels(dist_m: float) -> float:
    return dist_m / (2 * HALF_RANGE) * IMG_SIZE


# ---------------------------------------------------------------------------
# BEV image rendering
# ---------------------------------------------------------------------------

def render_bev(
    bev_feat: torch.Tensor,   # (1, L, C) on any device
    dets: list[dict],
    scene_name: str,
    frame_idx: int,
    elapsed_ms: float,
) -> np.ndarray:
    """
    Returns an (IMG_SIZE, IMG_SIZE, 3) uint8 BGR image.
    """
    # ---- 1. BEV feature magnitude → plasma heatmap -----------------------
    mag = bev_feat.squeeze(0).norm(dim=-1)       # (L,) on original device
    mag = mag.reshape(BEVFormerTiny.BEV_H, BEVFormerTiny.BEV_W).float().cpu().numpy()
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    mag_u8 = (mag * 255).astype(np.uint8)
    heat = cv2.applyColorMap(mag_u8, cv2.COLORMAP_PLASMA)      # (50, 50, 3)
    canvas = cv2.resize(heat, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # Darken slightly so overlaid lines are readable
    canvas = (canvas * 0.6).astype(np.uint8)

    # ---- 2. Distance rings at 10 / 20 / 30 / 40 m -----------------------
    EGO_X, EGO_Y = IMG_SIZE // 2, IMG_SIZE // 2
    ring_colour   = (90, 90, 90)
    ring_label_c  = (130, 130, 130)
    for d in [10, 20, 30, 40]:
        r = int(meters_to_pixels(d))
        cv2.circle(canvas, (EGO_X, EGO_Y), r, ring_colour, 1, cv2.LINE_AA)
        cv2.putText(canvas, f'{d}m', (EGO_X + r + 3, EGO_Y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, ring_label_c, 1, cv2.LINE_AA)

    # ---- 3. Thin grid lines at ±10 m intervals ---------------------------
    grid_colour = (55, 55, 55)
    for offset_m in range(-40, 50, 10):
        off_pix = int(meters_to_pixels(offset_m))
        # Vertical lines (±Y_lidar = horizontal in image)
        x_pix = EGO_X - off_pix
        if 0 <= x_pix < IMG_SIZE:
            cv2.line(canvas, (x_pix, 0), (x_pix, IMG_SIZE), grid_colour, 1)
        # Horizontal lines (±X_lidar = vertical in image)
        y_pix = EGO_Y - off_pix
        if 0 <= y_pix < IMG_SIZE:
            cv2.line(canvas, (0, y_pix), (IMG_SIZE, y_pix), grid_colour, 1)

    # Axis labels — draw real arrows (OpenCV's Hershey font can't render the
    # Unicode ↑ / ← glyphs, which previously showed up as "???").
    mid = IMG_SIZE // 2
    axis_c = (200, 200, 200)
    cv2.arrowedLine(canvas, (mid, 22), (mid, 6), axis_c, 1, cv2.LINE_AA, tipLength=0.5)
    cv2.putText(canvas, 'Fwd', (mid + 6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_c, 1, cv2.LINE_AA)
    cv2.arrowedLine(canvas, (24, mid), (6, mid), axis_c, 1, cv2.LINE_AA, tipLength=0.5)
    cv2.putText(canvas, 'L', (28, mid + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_c, 1, cv2.LINE_AA)

    # ---- 4. Ego vehicle triangle -----------------------------------------
    tri_size = 10
    ego_tri = np.array([
        [EGO_X,            EGO_Y - tri_size],   # tip (forward)
        [EGO_X - tri_size, EGO_Y + tri_size],
        [EGO_X + tri_size, EGO_Y + tri_size],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [ego_tri], (255, 255, 255))
    cv2.polylines(canvas, [ego_tri], True, (0, 0, 0), 1)

    # ---- 5. Detection boxes ----------------------------------------------
    ppm = IMG_SIZE / (2 * HALF_RANGE)   # pixels per metre

    for det in dets:
        cls_idx = det['label']
        color   = CLASS_COLORS_BGR[cls_idx % len(CLASS_COLORS_BGR)]
        name    = CLASS_NAMES[cls_idx]
        score   = det['score']

        px, py = lidar_to_img(det['cx'], det['cy'])

        # Skip boxes mostly outside the image
        margin = 40
        if not (-margin <= px < IMG_SIZE + margin and
                -margin <= py < IMG_SIZE + margin):
            continue

        w_pix = det['w'] * ppm
        l_pix = det['l'] * ppm

        corners = box_corners_img(px, py, w_pix, l_pix, det['yaw'])
        cv2.drawContours(canvas, [corners], 0, color, 2, cv2.LINE_AA)

        # Forward-direction tick (small line from center toward front of box)
        fwd_x = -np.sin(det['yaw'])
        fwd_y = -np.cos(det['yaw'])
        front_x = int(px + fwd_x * l_pix * 0.45)
        front_y = int(py + fwd_y * l_pix * 0.45)
        cv2.line(canvas, (px, py), (front_x, front_y), color, 2, cv2.LINE_AA)

        # Centre dot
        cv2.circle(canvas, (px, py), 2, color, -1)

        # Label: class abbreviation + score
        abbrev = name[:3].upper()
        label  = f'{abbrev} {score:.2f}'
        tx, ty = px + 4, py - 4
        # Black outline for readability
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)

    # ---- 6. Legend (top-right) -------------------------------------------
    legend_x = IMG_SIZE - 155
    legend_y = 10
    cv2.rectangle(canvas, (legend_x - 4, legend_y - 2),
                  (IMG_SIZE - 4, legend_y + len(CLASS_NAMES) * 15 + 4),
                  (20, 20, 20), -1)
    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS_BGR)):
        y = legend_y + i * 15 + 12
        cv2.rectangle(canvas, (legend_x, y - 8), (legend_x + 10, y + 2), color, -1)
        cv2.putText(canvas, cls_name, (legend_x + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    # ---- 7. Info bar (bottom) --------------------------------------------
    bar_h = 30
    cv2.rectangle(canvas, (0, IMG_SIZE - bar_h), (IMG_SIZE, IMG_SIZE), (15, 15, 15), -1)
    info = (f'Scene {scene_name} | frame {frame_idx:03d} | '
            f'{len(dets)} dets | {elapsed_ms:.0f} ms')
    cv2.putText(canvas, info, (8, IMG_SIZE - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Camera mosaic with projected 3-D box overlays
# ---------------------------------------------------------------------------

# Edges of a 3-D box: corners indexed as:
#   top face: 0,1,2,3   bottom face: 4,5,6,7
#   (0=front-left-top, 1=front-right-top, 2=back-right-top, 3=back-left-top, ...)
_BOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),   # top face
    (4,5),(5,6),(6,7),(7,4),   # bottom face
    (0,4),(1,5),(2,6),(3,7),   # verticals
]

def _box_corners_lidar(cx: float, cy: float, cz: float,
                        w: float, l: float, h: float, yaw: float) -> np.ndarray:
    """Returns (8, 3) float32 corners in LiDAR frame (x=fwd, y=left, z=up)."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_y, -sin_y, 0.],
                  [sin_y,  cos_y, 0.],
                  [0.,     0.,    1.]], dtype=np.float32)
    # local corners: [±l/2, ±w/2, ±h/2] — top face first, then bottom
    lx, wy, hz = l / 2., w / 2., h / 2.
    local = np.array([
        [ lx,  wy,  hz], [ lx, -wy,  hz],
        [-lx, -wy,  hz], [-lx,  wy,  hz],   # top
        [ lx,  wy, -hz], [ lx, -wy, -hz],
        [-lx, -wy, -hz], [-lx,  wy, -hz],   # bottom
    ], dtype=np.float32)
    return (R @ local.T).T + np.array([cx, cy, cz], dtype=np.float32)


def _project_corners(corners: np.ndarray, lidar2img: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    corners   : (8, 3) float32 LiDAR-frame points
    lidar2img : (4, 4) float32 projection matrix (already scaled to 800×480)
    Returns pts2d (8, 2) pixel coords and depths (8,) float32.
    """
    pts_h = np.hstack([corners, np.ones((8, 1), dtype=np.float32)])  # (8, 4)
    proj  = (lidar2img @ pts_h.T).T                                   # (8, 4)
    depths = proj[:, 2]
    safe_z = np.where(np.abs(depths) > 1e-4, depths, 1e-4)
    pts2d  = proj[:, :2] / safe_z[:, None]
    return pts2d.astype(np.float32), depths.astype(np.float32)


def _draw_boxes_on_thumb(thumb: np.ndarray, dets: list[dict],
                          lidar2img: np.ndarray,
                          orig_w: int = 800, orig_h: int = 480) -> np.ndarray:
    """
    thumb     : (THUMB_H, THUMB_W, 3) uint8 BGR image.
    lidar2img : (4, 4) projection matrix calibrated for (orig_w × orig_h) pixel space.
    """
    th, tw = thumb.shape[:2]
    sx, sy = tw / orig_w, th / orig_h

    for det in dets:
        corners = _box_corners_lidar(
            det['cx'], det['cy'], det['cz'],
            det['w'], det['l'], det['h'], det['yaw'],
        )
        pts2d, depths = _project_corners(corners, lidar2img)

        # Scale to thumbnail size
        pts2d[:, 0] *= sx
        pts2d[:, 1] *= sy

        color = CLASS_COLORS_BGR[det['label'] % len(CLASS_COLORS_BGR)]

        # Draw only edges where both endpoints are in front of the camera
        for i0, i1 in _BOX_EDGES:
            if depths[i0] <= 0.1 or depths[i1] <= 0.1:
                continue
            p0 = (int(pts2d[i0, 0]), int(pts2d[i0, 1]))
            p1 = (int(pts2d[i1, 0]), int(pts2d[i1, 1]))
            # At least one endpoint inside the image (with margin) → draw
            margin = 30
            if any(-margin <= p[0] <= tw + margin and
                   -margin <= p[1] <= th + margin
                   for p in (p0, p1)):
                cv2.line(thumb, p0, p1, color, 1, cv2.LINE_AA)

        # Place label at the top-most visible corner
        vis = depths > 0.1
        if not vis.any():
            continue
        vis_pts = pts2d[vis]
        # Top-most = smallest y-pixel value
        top_idx = int(vis_pts[:, 1].argmin())
        lx, ly = int(vis_pts[top_idx, 0]), int(vis_pts[top_idx, 1]) - 3
        if 0 <= lx < tw and 0 <= ly < th:
            name   = CLASS_NAMES[det['label']]
            label  = f"{name[:3].upper()} {det['score']:.2f}"
            cv2.putText(thumb, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(thumb, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    return thumb


def render_camera_mosaic(imgs_np: np.ndarray,
                          dets: list[dict] | None = None,
                          img_metas: list | None = None) -> np.ndarray:
    """
    imgs_np   : (6, 3, H, W) float32 [0,255] BGR — loader native format.
    dets      : optional decoded detections (if given, boxes are projected onto cams).
    img_metas : optional img_metas list (required when dets is not None).
    Returns a (2 * THUMB_H, 3 * THUMB_W, 3) BGR mosaic.
    """
    THUMB_W, THUMB_H = 400, 225
    cam_labels = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK',  'CAM_BACK_LEFT',   'CAM_BACK_RIGHT',
    ]
    # Layout: row-0 = FRONT, FR, FL; row-1 = BACK, BL, BR
    order = [0, 1, 2, 3, 5, 4]
    row0, row1 = [], []
    for i, cam_i in enumerate(order):
        # imgs_np is BGR (cv2.imread format) — no channel conversion needed
        img   = imgs_np[cam_i].transpose(1, 2, 0).astype(np.uint8)   # (H, W, 3) BGR
        thumb = cv2.resize(img, (THUMB_W, THUMB_H))

        # Project and draw 3-D boxes if provided
        if dets is not None and img_metas is not None:
            l2i = np.array(img_metas[0]['lidar2img'][cam_i], dtype=np.float32)
            thumb = _draw_boxes_on_thumb(thumb, dets, l2i)

        # Camera label (bottom-left, dark background for readability)
        lbl = cam_labels[cam_i]
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(thumb, (2, THUMB_H - th - 8), (tw + 6, THUMB_H - 2), (0,0,0), -1)
        cv2.putText(thumb, lbl, (4, THUMB_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        (row0 if i < 3 else row1).append(thumb)
    return np.vstack([np.hstack(row0), np.hstack(row1)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='BEVFormer-Tiny inference + BEV visualisation')
    parser.add_argument('--dataroot',   default='/Users/trish/Downloads/nuScenes_miniV1.0')
    parser.add_argument('--checkpoint', default='model/checkpoints/bevformer_tiny_fp16_epoch_24.pth')
    parser.add_argument('--scene',      type=int, default=0,
                        help='Scene index 0-9 (0=scene-0061, etc.)')
    parser.add_argument('--max-frames', type=int, default=40)
    parser.add_argument('--score-thr',  type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--out-dir',    default='bev_outputs')
    parser.add_argument('--save-cams',  action='store_true',
                        help='Also save 6-camera mosaic images')
    args = parser.parse_args()

    # ---- Device -----------------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'[INFO] device     : {device}')

    # ---- Model ------------------------------------------------------------
    model = BEVFormerTiny(pretrained_backbone=False)
    model.eval()
    load_checkpoint(model, args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[INFO] parameters : {n_params:.1f} M')

    # ---- Data loader ------------------------------------------------------
    loader = NuScenesMiniLoader(args.dataroot)
    scene  = loader.nusc.scene[args.scene]
    scene_name = scene['name']
    print(f'[INFO] scene      : {scene_name} ({args.scene})')
    print(f'[INFO] score_thr  : {args.score_thr}')

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_cams:
        os.makedirs(os.path.join(args.out_dir, 'cameras'), exist_ok=True)

    # ---- Inference loop ---------------------------------------------------
    prev_bev  = None
    frame_idx = 0
    total_ms  = 0.0

    print()
    with torch.no_grad():
        for sample in loader.iter_scene(scene_idx=args.scene):
            if frame_idx >= args.max_frames:
                break

            imgs      = sample['imgs']        # (1, 6, 3, H, W)
            img_metas = sample['img_metas']

            t0 = time.perf_counter()
            out = model(imgs, img_metas, prev_bev=prev_bev)
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_ms  += elapsed_ms

            # NaN/Inf guard — silent corruption on MPS can produce garbage boxes
            _bad = False
            for _key in ('cls_logits', 'reg_preds', 'ref_pts'):
                _t = out[_key]
                _n = int(torch.isnan(_t).sum())
                _i = int(torch.isinf(_t).sum())
                if _n or _i:
                    print(f'  [WARN] frame {frame_idx:03d}: {_key} '
                          f'has {_n} NaN + {_i} Inf — dropping prev_bev, skipping frame')
                    _bad = True
            if _bad:
                prev_bev = None
                frame_idx += 1
                continue

            prev_bev = out['bev_feat'].detach()

            # Decode with proper ref_pts
            dets = decode_predictions(
                out['cls_logits'][0].cpu(),
                out['reg_preds'][0].cpu(),
                out['ref_pts'][0].cpu(),
                score_thr=args.score_thr,
            )

            # Per-class detection counts for terminal log
            cls_counts: dict[str, int] = {}
            for d in dets:
                n = CLASS_NAMES[d['label']]
                cls_counts[n] = cls_counts.get(n, 0) + 1
            count_str = '  '.join(f'{n}:{c}' for n, c in sorted(cls_counts.items()))
            tok = img_metas[0]['sample_token']
            print(f'  frame {frame_idx:03d} | {elapsed_ms:6.1f} ms | '
                  f'{len(dets):3d} dets | {count_str or "—"} | {tok[:8]}')

            # Render and save BEV image
            bev_img = render_bev(
                out['bev_feat'], dets, scene_name, frame_idx, elapsed_ms)
            bev_path = os.path.join(args.out_dir, f'bev_{frame_idx:03d}.png')
            cv2.imwrite(bev_path, bev_img)

            # Optionally save camera mosaic with projected 3-D boxes
            if args.save_cams:
                cam_img  = render_camera_mosaic(imgs[0].cpu().numpy(), dets, img_metas)
                cam_path = os.path.join(args.out_dir, 'cameras',
                                        f'cams_{frame_idx:03d}.png')
                cv2.imwrite(cam_path, cam_img)

            frame_idx += 1

    avg_ms = total_ms / max(frame_idx, 1)
    print(f'\n[DONE] {frame_idx} frames | avg {avg_ms:.1f} ms ({1000/avg_ms:.1f} FPS)')
    print(f'[DONE] BEV images saved → {args.out_dir}/')


if __name__ == '__main__':
    main()
