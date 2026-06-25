"""
BEV scene canvas visualizer — global-frame, north-up.

    canvas = build_scene_canvas(pred_occupancy, ego_pose, nusc_map,
                                 patch_origin, patch_range)

Canvas coordinate system
------------------------
Origin (0, 0) of the global frame maps to canvas centre only when
patch_origin = (0, 0).  In practice patch_origin is fixed to the first
frame's ego position so the ego vehicle moves across the canvas as the
scene progresses.

    +X (global east)  →  col increases
    +Y (global north) →  row decreases   (standard image convention)

    col = canvas_size/2 + (gx − patch_origin_x) × scale
    row = canvas_size/2 − (gy − patch_origin_y) × scale

Four rendered layers (composited in order)
------------------------------------------
    1. BLACK  background
    2. GREEN  drivable area  (nuScenes map API, patch_angle=0 → north-up)
    3. COLOURED  detections   (ego→global transform applied; 4 class groups)
    4. BLUE   ego circle at actual global position; arrow shows actual heading
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion

__all__ = ['build_scene_canvas', 'make_trajectory_canvas', 'GROUP_COLORS', 'CLASS_GROUP']

# ── Class groups ───────────────────────────────────────────────────────────────

# Maps each of the 10 nuScenes class indices to a named group
CLASS_GROUP = [
    'vehicle',       # 0  car
    'vehicle',       # 1  truck
    'vehicle',       # 2  construction_vehicle
    'vehicle',       # 3  bus
    'vehicle',       # 4  trailer
    'barrier',       # 5  barrier
    'vehicle',       # 6  motorcycle
    'vehicle',       # 7  bicycle
    'pedestrian',    # 8  pedestrian
    'traffic_cone',  # 9  traffic_cone
]

# Per-group RGB fill colours — matched to eval_nuscenes._CLASS_PALETTE
GROUP_COLORS: dict[str, tuple] = {
    'vehicle':      (220,  50,  50),   # red      (palette index 0)
    'pedestrian':   ( 50, 100, 220),   # blue     (palette index 2)
    'barrier':      ( 50, 210, 210),   # cyan     (palette index 5 – static)
    'traffic_cone': (200,  50, 200),   # magenta  (palette index 4)
}

PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# nuScenes category prefix → our group name (for GT annotation drawing)
_NUSC_CAT_TO_GROUP: dict[str, str] = {
    'vehicle':                    'vehicle',
    'human.pedestrian':           'pedestrian',
    'movable_object.barrier':     'barrier',
    'movable_object.trafficcone': 'traffic_cone',
}

# Background: dark teal off-road / lighter teal drivable — from save_bev_viz bg formula
# off-road  ≈ float(0.06, 0.12, 0.12) → RGB( 15,  31,  31)
# drivable  ≈ float(0.14, 0.40, 0.36) → RGB( 36, 102,  92)
_BG_COLOR    = ( 15,  31,  31)
_ROAD_COLOR  = ( 36, 102,  92)
_EGO_FILL    = ( 40,  90, 210)
_EGO_OUTLINE = (240, 240, 240)


def _edge_color(fill: tuple) -> tuple:
    """Lighten a fill colour 50 % toward white for the box outline."""
    return tuple(min(255, c + (255 - c) // 2) for c in fill)


# ── Canvas ↔ global coordinate helpers ────────────────────────────────────────

def _global_to_pixel(gx: float, gy: float,
                      patch_ox: float, patch_oy: float,
                      canvas_half: int, scale: float) -> tuple[int, int]:
    """
    Global metres → canvas (col, row).
    patch_ox, patch_oy : global coords of canvas centre.
    """
    col = int(round(canvas_half + (gx - patch_ox) * scale))
    row = int(round(canvas_half - (gy - patch_oy) * scale))
    return col, row


def _box_corners_global(col_c: float, row_c: float,
                         l_px: float, w_px: float,
                         global_yaw: float) -> np.ndarray:
    """
    Rotated box corners for a NORTH-UP canvas.

    col_c, row_c  : box centre in canvas pixels
    l_px, w_px    : length (along heading) and width in pixels
    global_yaw    : heading in radians measured CCW from global +X (east)
                    yaw=0  → pointing east  → box stretches horizontally
                    yaw=π/2→ pointing north → box stretches vertically (upward)

    Canvas mapping:  d_col = +dx_global,  d_row = -dy_global
    Returns (4, 1, 2) int32 for cv2.fillPoly / cv2.polylines.
    """
    c, s = math.cos(global_yaw), math.sin(global_yaw)
    hl, hw = l_px / 2.0, w_px / 2.0

    # (d_col, d_row) offsets for [FL, FR, RR, RL]:
    #   fwd global = (c, s),  canvas = (+c, -s)
    #   left global = (-s, c), canvas = (-s, -c)
    offsets = np.array([
        [ c*hl - s*hw, -(s*hl + c*hw)],   # FL
        [ c*hl + s*hw, -(s*hl - c*hw)],   # FR
        [-c*hl + s*hw,  s*hl + c*hw ],    # RR
        [-c*hl - s*hw,  s*hl - c*hw ],    # RL
    ], dtype=np.float32)

    offsets[:, 0] += col_c
    offsets[:, 1] += row_c
    return offsets.astype(np.int32).reshape(-1, 1, 2)


# ── Layer 2: drivable area ─────────────────────────────────────────────────────

def _draw_drivable_area(
    canvas:      np.ndarray,
    nusc_map,
    patch_ox:    float,
    patch_oy:    float,
    patch_range: float,
    canvas_size: int,
) -> None:
    """
    Render drivable area onto *canvas* (in-place) using a north-up patch
    centred on (patch_ox, patch_oy) in global metres.

    nuScenes get_map_mask convention (from _polygon_geom_to_mask):
      trans_y = patch_h/2  →  global north (large y) maps to large row (bottom).
    Our canvas convention:
      row = canvas_half − (gy − patch_oy) × scale  →  north = row 0 (top).
    The mask must be flipped vertically before being applied.
    """
    if nusc_map is None:
        return
    try:
        mask = nusc_map.get_map_mask(
            (patch_ox, patch_oy, patch_range, patch_range),
            0.0,
            ['drivable_area'],
            canvas_size=(canvas_size, canvas_size),
        )                                          # (1, H, W) uint8
        # flip vertically: nuScenes rasters north→bottom, canvas is north→top
        road_mask = mask[0, ::-1, :]
        canvas[road_mask.astype(bool)] = _ROAD_COLOR
    except Exception:
        pass


# ── Layer 3: detections ────────────────────────────────────────────────────────

def _draw_detections(
    base:          np.ndarray,
    cls_logits:    torch.Tensor,   # (Q, C)
    reg_preds:     torch.Tensor,   # (Q, 10)
    ref_pts:       torch.Tensor,   # (Q, 3)  normalised [0,1] in PC_RANGE
    ego_tx:        float,          # ego global x (metres)
    ego_ty:        float,          # ego global y (metres)
    ego_yaw:       float,          # ego2global heading (radians, CCW from east)
    patch_ox:      float,
    patch_oy:      float,
    canvas_size:   int,
    scale:         float,
    score_thr:     float,
    box_alpha:     float,
    lidar2ego_yaw: float = 0.0,   # yaw of lidar2ego calibration (radians)
) -> np.ndarray:
    """
    Transform each detection from LiDAR frame → global frame, then draw.

    The BEVFormer BEV grid is defined in the LiDAR sensor frame, which in
    nuScenes has a fixed rotation relative to the ego frame (lidar2ego_yaw,
    typically ~-90° because the LiDAR x-axis = vehicle right, y-axis = forward).
    The full LiDAR→global rotation angle is ego_yaw + lidar2ego_yaw.

    Returns alpha-blended copy of *base*.
    """
    canvas_half = canvas_size // 2
    margin      = 16

    scores, labels = cls_logits.float().sigmoid().max(-1)

    # Take top-200 by score then filter by threshold (mirrors decode_predictions).
    order   = scores.argsort(descending=True)[:200]
    indices = [int(i) for i in order if float(scores[i]) > score_thr]

    det_layer = base.copy()
    # Combined LiDAR→global rotation: ego2global ∘ lidar2ego
    yaw_total = ego_yaw + lidar2ego_yaw
    cos_t, sin_t = math.cos(yaw_total), math.sin(yaw_total)

    for idx in indices:
        r      = reg_preds[idx].float()
        p      = ref_pts[idx].float()          # normalised [0,1] in LiDAR frame
        label  = int(labels[idx])
        group  = CLASS_GROUP[label % len(CLASS_GROUP)]
        fill   = GROUP_COLORS[group]

        # Denormalise from normalised BEV coords to LiDAR-frame metres
        x_lid = float(p[0]) * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
        y_lid = float(p[1]) * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
        # Size: reg_preds are log-encoded (w=index 2, l=index 3)
        w_m     = float(np.clip(np.exp(float(r[2])), 0.2, 20.0))
        l_m     = float(np.clip(np.exp(float(r[3])), 0.4, 20.0))
        yaw_lid = float(torch.atan2(r[6], r[7]))

        # LiDAR frame → global frame (full rotation: lidar2ego then ego2global)
        gx = cos_t * x_lid - sin_t * y_lid + ego_tx
        gy = sin_t * x_lid + cos_t * y_lid + ego_ty
        # BEVFormer trains with SECOND-format yaw: second_yaw = -nusc_yaw_lidar - π/2
        # Recover global yaw: global = yaw_total + nusc_yaw_lidar = yaw_total - yaw_lid - π/2
        global_yaw = yaw_total - yaw_lid - math.pi / 2

        col_c, row_c = _global_to_pixel(gx, gy, patch_ox, patch_oy,
                                         canvas_half, scale)

        if not (-margin <= col_c < canvas_size + margin and
                -margin <= row_c < canvas_size + margin):
            continue

        l_px = max(l_m * scale, 4.0)
        w_px = max(w_m * scale, 3.0)

        corners = _box_corners_global(col_c, row_c, l_px, w_px, global_yaw)
        cv2.fillPoly(det_layer, [corners], color=fill)
        cv2.polylines(det_layer, [corners], isClosed=True,
                      color=_edge_color(fill), thickness=1)

    return cv2.addWeighted(det_layer, box_alpha, base, 1.0 - box_alpha, 0)


# ── GT annotation boxes ────────────────────────────────────────────────────────

def _draw_gt_boxes(
    canvas:      np.ndarray,
    nusc,
    sample_token: str,
    patch_ox:    float,
    patch_oy:    float,
    canvas_size: int,
    scale:       float,
) -> None:
    """
    Draw ground-truth 3-D annotation boxes (outline-only) onto *canvas* in-place.

    Boxes are in global frame (nuScenes annotation.translation).  nuScenes size
    is [width, length, height]; yaw comes from the annotation quaternion.
    Outline-only style (no fill) distinguishes GT from the filled predicted boxes.
    """
    canvas_half = canvas_size // 2
    margin      = 20

    sample = nusc.get('sample', sample_token)
    for ann_token in sample['anns']:
        ann      = nusc.get('sample_annotation', ann_token)
        cat      = ann['category_name']       # e.g. 'vehicle.car'
        tx, ty   = ann['translation'][0], ann['translation'][1]
        w_m, l_m = ann['size'][0], ann['size'][1]  # width, length (metres)
        yaw      = Quaternion(ann['rotation']).yaw_pitch_roll[0]

        # Map category → group; skip unknowns
        group = None
        for prefix, g in _NUSC_CAT_TO_GROUP.items():
            if cat.startswith(prefix):
                group = g
                break
        if group is None:
            continue

        col_c, row_c = _global_to_pixel(tx, ty, patch_ox, patch_oy,
                                         canvas_half, scale)
        if not (-margin <= col_c < canvas_size + margin and
                -margin <= row_c < canvas_size + margin):
            continue

        color  = GROUP_COLORS[group]
        l_px   = max(l_m * scale, 4.0)
        w_px   = max(w_m * scale, 3.0)
        corners = _box_corners_global(col_c, row_c, l_px, w_px, yaw)
        cv2.polylines(canvas, [corners], isClosed=True,
                      color=color, thickness=1, lineType=cv2.LINE_AA)


# ── Layer 4: ego vehicle ───────────────────────────────────────────────────────

def _draw_ego(
    canvas:      np.ndarray,
    col_ego:     int,
    row_ego:     int,
    global_yaw:  float,
    scale:       float,
) -> None:
    """
    Draw ego circle at (col_ego, row_ego) with arrow pointing along global_yaw.
    global_yaw=0 → east → arrow points right on a north-up canvas.
    global_yaw=π/2 → north → arrow points up.
    """
    radius    = max(5, int(1.6 * scale))
    arrow_len = int(4.0 * scale)

    # Arrow tip in north-up canvas: +x = east = +col, +y = north = -row
    tip_col = int(col_ego + math.cos(global_yaw) * arrow_len)
    tip_row = int(row_ego - math.sin(global_yaw) * arrow_len)

    cv2.circle(canvas, (col_ego, row_ego), radius + 2, _EGO_OUTLINE,
               thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (col_ego, row_ego), radius, _EGO_FILL,
               thickness=-1, lineType=cv2.LINE_AA)
    cv2.arrowedLine(canvas, (col_ego, row_ego), (tip_col, tip_row),
                    _EGO_OUTLINE, thickness=2, tipLength=0.28,
                    line_type=cv2.LINE_AA)


# ── Public entry point ─────────────────────────────────────────────────────────

def build_scene_canvas(
    pred_occupancy: dict,
    ego_pose:       dict,
    nusc_map,
    patch_origin:   tuple[float, float],
    patch_range:    float = 150.0,
    canvas_size:    int   = 512,
    score_thr:      float = 0.25,
    box_alpha:      float = 0.60,
    lidar2ego_yaw:  float = 0.0,
) -> np.ndarray:
    """
    Composite a north-up BEV canvas from BEVFormer predictions.

    Parameters
    ----------
    pred_occupancy : dict
        BEVFormerTiny.forward() output — 'cls_logits' (1,Q,C), 'reg_preds' (1,Q,10).
    ego_pose : dict
        nuScenes ego_pose record: 'translation' (3,), 'rotation' (4,).
    nusc_map : NuScenesMap | None
        Map for this scene location.  None silently skips the road layer.
    patch_origin : (x_global, y_global)
        Global-frame metres corresponding to the canvas centre.  Set once at
        scene start (first frame's ego translation) and keep fixed so the ego
        vehicle moves across the canvas frame-by-frame.
    patch_range : float
        Total metres covered by canvas edge (default 150 m → 75 m each side).
    canvas_size : int
        Square canvas edge length in pixels.
    score_thr : float
        Minimum sigmoid score to render a detection.
    box_alpha : float
        Detection layer opacity.

    Returns
    -------
    np.ndarray  (canvas_size, canvas_size, 3) uint8 RGB
    """
    scale       = canvas_size / patch_range
    canvas_half = canvas_size // 2
    patch_ox, patch_oy = patch_origin

    ego_tx  = float(ego_pose['translation'][0])
    ego_ty  = float(ego_pose['translation'][1])
    ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]  # radians

    # ── Layer 1: background ────────────────────────────────────────────────────
    canvas = np.full((canvas_size, canvas_size, 3), _BG_COLOR, dtype=np.uint8)

    # ── Layer 2: drivable area ─────────────────────────────────────────────────
    _draw_drivable_area(canvas, nusc_map, patch_ox, patch_oy,
                        patch_range, canvas_size)

    # ── Layer 3: detections ────────────────────────────────────────────────────
    canvas = _draw_detections(
        canvas,
        pred_occupancy['cls_logits'][0].cpu(),
        pred_occupancy['reg_preds'][0].cpu(),
        pred_occupancy['ref_pts'][0].cpu(),
        ego_tx, ego_ty, ego_yaw,
        patch_ox, patch_oy,
        canvas_size, scale,
        score_thr, box_alpha,
        lidar2ego_yaw=lidar2ego_yaw,
    )

    # ── Layer 4: ego vehicle ───────────────────────────────────────────────────
    col_ego, row_ego = _global_to_pixel(ego_tx, ego_ty,
                                         patch_ox, patch_oy,
                                         canvas_half, scale)
    _draw_ego(canvas, col_ego, row_ego, ego_yaw, scale)

    return canvas


# ── GT trajectory canvas ───────────────────────────────────────────────────────

def make_trajectory_canvas(
    ego_history:  list,
    nusc_map,
    patch_origin: tuple[float, float],
    patch_range:  float = 150.0,
    canvas_size:  int   = 512,
    max_trail:    int   = 40,
    nusc          = None,
    sample_token: str   = '',
) -> np.ndarray:
    """
    GT-style trajectory accumulation canvas with ground-truth annotation boxes.

    Renders the same drivable-area map as build_scene_canvas, then overlays:
      - GT annotation boxes (outline-only, group colours) for the current sample
      - Fading blue dots for the last *max_trail* ego positions (oldest = dim)
      - Full ego circle + heading arrow at the current (most recent) position

    ego_history  : list of (tx, ty, yaw) tuples in global metres / radians.
                   The last entry is treated as the current position.
    nusc         : NuScenes instance (optional).  When provided together with
                   *sample_token*, GT boxes are drawn for that sample.
    sample_token : nuScenes sample token for the current frame.

    Returns (canvas_size, canvas_size, 3) uint8 RGB.
    """
    scale       = canvas_size / patch_range
    canvas_half = canvas_size // 2
    patch_ox, patch_oy = patch_origin

    canvas = np.full((canvas_size, canvas_size, 3), _BG_COLOR, dtype=np.uint8)
    _draw_drivable_area(canvas, nusc_map, patch_ox, patch_oy, patch_range, canvas_size)

    # GT annotation boxes (outline-only) for the current frame
    if nusc is not None and sample_token:
        _draw_gt_boxes(canvas, nusc, sample_token,
                       patch_ox, patch_oy, canvas_size, scale)

    if not ego_history:
        return canvas

    trail = ego_history[-max_trail:]
    n     = len(trail)

    # Past positions (all but the last): fading blue circles
    for i, (tx, ty, _) in enumerate(trail[:-1]):
        col, row = _global_to_pixel(tx, ty, patch_ox, patch_oy, canvas_half, scale)
        if not (0 <= col < canvas_size and 0 <= row < canvas_size):
            continue
        alpha  = (i + 1) / n           # 0 = oldest, 1 = just-before-current
        r      = int(10 + 30 * alpha)
        g      = int(20 + 70 * alpha)
        b      = int(60 + 150 * alpha)
        radius = max(2, int(2 + 3 * alpha))
        cv2.circle(canvas, (col, row), radius, (r, g, b), -1, lineType=cv2.LINE_AA)

    # Current position: full bright ego circle + heading arrow
    tx, ty, yaw = trail[-1]
    col, row = _global_to_pixel(tx, ty, patch_ox, patch_oy, canvas_half, scale)
    _draw_ego(canvas, col, row, yaw, scale)

    return canvas
