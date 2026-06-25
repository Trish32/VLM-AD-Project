#!/usr/bin/env python3
"""
BEVFormer-Tiny visualised inference + Qwen2.5VL-7B streaming decisions.

Usage (from tools/):
    # BEV only
    conda run -n simple_bev_vldrive python vis_infer.py \
        --dataroot /Users/trish/Downloads/nuScenes_miniV1.0 \
        --scene 0 --max-frames 10

    # BEV + VLM decisions (streaming, composite updated live)
    conda run -n simple_bev_vldrive python vis_infer.py \
        --scene 0 --max-frames 10 --vl --log decisions.jsonl

Per-frame output:
    bev_outputs/vis_NNN.png          — single prediction BEV canvas
    bev_outputs/latest_bev_grid.jpg  — simple_bev-style collage:
                                         [pred BEV | GT trajectory]   ← top row
                                         [FL | F | FR cameras]        ← front row
                                         [BL | B | BR cameras]        ← rear row
                                       During --vl: GT panel gets a streaming
                                       VLM reasoning/decision overlay.
"""

import argparse
import base64
import json
import math
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pyquaternion import Quaternion

# ── Path setup ─────────────────────────────────────────────────────────────────
TOOLS_DIR = Path(__file__).resolve().parent
ROOT      = TOOLS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from model import BEVFormerTiny
from data  import NuScenesMiniLoader
from visualizer import build_scene_canvas, make_trajectory_canvas, GROUP_COLORS, CLASS_GROUP
from eval   import _build_remap

OUT_DIR     = ROOT / 'bev_outputs'
LATEST_PATH = str(OUT_DIR / 'latest_bev_grid.jpg')

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# ── Fonts ──────────────────────────────────────────────────────────────────────
def _load_fonts(size_body=14, size_title=17):
    candidates = [
        '/System/Library/Fonts/Menlo.ttc',
        '/System/Library/Fonts/SFNSMono.ttf',
        '/System/Library/Fonts/Courier.ttc',
    ]
    for path in candidates:
        try:
            return (ImageFont.truetype(path, size_title),
                    ImageFont.truetype(path, size_body))
        except OSError:
            pass
    fb = ImageFont.load_default()
    return fb, fb

_, _FONT_BODY = _load_fonts()

# ── Qwen2.5VL system prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "Analyze this top-down Bird's Eye View (BEV) map of a driving scene.\n"
    "Canvas is global-frame, north-up.  Colour legend:\n"
    "  RED shapes     — vehicles (car, truck, bus, motorcycle, bicycle)\n"
    "  BLUE shapes    — pedestrians\n"
    "  CYAN shapes    — barriers / static obstacles\n"
    "  MAGENTA shapes — traffic cones\n"
    "  BLUE circle + white arrow — ego vehicle; arrow = heading direction.\n"
    "  Teal surface   — drivable road.\n\n"
    "Determine if there is a blocking hazard ahead and decide:\n"
    "[PROCEED, SLOW_DOWN, YIELD, STOP]\n\n"
    "Reply in EXACTLY this format (no extra lines):\n"
    "REASONING: <one sentence>\n"
    "DECISION: <PROCEED | SLOW_DOWN | YIELD | STOP>"
)

_VALID_DECISIONS = {"PROCEED", "SLOW_DOWN", "YIELD", "STOP"}
_DECISION_COLORS = {
    "PROCEED":   (80,  220,  80),
    "SLOW_DOWN": (220, 220,  50),
    "YIELD":     (220, 140,  50),
    "STOP":      (220,  50,  50),
    "UNKNOWN":   (150, 150, 150),
}
_LABELS = {
    "PROCEED":   "PROCEED   — path clear",
    "SLOW_DOWN": "SLOW DOWN — reduce speed",
    "YIELD":     "YIELD     — cross-traffic",
    "STOP":      "STOP      — immediate hazard",
}

# ── VLM overlay colours ────────────────────────────────────────────────────────
_BG_RGB   = (15, 31, 31)       # matches visualizer dark-teal background
_WHITE    = (230, 230, 230)
_GRAY     = (140, 155, 155)

def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or ['']

def _overlay_vl_text(panel: np.ndarray, text: str) -> np.ndarray:
    """
    Overlay streaming VLM text on the bottom portion of *panel* (RGB uint8).
    h_box and character wrap width are computed from the panel dimensions so
    the overlay works on any size image (GT panel, camera cell, etc.).
    The zone is darkened to a near-black teal before text is drawn.
    """
    H, W   = panel.shape[:2]
    h_box  = min(160, H // 2)          # at most half the image height
    y0     = H - h_box
    wrap_cols = max(18, W // 9)        # ~9 px per Menlo-14 character
    result = panel.copy()

    # Darken overlay zone: 35% original + 65% background teal
    result[y0:] = (
        result[y0:].astype(np.float32) * 0.35
        + np.array(_BG_RGB, np.float32) * 0.65
    ).clip(0, 255).astype(np.uint8)

    img  = Image.fromarray(result)
    draw = ImageDraw.Draw(img)
    pad  = 8

    reasoning, decision = _parse_response(text)
    if not reasoning:
        reasoning = text.strip()

    draw.line([(0, y0), (W, y0)], fill=(60, 100, 100), width=1)
    draw.text((pad, y0 + 4), "REASONING", font=_FONT_BODY, fill=_WHITE)
    y = y0 + 20
    for ln in _wrap(reasoning, wrap_cols)[:3]:
        draw.text((pad, y), ln, font=_FONT_BODY, fill=_GRAY)
        y += 16

    if decision == "UNKNOWN" and text and not text.rstrip().endswith('.'):
        draw.text((pad, y), "▌", font=_FONT_BODY, fill=(100, 200, 200))
    elif decision != "UNKNOWN":
        dec_color = _DECISION_COLORS.get(decision, _DECISION_COLORS["UNKNOWN"])
        dec_label = _LABELS.get(decision, decision)
        y_box = H - 30
        draw.rectangle([(pad, y_box), (W - pad, y_box + 22)],
                       fill=tuple(max(0, c // 5) for c in dec_color),
                       outline=dec_color, width=2)
        draw.text((pad + 6, y_box + 4), dec_label, font=_FONT_BODY, fill=dec_color)

    return np.array(img, dtype=np.uint8)


def _save_bev_dual(pred_panel: np.ndarray,
                   gt_panel:   np.ndarray,
                   vl_text:    str = "",
                   cam_grid:   np.ndarray | None = None) -> None:
    """
    Save composite as latest_bev_grid.jpg (JPEG quality=95, PIL RGB).

    Layout (simple_bev vis_collage style):
        ┌──────────────┬──┬──────────────┐
        │  pred BEV    │  │  GT traj     │  ← 512 × (512+2+512)
        ├──────────────┴──┴──────────────┤
        │  FL  │   F  │  FR              │  ← cell_h × total_w
        │  BL  │   B  │  BR              │    (only when cam_grid provided)
        └──────────────────────────────────┘

    When vl_text is non-empty the VLM reasoning/decision overlay is drawn on
    the bottom of the FRONT camera cell (row 0, centre column of cam_grid).
    The GT trajectory panel is always shown clean.
    """
    H   = pred_panel.shape[0]
    sep = np.full((H, 2, 3), 255, dtype=np.uint8)           # white 2-px line
    top = np.concatenate([pred_panel, sep, gt_panel], axis=1)

    if cam_grid is not None:
        # Apply VLM text overlay to the FRONT camera cell (row 0, col 1).
        if vl_text:
            grid_out  = cam_grid.copy()
            cell_w    = grid_out.shape[1] // 3
            cell_h    = grid_out.shape[0] // 2
            front     = grid_out[0:cell_h, cell_w:2 * cell_w]   # (H, W, 3) RGB
            grid_out[0:cell_h, cell_w:2 * cell_w] = _overlay_vl_text(front, vl_text)
        else:
            grid_out = cam_grid

        bev_w = top.shape[1]
        if grid_out.shape[1] != bev_w:
            new_h = int(round(grid_out.shape[0] * bev_w / grid_out.shape[1]))
            grid_out = cv2.resize(grid_out, (bev_w, new_h),
                                  interpolation=cv2.INTER_AREA)
        h_sep     = np.full((4, bev_w, 3), 40, dtype=np.uint8)
        composite = np.concatenate([top, h_sep, grid_out], axis=0)
    else:
        composite = top

    Image.fromarray(composite, mode='RGB').save(LATEST_PATH, quality=95)


# ── Camera grid (simple_bev collage layout) ────────────────────────────────────
#
# Matches vis_collage.py constraint:  bev_total_w == cam_cell_w * 3
#
#   ┌──────────────┬──────────────┐
#   │  pred BEV    │  GT traj     │   512 × 1026
#   ├──────────────┴──────────────┤
#   │  FL  │   F  │  FR           │   cell_h × 1026
#   │  BL  │   B  │  BR           │
#   └──────────────────────────────┘

CAM_ORDER_VIS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',
]

_CAM_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # top face
    (4, 5), (5, 6), (6, 7), (7, 4),   # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),   # vertical pillars
]

_CAM_FULL_W, _CAM_FULL_H = 1600, 900
_PC_VIS = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


def _box_corners_3d(cx: float, cy: float, cz: float,
                    l: float, w: float, h: float, yaw: float) -> np.ndarray:
    """8 corners of a 3-D box in ego/LiDAR frame (x=east, y=north, z=up)."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw, hh = l / 2.0, w / 2.0, h / 2.0
    local = np.array([
        [ hl,  hw,  hh], [ hl, -hw,  hh], [-hl, -hw,  hh], [-hl,  hw,  hh],
        [ hl,  hw, -hh], [ hl, -hw, -hh], [-hl, -hw, -hh], [-hl,  hw, -hh],
    ], dtype=np.float32)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (R @ local.T).T + np.array([cx, cy, cz], dtype=np.float32)


def _get_vis_lidar2img(nusc, sample_token: str, cam_name: str) -> np.ndarray:
    """
    4×4 lidar→pixel projection using the ORIGINAL 1600×900 camera intrinsics.
    Same chain as data.loader._get_lidar2img but without the 800×480 K scaling.
    """
    sample   = nusc.get('sample', sample_token)
    sd_token = sample['data'][cam_name]
    sd       = nusc.get('sample_data', sd_token)
    cs       = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep       = nusc.get('ego_pose', sd['ego_pose_token'])

    K = np.eye(4, dtype=np.float64)
    K[:3, :3] = np.array(cs['camera_intrinsic'])   # full-res, no scale

    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
    cam2ego[:3,  3] = np.array(cs['translation'])
    ego2cam = np.linalg.inv(cam2ego)

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(ep['rotation']).rotation_matrix
    ego2global[:3,  3] = np.array(ep['translation'])
    global2ego = np.linalg.inv(ego2global)

    lidar_tok   = sample['data']['LIDAR_TOP']
    lidar_sd    = nusc.get('sample_data', lidar_tok)
    lidar_cs    = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ep    = nusc.get('ego_pose',           lidar_sd['ego_pose_token'])

    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(lidar_cs['rotation']).rotation_matrix
    lidar2ego[:3,  3] = np.array(lidar_cs['translation'])

    lego2global = np.eye(4)
    lego2global[:3, :3] = Quaternion(lidar_ep['rotation']).rotation_matrix
    lego2global[:3,  3] = np.array(lidar_ep['translation'])

    return (K @ ego2cam @ global2ego @ lego2global @ lidar2ego).astype(np.float32)


def _draw_pred_on_cam(img_bgr: np.ndarray, lidar2img: np.ndarray,
                       cls_logits_1: torch.Tensor, reg_preds_1: torch.Tensor,
                       ref_pts_1: torch.Tensor,
                       score_thr: float) -> np.ndarray:
    """
    Project predicted 3-D boxes (wireframe) onto *img_bgr* (in-place copy, BGR).
    Uses full-res 1600×900 intrinsics via *lidar2img*.
    Position from iteratively refined ref_pts; size from log-encoded reg_preds.
    """
    img = img_bgr.copy()
    H, W = img.shape[:2]

    scores, labels = cls_logits_1.float().sigmoid().max(-1)
    # top-200 by score, then threshold — mirrors decode_predictions
    order = scores.argsort(descending=True)[:200]
    keep  = [int(i) for i in order if float(scores[i]) > score_thr]

    for idx in keep:
        r     = reg_preds_1[idx].float().numpy()
        p     = ref_pts_1[idx].float().numpy()    # normalised [0,1] refined position
        lbl   = int(labels[idx])
        group = CLASS_GROUP[lbl % len(CLASS_GROUP)]
        rgb_c = GROUP_COLORS[group]
        bgr   = (int(rgb_c[2]), int(rgb_c[1]), int(rgb_c[0]))

        # Position from refined reference points (not raw regression offsets)
        cx  = float(p[0]) * (_PC_VIS[3] - _PC_VIS[0]) + _PC_VIS[0]
        cy  = float(p[1]) * (_PC_VIS[4] - _PC_VIS[1]) + _PC_VIS[1]
        cz  = float(p[2]) * (_PC_VIS[5] - _PC_VIS[2]) + _PC_VIS[2]
        # Size: log-encoded — w=index 2, l=index 3, h=index 5
        w   = float(np.clip(np.exp(r[2]), 0.2, 20.0))
        l   = float(np.clip(np.exp(r[3]), 0.4, 20.0))
        h   = float(np.clip(np.exp(r[5]), 0.2, 10.0))
        yaw = math.atan2(float(r[6]), float(r[7]))

        corners  = _box_corners_3d(cx, cy, cz, l, w, h, yaw)         # (8, 3)
        corners_h = np.concatenate([corners,
                                     np.ones((8, 1), dtype=np.float32)], axis=1)
        proj  = (lidar2img @ corners_h.T).T                           # (8, 4)
        depths = proj[:, 2]

        if depths.max() < 0.1 or (depths > 0.1).sum() < 4:
            continue

        front = depths > 0.1
        denom = np.where(np.abs(proj[:, 2]) > 1e-6, proj[:, 2], 1e-6)
        u = proj[:, 0] / denom
        v = proj[:, 1] / denom
        in_img = (u > -50) & (u < W + 50) & (v > -50) & (v < H + 50)
        if not (in_img & front).any():
            continue

        pts = np.stack([u, v], axis=1).astype(np.float32)
        for i0, i1 in _CAM_EDGES:
            if not (front[i0] and front[i1]):
                continue
            p0 = (int(np.clip(pts[i0, 0], -2000, W + 2000)),
                  int(np.clip(pts[i0, 1], -2000, H + 2000)))
            p1 = (int(np.clip(pts[i1, 0], -2000, W + 2000)),
                  int(np.clip(pts[i1, 1], -2000, H + 2000)))
            cv2.line(img, p0, p1, bgr, 2, lineType=cv2.LINE_AA)

        # Class label at the topmost visible corner
        vis_mask = front & (u > -50) & (u < W + 50) & (v > -50) & (v < H + 50)
        if vis_mask.any():
            vis_v = pts[vis_mask, 1]
            top_local = int(vis_v.argmin())
            vis_pts = pts[vis_mask]
            lx = int(np.clip(vis_pts[top_local, 0], 0, W - 1))
            ly = int(np.clip(vis_pts[top_local, 1] - 4, 4, H - 1))
            name  = CLASS_NAMES[lbl % len(CLASS_NAMES)]
            text  = f"{name[:3].upper()} {scores[idx]:.2f}"
            cv2.putText(img, text, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 1, cv2.LINE_AA)

    return img


def _make_cam_grid(nusc, sample_token: str, dataroot: str,
                   cls_logits_1: torch.Tensor, reg_preds_1: torch.Tensor,
                   ref_pts_1: torch.Tensor,
                   score_thr: float, total_w: int) -> np.ndarray:
    """
    2×3 RGB camera grid matching simple_bev vis_collage layout.
    total_w must equal 3 × cell_w (maintained by caller passing bev_total_w).

    Row 0: FL | F  | FR
    Row 1: BL | B  | BR
    """
    cell_w = total_w // 3
    cell_h = int(round(cell_w * _CAM_FULL_H / _CAM_FULL_W))  # 16:9 aspect

    sample = nusc.get('sample', sample_token)
    rows   = []
    for r_idx in range(2):
        row_cells = []
        for cam_name in CAM_ORDER_VIS[r_idx * 3 : r_idx * 3 + 3]:
            sd_token = sample['data'][cam_name]
            sd       = nusc.get('sample_data', sd_token)
            img_path = Path(dataroot) / sd['filename']

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                img_bgr = np.zeros((_CAM_FULL_H, _CAM_FULL_W, 3), dtype=np.uint8)

            l2i = _get_vis_lidar2img(nusc, sample_token, cam_name)
            img_bgr = _draw_pred_on_cam(img_bgr, l2i,
                                         cls_logits_1, reg_preds_1, ref_pts_1,
                                         score_thr)

            # Camera name label: bottom-left, dark background for readability
            label  = cam_name.replace('CAM_', '').replace('_', ' ')
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
            cv2.rectangle(img_bgr, (8, _CAM_FULL_H - lh - 16),
                          (lw + 16, _CAM_FULL_H - 4), (0, 0, 0), -1)
            cv2.putText(img_bgr, label, (14, _CAM_FULL_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220, 220, 220), 2, cv2.LINE_AA)

            cell = cv2.resize(img_bgr, (cell_w, cell_h),
                               interpolation=cv2.INTER_AREA)
            row_cells.append(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))

        rows.append(np.concatenate(row_cells, axis=1))

    return np.concatenate(rows, axis=0)   # (2*cell_h, total_w, 3) RGB


# ── Ollama streaming ───────────────────────────────────────────────────────────

def _encode_image(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def _query_ollama_streaming(b64: str, model: str, base_url: str,
                              timeout: int, on_update, update_every: int = 6
                              ) -> str:
    """
    Stream from Ollama /api/generate.
    Calls on_update(accumulated_text) every *update_every* tokens and on done.
    Returns the full response string.
    """
    payload = json.dumps({
        "model":  model,
        "prompt": _SYSTEM_PROMPT,
        "images": [b64],
        "stream": True,
        "options": {"temperature": 0.1, "num_predict": 140},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    accumulated  = ""
    token_count  = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                try:
                    obj = json.loads(raw_line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                token        = obj.get("response", "")
                accumulated += token
                token_count += 1

                done = obj.get("done", False)
                if token_count % update_every == 0 or '\n' in token or done:
                    on_update(accumulated)

                if done:
                    break

    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            msg = json.loads(raw).get("error", raw)
        except Exception:
            msg = raw
        raise RuntimeError(f"HTTP {exc.code}: {msg}") from None

    return accumulated


def _parse_response(text: str) -> tuple[str, str]:
    reasoning, decision = "", "UNKNOWN"
    for line in text.strip().splitlines():
        s = line.strip()
        if s.startswith("REASONING:"):
            reasoning = s[len("REASONING:"):].strip()
        elif s.startswith("DECISION:"):
            tok = s[len("DECISION:"):].strip().upper().rstrip(".")
            decision = tok if tok in _VALID_DECISIONS else next(
                (d for d in _VALID_DECISIONS if d in tok), "UNKNOWN"
            )
    return reasoning, decision


# ── Device ─────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Map loader ─────────────────────────────────────────────────────────────────

def _load_nusc_map(dataroot: str, location: str):
    try:
        from nuscenes.map_expansion.map_api import NuScenesMap
        return NuScenesMap(dataroot=dataroot, map_name=location)
    except FileNotFoundError:
        print(f'[WARN] Map expansion files missing for "{location}" — road layer skipped.')
        return None
    except Exception as exc:
        print(f'[WARN] NuScenesMap failed ({exc}) — road layer skipped.')
        return None


def _get_ego_pose(nusc, sample_token: str) -> dict:
    sample    = nusc.get('sample', sample_token)
    lidar_tok = sample['data']['LIDAR_TOP']
    lsd       = nusc.get('sample_data', lidar_tok)
    return nusc.get('ego_pose', lsd['ego_pose_token'])


def _top3_str(cls_logits_1: torch.Tensor) -> str:
    per_class = cls_logits_1.float().sigmoid().max(0).values.cpu()
    idx = per_class.topk(3).indices
    return ', '.join(f'{CLASS_NAMES[i]}={per_class[i]:.3f}' for i in idx)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='BEVFormer-Tiny + Qwen2.5VL-7B streaming BEV visualisation'
    )
    ap.add_argument('--dataroot',       default='/Users/trish/Downloads/nuScenes_miniV1.0')
    ap.add_argument('--scene',          type=int,   default=0)
    ap.add_argument('--max-frames',     type=int,   default=40)
    ap.add_argument('--score-thr',      type=float, default=0.25)
    ap.add_argument('--canvas',         type=int,   default=512)
    ap.add_argument('--range',          type=float, default=150.)
    ap.add_argument('--checkpoint',
                    default='model/checkpoints/bevformer_tiny_fp16_epoch_24.pth')
    ap.add_argument('--vl',             action='store_true',
                    help='Enable Qwen2.5VL-7B decisions (streaming)')
    ap.add_argument('--ollama-url',     default='http://localhost:11434')
    ap.add_argument('--ollama-model',   default='qwen2.5vl:7b')
    ap.add_argument('--ollama-timeout', type=int, default=90)
    ap.add_argument('--update-every',   type=int, default=6,
                    help='Tokens between composite image refreshes (default 6)')
    ap.add_argument('--log',            default=None)
    args = ap.parse_args()

    device = _get_device()
    print(f'[INFO] Device     : {device}')

    # ── Model ──────────────────────────────────────────────────────────────────
    model = BEVFormerTiny(pretrained_backbone=False)
    model.eval()
    if args.checkpoint:
        # Use the same full remap as infer.py / eval.py:
        # _build_remap covers backbone + neck + BEV encoder + decoder +
        # cls/reg branches + BEV queries + positional encodings.
        # load_official_weights only remaps backbone/neck and skips everything
        # else — leaving the entire detection model with random weights.
        ckpt  = torch.load(args.checkpoint, map_location='cpu')
        raw   = ckpt.get('state_dict', ckpt)
        remap = _build_remap(raw)
        result = model.load_state_dict(remap, strict=False)
        loaded = len(remap) - len(result.unexpected_keys)
        print(f'[INFO] Checkpoint : {args.checkpoint}')
        print(f'[INFO] Keys loaded: {loaded}/{len(remap)} remapped  '
              f'missing={len(result.missing_keys)}')
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[INFO] Parameters : {n_params:.1f} M')

    print('[INFO] Class groups:')
    shown = set()
    for _, group in zip(CLASS_NAMES, CLASS_GROUP):
        if group not in shown:
            shown.add(group)
            members = [n for n, g in zip(CLASS_NAMES, CLASS_GROUP) if g == group]
            print(f'         {group:<14} {str(GROUP_COLORS[group]):<20}'
                  f' ← {", ".join(members)}')

    # ── Data ───────────────────────────────────────────────────────────────────
    loader = NuScenesMiniLoader(args.dataroot)
    nusc   = loader.nusc
    scene  = nusc.scene[args.scene]
    loc    = nusc.get('log', scene['log_token'])['location']
    print(f'[INFO] nuScenes   : {len(loader)} samples / {len(nusc.scene)} scenes')
    print(f'[INFO] Scene {args.scene:2d}    : {scene["name"]}  ({loc})')

    nusc_map = _load_nusc_map(args.dataroot, loc)
    if nusc_map is not None:
        print(f'[INFO] Map        : loaded ({loc})')

    # LiDAR→ego calibration yaw — fixed per vehicle, needed for LiDAR→global transform.
    # In nuScenes the LIDAR_TOP is typically mounted with ~-90° yaw relative to ego
    # (LiDAR x = vehicle right, LiDAR y = vehicle forward).
    _first_sample = nusc.get('sample', scene['first_sample_token'])
    _lidar_sd     = nusc.get('sample_data', _first_sample['data']['LIDAR_TOP'])
    _lidar_cs     = nusc.get('calibrated_sensor', _lidar_sd['calibrated_sensor_token'])
    lidar2ego_yaw = Quaternion(_lidar_cs['rotation']).yaw_pitch_roll[0]
    print(f'[INFO] lidar2ego yaw: {math.degrees(lidar2ego_yaw):.2f}°')

    if args.vl:
        print(f'[INFO] VLM        : {args.ollama_model}  @  {args.ollama_url}')
        print(f'[INFO] Composite  : {LATEST_PATH}  (updated every {args.update_every} tokens)')

    OUT_DIR.mkdir(exist_ok=True)
    log_fh = open(args.log, 'a') if args.log else None

    prev_bev     = None
    frame_idx    = 0
    patch_origin = None
    ego_history: list = []   # [(tx, ty, yaw), ...] accumulated across frames

    print('─' * 72)

    with torch.no_grad():
        for sample in loader.iter_scene(scene_idx=args.scene):
            if frame_idx >= args.max_frames:
                break

            imgs, img_metas = sample['imgs'], sample['img_metas']

            # ── BEV inference ──────────────────────────────────────────────────
            t0  = time.perf_counter()
            out = model(imgs, img_metas, prev_bev=prev_bev)
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            bev_ms   = (time.perf_counter() - t0) * 1000
            prev_bev = out['bev_feat'].detach()

            sample_token = img_metas[0]['sample_token']
            ego_pose     = _get_ego_pose(nusc, sample_token)

            # Accumulate true ego position for the trajectory GT panel
            ego_tx  = float(ego_pose['translation'][0])
            ego_ty  = float(ego_pose['translation'][1])
            ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]
            ego_history.append((ego_tx, ego_ty, ego_yaw))

            if patch_origin is None:
                patch_origin = (ego_tx, ego_ty)
                print(f'[INFO] Patch origin ({patch_origin[0]:.1f}, '
                      f'{patch_origin[1]:.1f}) m  range {args.range} m')
                print('─' * 72)

            # ── Render prediction canvas ───────────────────────────────────────
            canvas = build_scene_canvas(
                out, ego_pose, nusc_map,
                patch_origin=patch_origin,
                patch_range=args.range,
                canvas_size=args.canvas,
                score_thr=args.score_thr,
                lidar2ego_yaw=lidar2ego_yaw,
            )
            out_path = str(OUT_DIR / f'vis_{frame_idx:03d}.png')
            cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

            # ── Render GT trajectory canvas ────────────────────────────────────
            trail_canvas = make_trajectory_canvas(
                ego_history, nusc_map, patch_origin,
                patch_range=args.range, canvas_size=args.canvas,
                nusc=nusc, sample_token=sample_token,
            )

            # ── 6-camera grid with predicted 3-D boxes ─────────────────────────
            # total_w = 2 × bev_panel + 2-px sep (must equal 3 × cell_w)
            bev_total_w = canvas.shape[0] * 2 + 2
            cam_grid = _make_cam_grid(
                nusc, sample_token, args.dataroot,
                out['cls_logits'][0].cpu(),
                out['reg_preds'][0].cpu(),
                out['ref_pts'][0].cpu(),
                args.score_thr,
                total_w=bev_total_w,
            )

            # ── Save composite ─────────────────────────────────────────────────
            # Layout: [pred | sep | traj] on top, [camera grid] below
            _save_bev_dual(canvas, trail_canvas, cam_grid=cam_grid)

            # ── VLM streaming ──────────────────────────────────────────────────
            reasoning = decision = ""
            vl_ms = 0.0
            if args.vl:
                def _on_token(text, _c=canvas, _t=trail_canvas, _g=cam_grid):
                    _save_bev_dual(_c, _t, vl_text=text, cam_grid=_g)

                t1 = time.monotonic()
                try:
                    raw = _query_ollama_streaming(
                        _encode_image(out_path),
                        args.ollama_model, args.ollama_url,
                        args.ollama_timeout, _on_token,
                        update_every=args.update_every,
                    )
                    reasoning, decision = _parse_response(raw)
                    _save_bev_dual(canvas, trail_canvas, vl_text=raw, cam_grid=cam_grid)
                except urllib.error.URLError as exc:
                    raw = f"[OLLAMA UNREACHABLE: {exc.reason}]"
                    _save_bev_dual(canvas, trail_canvas, vl_text=raw, cam_grid=cam_grid)
                except Exception as exc:
                    raw = f"[VLM ERROR: {exc}]"
                    _save_bev_dual(canvas, trail_canvas, vl_text=raw, cam_grid=cam_grid)
                vl_ms = (time.monotonic() - t1) * 1000

            # ── Console ────────────────────────────────────────────────────────
            print(f'  frame {frame_idx:3d} | bev {bev_ms:6.1f} ms'
                  + (f' | vl {vl_ms:5.0f} ms' if args.vl else '')
                  + f' | {_top3_str(out["cls_logits"][0])}'
                  + f' | {sample_token[:8]}')
            if reasoning:
                print(f'           reasoning : {reasoning}')
            if decision:
                label = _LABELS.get(decision, decision)
                print(f'           decision  : {label}')

            if log_fh and args.vl:
                log_fh.write(json.dumps({
                    "frame": frame_idx, "token": sample_token,
                    "decision": decision, "reasoning": reasoning,
                    "bev_ms": round(bev_ms, 1), "vl_ms": round(vl_ms, 1),
                }) + '\n')
                log_fh.flush()

            frame_idx += 1

    if log_fh:
        log_fh.close()
    print(f'\n[DONE] {frame_idx} frames → {OUT_DIR}/')


if __name__ == '__main__':
    main()
