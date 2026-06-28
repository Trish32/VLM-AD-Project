#!/usr/bin/env python3
"""
Tracking-task visualisation for Sparse4D-v3 (detection-AND-tracking).

Runs the model over one nuScenes scene and renders a top-down BEV per frame in
which every agent box is coloured by its **persistent track id** (assigned by the
InstanceBank's temporal propagation, not a separate matcher). Because the colour
follows the id, an object keeps the same colour across the whole clip, and a
short fading trail of its past centres makes the tracking obvious. Frames are
stitched into a GIF.

The BEV is drawn in the LIDAR_TOP frame plotted forward-up: lidar +x = vehicle
right → screen right, lidar +y = forward → screen up (no transform needed). Box
slot 3 is the length (extent along heading) and slot 4 the width, per the
Sparse4D anchor convention, so the box long axis aligns with the heading.

Usage (from the project root):
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
        python sparse4d_vl/tools/visualize_track.py --scene 0 --max-frames 20
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sparse4d_vl.data.loader import NuScenesSparse4DLoader            # noqa: E402
from sparse4d_vl.model.sparse4d_v3 import Sparse4Dv3                  # noqa: E402
from sparse4d_vl.model.checkpoint import load_checkpoint             # noqa: E402
from sparse4d_vl.tools.visualizer import _draw_box_on_image, CAM_NAMES  # noqa: E402

_PALETTE = plt.colormaps.get_cmap("tab20")


def _color(tid: int):
    return _PALETTE((tid % 20) / 20.0)


def _corners(cx, cy, yaw, length, width):
    """Four corners of an oriented box (length along heading)."""
    c, s = np.cos(yaw), np.sin(yaw)
    hx, hy = length / 2.0, width / 2.0
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    return local @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])


def render(out_path, boxes, tids, trails, frame_idx, scene, lim=50.0):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_facecolor("#0e0e12"); ax.grid(True, color="#2a2a33", lw=0.4)

    # past trails (drawn under the boxes), colour = track id
    for tid, pts in trails.items():
        if len(pts) > 1:
            p = np.asarray(pts)
            ax.plot(p[:, 0], p[:, 1], "-", color=_color(tid), lw=1.2, alpha=0.6)

    # current boxes — slot 3 = length (along yaw), slot 4 = width. The colour
    # (track id) plus the trail already convey identity, so no per-box text.
    for b, tid in zip(boxes, tids):
        col = _color(int(tid))
        ax.add_patch(Polygon(_corners(b[0], b[1], b[6], b[3], b[4]), closed=True,
                             facecolor=col, edgecolor="white", alpha=0.7, lw=0.9))

    # ego at the lidar origin, heading up
    ax.add_patch(Polygon(_corners(0, 0, np.pi / 2, 4.08, 1.85), closed=True,
                         fill=False, edgecolor="#9aa0a6", ls="--", lw=1.2))
    ax.text(-lim + 1.5, lim - 2.5,
            f"scene {scene}  frame {frame_idx:02d}  tracks {len(boxes)}",
            color="white", fontsize=9, va="top", family="monospace")
    ax.set_title("Sparse4D-v3 detection + tracking (colour = track id)",
                 color="white", fontsize=10)
    ax.set_xlabel("← left   (m)   right →", color="#9aa0a6", fontsize=8)
    ax.set_ylabel("← back   (m)   forward →", color="#9aa0a6", fontsize=8)
    ax.tick_params(colors="#9aa0a6", labelsize=7)
    fig.tight_layout(); fig.savefig(out_path, facecolor="#0e0e12"); plt.close(fig)


def _bgr(tid: int):
    r = _color(tid)
    return (int(r[2] * 255), int(r[1] * 255), int(r[0] * 255))


def render_cameras(imgs_np, projection, boxes_cam, tids, scores):
    """2×3 surround-camera grid with 3-D boxes coloured by track id (BGR array).

    `boxes_cam` are the LIDAR-frame boxes with slots 3/4 swapped so the projection
    (`projection_mat` = lidar→pixel) draws the box length along the heading.
    """
    cam_imgs = []
    for ci in range(imgs_np.shape[0]):
        img = imgs_np[ci].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        proj = projection[ci]
        for k in range(len(boxes_cam)):
            img = _draw_box_on_image(img, boxes_cam[k], proj, color=_bgr(int(tids[k])),
                                     label=str(int(tids[k])), score=float(scores[k]))
        cv2.putText(img, CAM_NAMES[ci].replace("CAM_", ""), (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cam_imgs.append(img)
    order = [2, 0, 1, 4, 3, 5]   # FL,F,FR / BL,B,BR  (matches tools/visualizer)
    row0 = np.hstack([cam_imgs[i] for i in order[:3]])
    row1 = np.hstack([cam_imgs[i] for i in order[3:]])
    return np.vstack([row0, row1])


def combine(cam_bgr, bev_path, out_path, height=420):
    """SparseDrive-style composite: surround cameras (left) | BEV (right)."""
    cam = Image.fromarray(cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB))
    bev = Image.open(bev_path).convert("RGB")
    cam = cam.resize((max(1, round(cam.width * height / cam.height)), height))
    bev = bev.resize((height, height))
    combo = Image.new("RGB", (cam.width + bev.width, height), "#0e0e12")
    combo.paste(cam, (0, 0))
    combo.paste(bev, (cam.width, 0))
    combo.save(out_path)


def make_gif(paths, gif_path, duration_ms=400, max_height=260):
    frames = [Image.open(p).convert("RGB") for p in paths]
    if max_height:
        frames = [f.resize((round(f.width * max_height / f.height), max_height))
                  if f.height > max_height else f for f in frames]
    pframes = [f.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)
               for f in frames]
    pframes[0].save(gif_path, save_all=True, append_images=pframes[1:],
                    duration=duration_ms, loop=0, disposal=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth")
    ap.add_argument("--dataroot", default="/Users/trish/Downloads/nuScenes_miniV1.0")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--out", default="sparse4d_track_outputs")
    args = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    loader = NuScenesSparse4DLoader(args.dataroot, version="v1.0-mini")
    model = Sparse4Dv3(pretrained_backbone=False).to(dev)
    model.eval()
    load_checkpoint(model, args.checkpoint, version="v3")
    model.reset_state()                                  # new scene → reset cache + ids
    print(f"[track-viz] scene {args.scene}  device {dev}")

    outdir = Path(args.out); outdir.mkdir(exist_ok=True)
    for pat in ("track_*.png", "bev_*.png", "combo_*.png"):
        for old in outdir.glob(pat):
            old.unlink()
    trails = defaultdict(lambda: deque(maxlen=12))
    paths = []
    with torch.no_grad():
        for i, frame in enumerate(loader.iter_scene(args.scene)):
            if i >= args.max_frames:
                break
            metas = frame["img_metas"]
            out = model(frame["imgs"].float(), metas)
            det = out["detections"][0]
            boxes  = det["boxes_3d"].cpu().numpy()
            scores = det["scores_3d"].cpu().numpy()
            tids   = det.get("track_ids")
            tids   = tids.cpu().numpy() if tids is not None else np.full(len(boxes), -1)
            keep = (scores >= args.score_thresh) & (tids >= 0)
            boxes, scores, tids = boxes[keep], scores[keep], tids[keep]
            for b, t in zip(boxes, tids):
                trails[int(t)].append((float(b[0]), float(b[1])))

            # BEV panel (track-coloured boxes + trails)
            bev_p = str(outdir / f"bev_{i:03d}.png")
            render(bev_p, boxes, tids, trails, i, args.scene)

            # Surround-camera panel: swap slots 3/4 so the projection draws length
            # along heading, colour each box by its track id.
            cam_boxes = boxes.copy()
            if len(cam_boxes):
                cam_boxes[:, [3, 4]] = boxes[:, [4, 3]]
            cam_grid = render_cameras(frame["imgs"][0].numpy(),
                                      metas["projection_mat"], cam_boxes, tids, scores)
            combo_p = str(outdir / f"combo_{i:03d}.png")
            combine(cam_grid, bev_p, combo_p)
            paths.append(combo_p)
            print(f"  frame {i:02d}: {len(boxes):3d} tracked (>= {args.score_thresh})")

    gif = str(outdir / f"scene{args.scene}_track.gif")
    make_gif(paths, gif)
    print(f"[track-viz] wrote {gif}  ({len(paths)} frames)")


if __name__ == "__main__":
    main()
