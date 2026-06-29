"""Shared BEVFusion visualisation: 6 surround cameras (with projected 3-D boxes)
+ a LiDAR-frame BEV (points + boxes). Used by both bevfusion_vl and
BEVFusion_robust_vl visualize.py drivers.

Boxes are lidar-frame [x, y, z_bottom, w, l, h, yaw, vx, vy]; projection uses the
ORIGINAL nuScenes camera intrinsics so boxes overlay the raw full-res images.
"""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

# camera grid order (display): front row, back row
CAM_GRID = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

# per-class RGB (10 nuScenes classes), vehicles warm, VRUs cool, static teal/grey
CLASS_RGB = [
    (255,  60,  60), (255, 140,  40), (200, 120,  40), (255, 200,  40),
    (180,  80, 200), ( 40, 180, 255), ( 60, 220, 220), ( 60, 220,  90),
    (240, 240,  60), (160, 160, 170),
]


def vis_lidar2img(nusc, sample_token, cam_name):
    """4x4 lidar→pixel with ORIGINAL (full-res) intrinsics."""
    sample = nusc.get('sample', sample_token)
    cam_sd = nusc.get('sample_data', sample['data'][cam_name])
    lid_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    def s2g(sd):
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ep = nusc.get('ego_pose', sd['ego_pose_token'])
        s2e = np.eye(4); s2e[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
        s2e[:3, 3] = cs['translation']
        e2g = np.eye(4); e2g[:3, :3] = Quaternion(ep['rotation']).rotation_matrix
        e2g[:3, 3] = ep['translation']
        return e2g @ s2e, cs

    cam2global, cam_cs = s2g(cam_sd)
    lid2global, _ = s2g(lid_sd)
    K = np.eye(4); K[:3, :3] = np.array(cam_cs['camera_intrinsic'])
    lidar2cam = np.linalg.inv(cam2global) @ lid2global
    return (K @ lidar2cam).astype(np.float64)


def _corners_3d(b):
    """8 corners (lidar frame) of box [x,y,z_bottom,w,l,h,yaw].

    The detector's yaw maps to the lidar-frame heading as ``-yaw - pi/2`` (the same
    convention the nuScenes eval uses), with box length l along that heading and
    width w perpendicular; z is the box bottom so corners span [z, z+h].
    """
    x, y, z, w, l, h, yaw = b[:7]
    head = -float(yaw) - math.pi / 2
    c, s = math.cos(head), math.sin(head)
    # length l along heading, width w perpendicular
    xs = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    ys = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    zs = [0, 0, 0, 0, h, h, h, h]
    out = np.zeros((8, 3))
    for i in range(8):
        out[i, 0] = x + c * xs[i] - s * ys[i]
        out[i, 1] = y + s * xs[i] + c * ys[i]
        out[i, 2] = z + zs[i]
    return out


_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def _draw_boxes_cam(img, lidar2img, boxes, labels):
    H, W = img.shape[:2]
    for b, lab in zip(boxes, labels):
        corners = np.concatenate([_corners_3d(b), np.ones((8, 1))], 1)
        proj = (lidar2img @ corners.T).T               # (8,4)
        depth = proj[:, 2]
        if (depth > 0.1).sum() < 8:                    # require fully in front
            continue
        uv = proj[:, :2] / depth[:, None]
        if (uv[:, 0].max() < 0 or uv[:, 0].min() > W or
                uv[:, 1].max() < 0 or uv[:, 1].min() > H):
            continue
        rgb = CLASS_RGB[int(lab) % len(CLASS_RGB)]
        col = (rgb[2], rgb[1], rgb[0])
        for a, bb in _EDGES:
            cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[bb].astype(int)), col, 2, cv2.LINE_AA)
    return img


def camera_grid(nusc, sample_token, dataroot, boxes, labels, cell=(480, 270)):
    sample = nusc.get('sample', sample_token)
    cells = []
    for cam in CAM_GRID:
        sd = nusc.get('sample_data', sample['data'][cam])
        img = cv2.imread(str(Path(dataroot) / sd['filename']))
        if img is None:
            img = np.zeros((900, 1600, 3), np.uint8)
        img = _draw_boxes_cam(img, vis_lidar2img(nusc, sample_token, cam), boxes, labels)
        lab = cam.replace('CAM_', '').replace('_', ' ')
        cv2.putText(img, lab, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, lab, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235), 2, cv2.LINE_AA)
        cell_img = cv2.resize(img, cell, interpolation=cv2.INTER_AREA)
        cells.append(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
    row0 = np.hstack(cells[:3]); row1 = np.hstack(cells[3:])
    return np.vstack([row0, row1])


def bev_panel(points, boxes, labels, pc_range, size):
    """Top-down LiDAR BEV (size×size RGB): points (height-shaded) + boxes, ego centre."""
    lo, hi = pc_range[0], pc_range[3]
    rng = hi - lo
    img = np.full((size, size, 3), 14, np.uint8)

    def to_px(x, y):
        # nuScenes LiDAR frame: +y = forward → up, +x = right → right.
        col = ((x - lo) / rng * size)
        row = ((hi - y) / rng * size)
        return col, row

    if points is not None and len(points):
        p = points.detach().cpu().numpy() if hasattr(points, 'detach') else np.asarray(points)
        xs, ys, zs = p[:, 0], p[:, 1], p[:, 2]
        m = (xs > lo) & (xs < hi) & (ys > lo) & (ys < hi)
        xs, ys, zs = xs[m], ys[m], zs[m]
        col = ((xs - lo) / rng * size).astype(int).clip(0, size - 1)
        row = ((hi - ys) / rng * size).astype(int).clip(0, size - 1)
        t = ((zs + 5) / 8).clip(0, 1)
        shade = (60 + t * 150).astype(np.uint8)
        img[row, col] = np.stack([shade, shade, (shade * 0.7).astype(np.uint8)], 1)

    for b, lab in zip(boxes, labels):
        corners = _corners_3d(b)[:4, :2]               # bottom rectangle
        pts = np.array([to_px(cx, cy) for cx, cy in corners], np.int32)
        rgb = CLASS_RGB[int(lab) % len(CLASS_RGB)]
        cv2.polylines(img, [pts], True, (rgb[2], rgb[1], rgb[0]), 2, cv2.LINE_AA)
        # heading tick (front edge midpoint)
        f = ((pts[0] + pts[1]) / 2).astype(int)
        cv2.line(img, tuple(pts.mean(0).astype(int)), tuple(f), (rgb[2], rgb[1], rgb[0]), 2, cv2.LINE_AA)
    c = size // 2
    cv2.drawMarker(img, (c, c), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 12, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def composite(nusc, sample_token, dataroot, points, boxes, scores, labels,
              pc_range, classes, title, score_thr=0.3):
    """Full frame: camera grid (top) + BEV (bottom-left) + legend."""
    import numpy as np
    boxes = np.asarray(boxes.detach().cpu() if hasattr(boxes, 'detach') else boxes)
    scores = np.asarray(scores.detach().cpu() if hasattr(scores, 'detach') else scores)
    labels = np.asarray(labels.detach().cpu() if hasattr(labels, 'detach') else labels)
    keep = scores >= score_thr
    boxes, labels = boxes[keep], labels[keep]

    grid = camera_grid(nusc, sample_token, dataroot, boxes, labels)   # (2*270, 3*480, 3)
    gh, gw = grid.shape[:2]
    bev = bev_panel(points, boxes, labels, pc_range, size=gh)         # gh×gh

    pad = np.full((gh, 14, 3), 14, np.uint8)
    comp = np.concatenate([grid, pad, bev], axis=1)
    return comp


def make_gif(frames, gif_path, duration_ms=500, max_height=360):
    imgs = [Image.fromarray(f) for f in frames]
    if max_height:
        imgs = [im.resize((round(im.width * max_height / im.height), max_height))
                if im.height > max_height else im for im in imgs]
    p = [im.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE) for im in imgs]
    p[0].save(gif_path, save_all=True, append_images=p[1:], duration=duration_ms,
              loop=0, disposal=2)
