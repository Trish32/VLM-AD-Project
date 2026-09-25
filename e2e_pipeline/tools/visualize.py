#!/usr/bin/env python3
"""Render a closed-loop rollout as an animated GIF.

Layout follows `diffusiondrive_planner/assets/demo_scene-0916.gif`: the camera
with projected geometry on the left, the occupancy branch on the right, and the
numbers underneath as discrete tiles rather than a wall of text.

What each panel is for:

  CAMERA   what the perception branch found, drawn where a human would look for
           it -- 3-D agent boxes and the chosen plan laid on the road surface.
           This is the panel that makes a bad detection obvious; a box floating
           off a car is visible here and invisible in a BEV raster.

  OCCUPANCY  the dense branch reduced to what the planner actually consumes:
           drivable / unknown / obstacle, plus every candidate trajectory
           coloured by the filter's verdict. A rollout that emergency-brakes
           every step and one that flows cleanly look completely different here.

  TILES    the five metric families, each with the number that would appear in
           a report and a one-word statement of what it measures.

A note on the camera, because it is the one thing in this picture that is not
literally true: the loop simulates ego motion, so the ego drifts away from the
logged trajectory, and nuScenes has no image from the simulated pose. The
overlay projects world-frame geometry through the *logged* camera -- exact
projection, logged viewpoint -- and prints the divergence between the two poses
on the panel. When it is small the overlay reads as a normal camera view; when
it grows, the printed number is the honest caveat.

Usage:
    conda run -n simple_bev_vldrive python -m e2e_pipeline.tools.visualize \\
        --scene 0 --steps 20 --out e2e_pipeline/assets/closed_loop_scene0.gif
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..closed_loop import ClosedLoopRunner, GTWorldModel, LoopConfig
from ..metrics.metrics import evaluate
from ..planner.safety_filter import SafetyFilter
from ..scene import ego_footprint_corners

# Palette shared with bevformer_vldrive/tools/visualizer.py.
BG = (11, 20, 20)
PANEL_BG = (15, 31, 31)
ROAD = (36, 102, 92)
UNKNOWN = (48, 54, 54)
OBSTACLE = (150, 52, 52)
EGO = (40, 90, 210)
EGO_EDGE = (240, 240, 240)
AGENT = (210, 80, 80)
FEASIBLE = (70, 200, 130)
REJECTED = (122, 62, 62)
CHOSEN = (250, 196, 60)
TEXT = (232, 232, 232)
DIM = (136, 152, 152)
RULE = (44, 66, 66)

CAM_W, CAM_H = 820, 461          # 1600x900 front camera, scaled
BEV = 461                        # occupancy panel, square
HEAD_H = 34
STATUS_H = 32
TILE_H = 152
W = CAM_W + BEV
H = HEAD_H + CAM_H + STATUS_H + TILE_H

RANGE_M = 40.0                   # half-extent of the occupancy view, metres

# Occ3D-nuScenes palette, byte-identical to FlashOcc's own `vis_occ.py` and to
# Occupancy/FlashOcc/tools/visualize_occ.py::PALETTE. Copied rather than
# imported because that module pulls in matplotlib, which this one does not need
# and which costs ~1 s of import per invocation.
OCC_PALETTE = [
    (0, 0, 0), (255, 120, 50), (255, 192, 203), (255, 255, 0),
    (0, 150, 245), (0, 255, 255), (200, 180, 0), (255, 0, 0),
    (255, 240, 150), (135, 60, 0), (160, 32, 240), (255, 0, 255),
    (139, 137, 137), (75, 0, 75), (150, 240, 80), (230, 230, 250),
    (0, 175, 0), (255, 255, 255),
]
OCC_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'constr.veh',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free',
]
FREE_ID = 17

#: Drawn in the legend, in this order, when present in the frame. The full 18
#: will not fit in a 461 px panel and most carry <0.1% of cells.
LEGEND_PRIORITY = [11, 13, 14, 16, 15, 4, 10, 3, 7, 2, 6, 1, 8, 9, 12, 0]


def _font(size=13, bold=False):
    names = (('/System/Library/Fonts/Menlo.ttc', 1 if bold else 0),
             ('/System/Library/Fonts/SFNSMono.ttf', 0))
    for path, idx in names:
        try:
            return ImageFont.truetype(path, size, index=idx)
        except OSError:
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Occupancy panel (ego frame, forward = up)
# ---------------------------------------------------------------------------


def _to_px(xy: np.ndarray) -> np.ndarray:
    """Ego-frame metres -> panel pixels, forward = up, left = left."""
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    s = BEV / (2 * RANGE_M)
    return np.stack([BEV / 2 - xy[:, 1] * s, BEV / 2 - xy[:, 0] * s], axis=1)


def _semantic_layer(fs) -> Image.Image | None:
    """Full Occ3D class map as an image, or None if the source has no semantics.

    WHY NOT THE THREE-COLOUR VERSION. Reducing to drivable / obstacle / unknown
    is what the PLANNER consumes, and showing only that makes the panel a
    picture of the reduction rather than of the occupancy. A sidewalk, a
    vegetation bank and a parked truck are all "obstacle" to the filter and look
    identical, so a viewer cannot tell a mis-segmentation from a real hazard,
    and cannot see that the road surface is the only class the traversable mask
    keeps out of six ground-ish ones.

    Rendered by resampling the (nx, ny) label grid with NEAREST, never a smooth
    filter: interpolating class ids invents classes, and the halfway point
    between `car` (4) and `construction_vehicle` (5) is not a meaningful label.
    """
    sem = getattr(fs, 'semantics', None)
    if sem is None:
        return None
    sem = np.asarray(sem)
    lut = np.array(OCC_PALETTE, dtype=np.uint8)
    rgb = lut[np.clip(sem, 0, len(OCC_PALETTE) - 1)]      # (nx, ny, 3)
    # ego frame is +x forward / +y left; the panel is forward-up, left-left, so
    # transpose to (row=y, col=x) then flip both axes.
    rgb = np.transpose(rgb, (1, 0, 2))[::-1, ::-1]
    return Image.fromarray(rgb, 'RGB').resize((BEV, BEV), Image.NEAREST)


def _draw_freespace(draw: ImageDraw.ImageDraw, fs) -> None:
    """Drivable / unknown as coarse tiles — full-resolution cells are invisible.

    The fallback for sources with no semantics (the synthetic corridor).
    """
    step = 4
    res, ox, oy = fs.res, fs.origin[0], fs.origin[1]
    nx, ny = fs.traversable.shape
    for ix in range(0, nx, step):
        for iy in range(0, ny, step):
            if fs.obstacle[ix:ix + step, iy:iy + step].any():
                c = OBSTACLE
            elif fs.unknown[ix:ix + step, iy:iy + step].all():
                c = UNKNOWN
            elif fs.traversable[ix:ix + step, iy:iy + step].any():
                c = ROAD
            else:
                continue
            p0 = _to_px([[ox + ix * res, oy + iy * res]])[0]
            p1 = _to_px([[ox + (ix + step) * res, oy + (iy + step) * res]])[0]
            draw.rectangle([min(p0[0], p1[0]), min(p0[1], p1[1]),
                            max(p0[0], p1[0]), max(p0[1], p1[1])], fill=c)


def _draw_occ_legend(d, fs, f) -> None:
    """Colour key for the classes actually present, commonest first."""
    sem = getattr(fs, 'semantics', None)
    if sem is None:
        d.text((10, BEV - 42), 'teal drivable   grey unknown   red obstacle',
               font=f, fill=(86, 112, 112))
        return
    sem = np.asarray(sem)
    present = [(c, float((sem == c).mean())) for c in LEGEND_PRIORITY]
    present = [(c, s) for c, s in present if s > 0.004][:8]
    rows, row, x = [], [], 0
    for c, share in present:
        w = 16 + int(d.textlength(OCC_NAMES[c], font=f)) + 8
        if x + w > BEV - 16 and row:
            rows.append(row)
            row, x = [], 0
        row.append(c)
        x += w
    if row:
        rows.append(row)

    # Dark backing: the palette contains pure white and pure yellow, so legend
    # text drawn straight onto it is unreadable in exactly the frames where the
    # legend matters most.
    top = BEV - 20 - 15 * len(rows)
    d.rectangle([0, top - 6, BEV, BEV], fill=(12, 20, 20))
    for r, cs in enumerate(rows):
        x = 10
        y = top + 15 * r
        for c in cs:
            d.rectangle([x, y + 2, x + 9, y + 11], fill=OCC_PALETTE[c],
                        outline=(60, 76, 76))
            d.text((x + 13, y), OCC_NAMES[c], font=f, fill=(168, 182, 182))
            x += 16 + int(d.textlength(OCC_NAMES[c], font=f)) + 8


def _poly(draw, pts_m, colour, width=2, closed=True, casing=True):
    """Polyline in ego metres, with a dark casing so it reads on any background.

    The semantic panel contains saturated green, magenta and white, so a plain
    green "feasible" line is invisible over vegetation and a white one over
    free space. A 2 px darker underlay costs nothing and makes every overlay
    legible regardless of what class is beneath it.
    """
    seq = [tuple(p) for p in _to_px(pts_m)]
    if closed:
        seq.append(seq[0])
    if casing:
        draw.line(seq, fill=(10, 16, 16), width=width + 3)
    draw.line(seq, fill=colour, width=width)


def _render_bev(scene, result) -> Image.Image:
    img = Image.new('RGB', (BEV, BEV), PANEL_BG)
    layer = _semantic_layer(scene.freespace)
    if layer is not None:
        # dimmed, so the overlaid trajectories and boxes stay readable against
        # a palette that includes pure white (free) and pure yellow (bicycle)
        img = Image.blend(img, layer, 0.72)
    d = ImageDraw.Draw(img)
    f = _font(11)

    if layer is None:
        _draw_freespace(d, scene.freespace)

    origin = _to_px([[0, 0]])[0]
    for r in (10, 20, 30):
        rr = r * BEV / (2 * RANGE_M)
        d.ellipse([origin[0] - rr, origin[1] - rr,
                   origin[0] + rr, origin[1] + rr], outline=(46, 72, 72))
        d.text((origin[0] + 4, origin[1] - rr - 12), f'{r}m',
               font=f, fill=(74, 100, 100))

    for a in scene.agents:
        _poly(d, ego_footprint_corners(a.xy, a.yaw, a.lwh[0], a.lwh[1]),
              AGENT, width=2)

    for i, v in enumerate(result.verdicts):
        _poly(d, np.vstack([[0.0, 0.0], result.candidates[i]]),
              FEASIBLE if v.feasible else REJECTED, width=2, closed=False)

    if not result.emergency:
        _poly(d, np.vstack([[0.0, 0.0], result.trajectory]),
              CHOSEN, width=4, closed=False)
        for p in _to_px(result.trajectory):
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=CHOSEN)

    ego_poly = ego_footprint_corners(np.zeros(2), 0.0,
                                     scene.ego.length, scene.ego.width)
    d.polygon([tuple(p) for p in _to_px(ego_poly)], fill=EGO, outline=EGO_EDGE)

    d.text((10, 8), 'FLASHOCC SEMANTICS  +  CANDIDATES', font=_font(12, bold=True),
           fill=TEXT)
    # legend first: it lays down a dark backing strip that would otherwise
    # cover the candidate-colour line drawn under it
    _draw_occ_legend(d, scene.freespace, f)
    d.text((10, BEV - 16), 'green feasible   dark-red rejected   amber chosen',
           font=f, fill=(110, 134, 134))
    return img


# ---------------------------------------------------------------------------
# Camera panel (world frame projected through the logged camera)
# ---------------------------------------------------------------------------


def _project(P: np.ndarray, pts_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World (N,3) -> pixel (N,2) in full-resolution camera coords, + in-front mask."""
    pts = np.asarray(pts_w, dtype=np.float64).reshape(-1, 3)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = (P @ hom.T).T
    z = cam[:, 2]
    ok = z > 0.5
    uv = np.zeros((len(pts), 2))
    safe = np.where(ok, z, 1.0)
    uv[:, 0] = cam[:, 0] / safe
    uv[:, 1] = cam[:, 1] / safe
    return uv, ok


def _box_corners_3d(xy_w, yaw_w, l, w, h) -> np.ndarray:
    """Eight corners, world frame, box resting on z = 0 (ego-frame ground)."""
    x = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * h
    c, s = np.cos(yaw_w), np.sin(yaw_w)
    return np.stack([xy_w[0] + c * x - s * y, xy_w[1] + s * x + c * y, z], axis=1)


_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
          (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7))


def _render_camera(cam_info, rec, scene, result) -> Image.Image:
    img = Image.open(cam_info['path']).convert('RGB')
    full_w = img.width
    img = img.resize((CAM_W, CAM_H), Image.BILINEAR)
    over = Image.new('RGBA', (CAM_W, CAM_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    P, k = cam_info['P'], CAM_W / full_w

    c, s = np.cos(rec.ego_yaw), np.sin(rec.ego_yaw)
    R = np.array([[c, -s], [s, c]])           # sim-ego -> world

    def to_world(xy_ego):
        return (R @ np.asarray(xy_ego, dtype=np.float64).reshape(-1, 2).T).T + rec.ego_xy

    # --- planned path, as a ribbon on the road surface ---------------------
    if not result.emergency and len(result.trajectory):
        path = to_world(np.vstack([[0.0, 0.0], result.trajectory]))
        half = scene.ego.width / 2
        for i in range(len(path) - 1):
            seg = path[i + 1] - path[i]
            n = np.linalg.norm(seg)
            if n < 1e-6:
                continue
            perp = np.array([-seg[1], seg[0]]) / n * half
            quad = np.array([path[i] + perp, path[i + 1] + perp,
                             path[i + 1] - perp, path[i] - perp])
            uv, ok = _project(P, np.column_stack([quad, np.zeros(4)]))
            if not ok.all():
                continue
            fade = int(150 * (1 - i / max(1, len(path) - 1)) + 45)
            od.polygon([tuple(p * k) for p in uv],
                       fill=(*CHOSEN, fade), outline=(*CHOSEN, 210))

    # --- agent boxes -------------------------------------------------------
    for a in scene.agents:
        if np.linalg.norm(a.xy) > 45.0 or a.xy[0] < 0.5:
            continue
        xy_w = to_world(a.xy)[0]
        corners = _box_corners_3d(xy_w, a.yaw + rec.ego_yaw,
                                  a.lwh[0], a.lwh[1], a.lwh[2])
        uv, ok = _project(P, corners)
        if not ok.all():
            continue
        uv = uv * k
        if uv[:, 0].max() < -50 or uv[:, 0].min() > CAM_W + 50:
            continue
        for i, j in _EDGES:
            od.line([tuple(uv[i]), tuple(uv[j])], fill=(*AGENT, 235), width=2)
        # Front face filled, so heading is readable at a glance.
        od.polygon([tuple(uv[i]) for i in (0, 1, 5, 4)], fill=(*AGENT, 46))

    img = Image.alpha_composite(img.convert('RGBA'), over).convert('RGB')
    d = ImageDraw.Draw(img)
    div = float(np.linalg.norm(rec.ego_xy - cam_info['ego_xy']))
    d.rectangle([0, 0, CAM_W, 22], fill=(11, 20, 20))
    d.text((10, 5), 'CAM_FRONT   3-D detections + planned path',
           font=_font(12, bold=True), fill=TEXT)
    txt = f'logged viewpoint · sim ego {div:.1f} m away'
    d.text((CAM_W - 14 - d.textlength(txt, font=_font(11)), 5), txt,
           font=_font(11), fill=(168, 168, 116) if div > 3 else DIM)
    return img


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def _tile(d, x, y, w, label, value, unit, sub, colour):
    d.rectangle([x, y, x + w, y + TILE_H - 20], fill=PANEL_BG, outline=RULE)
    d.text((x + 12, y + 10), label, font=_font(11, bold=True), fill=DIM)
    d.text((x + 12, y + 34), value, font=_font(34, bold=True), fill=colour)
    if unit:
        d.text((x + 16 + d.textlength(value, font=_font(34, bold=True)), y + 54),
               unit, font=_font(12), fill=DIM)
    for i, ln in enumerate(sub.split('\n')[:2]):
        d.text((x + 12, y + 84 + i * 15), ln, font=_font(11), fill=(104, 126, 126))


def render_frame(rec, scene, result, cam_info, step: int, n_steps: int,
                 metrics: dict, scene_name: str, intent_line: str = ''
                 ) -> Image.Image:
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)

    # --- header ------------------------------------------------------------
    d.text((14, 9), 'E2E PIPELINE  ·  closed loop', font=_font(14, bold=True),
           fill=TEXT)
    d.text((250, 11), 'perception → occupancy → prediction → planning → '
                      'safety filter → control', font=_font(11), fill=DIM)
    head = f'{scene_name}   step {step + 1}/{n_steps}   t={rec.t:.1f}s'
    d.text((W - 14 - d.textlength(head, font=_font(11)), 11), head,
           font=_font(11), fill=DIM)

    y = HEAD_H
    if cam_info is not None:
        img.paste(_render_camera(cam_info, rec, scene, result), (0, y))
    img.paste(_render_bev(scene, result), (CAM_W, y))
    d.line([CAM_W, y, CAM_W, y + CAM_H], fill=RULE)

    # --- status strip ------------------------------------------------------
    y += CAM_H
    d.rectangle([0, y, W, y + STATUS_H], fill=(18, 30, 30))
    status = ('EMERGENCY BRAKE' if result.emergency
              else f'chose candidate {result.chosen_index}')
    scol = (235, 92, 92) if result.emergency else FEASIBLE
    parts = [(f'speed {rec.ego_v:5.2f} m/s', TEXT),
             (f'accel {rec.accel:+5.2f} m/s²', DIM),
             (f'steer {rec.steer:+.3f} rad', DIM),
             (f'{result.feasible_count}/{len(result.verdicts)} candidates '
              f'feasible', DIM),
             (status, scol)]
    x = 14
    for txt, col in parts:
        d.text((x, y + 9), txt, font=_font(12, bold=col is scol), fill=col)
        x += int(d.textlength(txt, font=_font(12))) + 30
    if intent_line:
        d.text((x, y + 9), f'intent: {intent_line[:40]}', font=_font(12),
               fill=(150, 190, 190))

    # --- metric tiles ------------------------------------------------------
    y += STATUS_H + 10
    s, rt, c, lat = (metrics['safety'], metrics['route'],
                     metrics['comfort'], metrics.get('latency', {}))
    mc = s['min_clearance_m']
    tiles = [
        ('SAFETY', str(s['n_collision_steps']), 'collisions',
         'steps with any\nfootprint overlap',
         FEASIBLE if s['n_collision_steps'] == 0 else (235, 92, 92)),
        ('CLEARANCE', f'{mc:.2f}' if mc is not None else '—', 'm',
         'closest approach\nto any agent',
         FEASIBLE if (mc is None or mc > 0.5) else (235, 176, 92)),
        ('ROUTE', f"{rt['completion']:.0%}" if rt.get('completion') is not None
         else '—', '', 'of the logged\nroute covered', (110, 180, 240)),
        ('COMFORT', f"{c['jerk_rms']:.2f}" if not c.get('insufficient_data')
         else '—', 'm/s³', 'RMS jerk over\nthe rollout', (190, 160, 240)),
        ('INTERVENTIONS', str(s['emergency_brakes']), 'brakes',
         'filter rejected every\ncandidate',
         FEASIBLE if s['emergency_brakes'] == 0 else (235, 176, 92)),
    ]
    if lat.get('total_ms_mean') is not None:
        tiles.append(('LATENCY', f"{lat['total_ms_mean']:.0f}", 'ms',
                      'per closed-loop\nstep, mean', (140, 200, 200)))

    gap, n = 10, len(tiles)
    tw = (W - 28 - gap * (n - 1)) / n
    for i, (label, value, unit, sub, col) in enumerate(tiles):
        _tile(d, 14 + i * (tw + gap), y, tw, label, value, unit, sub, col)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot',
                    default=os.environ.get('NUSCENES_DATAROOT') or
                    os.path.expanduser('~/Downloads/nuScenes_miniV1.0'))
    ap.add_argument('--scene', type=int, default=5)
    ap.add_argument('--steps', type=int, default=24)
    ap.add_argument('--initial-speed', type=float, default=None,
                    help='m/s; default is the logged ego speed at frame 0')
    ap.add_argument('--anchors',
                    default='diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy')
    ap.add_argument('--frame-ms', type=int, default=500)
    ap.add_argument('--width', type=int, default=1040,
                    help='output width; the camera photo dominates GIF size')
    ap.add_argument('--colors', type=int, default=128)
    ap.add_argument('--no-camera', action='store_true',
                    help='skip the camera panel (occupancy + tiles only)')
    ap.add_argument('--occupancy', choices=('flashocc', 'corridor'),
                    default='flashocc',
                    help='flashocc shows all 18 Occ3D classes; corridor is the '
                         'synthetic band, which has no semantics to show')
    ap.add_argument('--occlusion', choices=('temporal', 'raycast', 'none'),
                    default='temporal')
    ap.add_argument('--command', type=int, default=None,
                    help='fixed drive command; default derives it per step from '
                         'the route, which is what the pipeline now does')
    ap.add_argument('--out', default='e2e_pipeline/assets/closed_loop.gif')
    args = ap.parse_args()

    from nuscenes import NuScenes

    from ..closed_loop import constant_velocity_planner, diffusiondrive_anchor_planner

    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)
    if args.occupancy == 'flashocc':
        # Real FlashOcc output, so the panel can show all 18 Occ3D classes. The
        # synthetic corridor has no semantics to show -- it is a band drawn
        # around the logged route, which is why the old panel could only ever
        # display the three-way reduction.
        from ..closed_loop import FlashOccWorldModel
        world = FlashOccWorldModel(nusc, scene_idx=args.scene,
                                   occlusion=args.occlusion)
    else:
        world = GTWorldModel(nusc, scene_idx=args.scene)
    v0 = world.initial_speed() if args.initial_speed is None else args.initial_speed
    cfg = LoopConfig(max_steps=args.steps, initial_speed=v0)

    planner = (diffusiondrive_anchor_planner(args.anchors, cfg.dt)
               if Path(args.anchors).exists()
               else constant_velocity_planner(cfg.horizon, cfg.dt))

    # Re-run the loop capturing the per-step scene and filter result, which the
    # runner does not retain (records hold metrics inputs, not renderables).
    captured = []
    orig_call = SafetyFilter.__call__

    def capturing(self, candidates, scene, scores=None, risk=None):
        res = orig_call(self, candidates, scene, scores, risk)
        res.candidates = np.asarray(candidates)
        captured.append((scene, res))
        return res

    SafetyFilter.__call__ = capturing
    try:
        # `command=None` derives the drive command per step from the route.
        # Pinning it to 2 ('straight'), which this used to do implicitly, plans
        # straight through every turn -- on scene-0796 that is most of the run.
        records, metrics = ClosedLoopRunner(world, planner, cfg).run(
            command=args.command)
    finally:
        SafetyFilter.__call__ = orig_call

    name = nusc.scene[args.scene]['name']
    print(f'[INFO] {name}: {len(records)} steps, {len(captured)} captured')

    frames = []
    for k, rec in enumerate(records):
        if k >= len(captured):
            break
        scene, result = captured[k]
        cam = None if args.no_camera else world.camera_at(rec.t)
        frames.append(render_frame(rec, scene, result, cam, k, len(records),
                                   evaluate(records[:k + 1], world.route(), cfg.dt),
                                   name))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.width and args.width < W:
        h = int(round(H * args.width / W))
        frames = [f.resize((args.width, h), Image.LANCZOS) for f in frames]
    pf = [f.quantize(colors=args.colors, method=Image.MEDIANCUT,
                     dither=Image.NONE) for f in frames]
    pf[0].save(out, save_all=True, append_images=pf[1:],
               duration=args.frame_ms, loop=0, disposal=2)
    print(f'[DONE] {len(frames)} frames {frames[0].size} -> {out}  '
          f'({out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
