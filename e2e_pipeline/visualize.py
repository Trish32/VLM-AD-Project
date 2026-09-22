#!/usr/bin/env python3
"""Render a closed-loop rollout as an animated GIF.

Shows what is otherwise only visible in log lines: which candidate trajectories
the planner proposed, which gates rejected which of them and why, what the
occupancy branch considered drivable, and where the ego actually went as a
result. A rollout that emergency-brakes every step and one that flows cleanly
produce very different pictures, and the difference is the point.

Colour follows the rest of the repo (visualizer.py): dark teal background, teal
drivable surface, blue ego.

Usage:
    conda run -n simple_bev_vldrive python -m e2e_pipeline.visualize \\
        --scene 0 --steps 20 --out e2e_pipeline/assets/closed_loop_scene0.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .closed_loop import ClosedLoopRunner, GTWorldModel, LoopConfig
from .metrics import evaluate
from .safety_filter import SafetyFilter
from .scene import ego_footprint_corners

BG = (15, 31, 31)
ROAD = (36, 102, 92)
UNKNOWN = (52, 58, 58)
OBSTACLE = (150, 52, 52)
EGO = (40, 90, 210)
EGO_EDGE = (240, 240, 240)
AGENT = (200, 70, 70)
FEASIBLE = (70, 200, 130)
REJECTED = (120, 60, 60)
CHOSEN = (250, 220, 90)
TEXT = (228, 228, 228)
DIM = (140, 155, 155)

PANEL = 560          # BEV panel side, px
SIDE = 430           # text panel width, px
RANGE_M = 40.0       # half-extent of the BEV view, metres


def _font(size=13):
    for p in ('/System/Library/Fonts/Menlo.ttc',
              '/System/Library/Fonts/SFNSMono.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _to_px(xy: np.ndarray) -> np.ndarray:
    """Ego-frame metres -> panel pixels, forward = up, left = left."""
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    s = PANEL / (2 * RANGE_M)
    col = PANEL / 2 - xy[:, 1] * s          # +y (left) -> left on screen
    row = PANEL / 2 - xy[:, 0] * s          # +x (forward) -> up
    return np.stack([col, row], axis=1)


def _draw_freespace(draw: ImageDraw.ImageDraw, fs) -> None:
    """Drivable / unknown as coarse tiles — full-resolution cells are invisible."""
    step = 4
    res, ox, oy = fs.res, fs.origin[0], fs.origin[1]
    nx, ny = fs.traversable.shape
    for ix in range(0, nx, step):
        for iy in range(0, ny, step):
            block_t = fs.traversable[ix:ix + step, iy:iy + step]
            block_o = fs.obstacle[ix:ix + step, iy:iy + step]
            block_u = fs.unknown[ix:ix + step, iy:iy + step]
            if block_o.any():
                c = OBSTACLE
            elif block_u.all():
                c = UNKNOWN
            elif block_t.any():
                c = ROAD
            else:
                continue
            x0 = ox + ix * res
            y0 = oy + iy * res
            p0 = _to_px([[x0, y0]])[0]
            p1 = _to_px([[x0 + step * res, y0 + step * res]])[0]
            draw.rectangle([min(p0[0], p1[0]), min(p0[1], p1[1]),
                            max(p0[0], p1[0]), max(p0[1], p1[1])], fill=c)


def _draw_poly(draw, pts_m, colour, width=2, closed=True):
    px = _to_px(pts_m)
    seq = [tuple(p) for p in px]
    if closed:
        seq.append(seq[0])
    draw.line(seq, fill=colour, width=width)


def render_frame(rec, scene, result, intent_line: str, step: int, n_steps: int,
                 metrics: dict) -> Image.Image:
    """One composite: BEV on the left, verdict / metric readout on the right."""
    img = Image.new('RGB', (PANEL + SIDE, PANEL), BG)
    d = ImageDraw.Draw(img)
    f, fb = _font(13), _font(15)

    _draw_freespace(d, scene.freespace)

    # Range rings, so distances are readable without a scale bar.
    for r in (10, 20, 30):
        p = _to_px([[0, 0]])[0]
        rr = r * PANEL / (2 * RANGE_M)
        d.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr],
                  outline=(45, 70, 70))
        d.text((p[0] + 4, p[1] - rr - 14), f'{r}m', font=f, fill=(70, 95, 95))

    for a in scene.agents:
        _draw_poly(d, ego_footprint_corners(a.xy, a.yaw, a.lwh[0], a.lwh[1]),
                   AGENT, width=2)

    # Candidates, coloured by whether the filter accepted them.
    for i, v in enumerate(result.verdicts):
        traj = result.candidates[i]
        pts = np.vstack([[0.0, 0.0], traj])
        _draw_poly(d, pts, FEASIBLE if v.feasible else REJECTED,
                   width=2, closed=False)

    if not result.emergency:
        pts = np.vstack([[0.0, 0.0], result.trajectory])
        _draw_poly(d, pts, CHOSEN, width=4, closed=False)
        for p in _to_px(result.trajectory):
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=CHOSEN)

    ego_poly = ego_footprint_corners(np.zeros(2), 0.0,
                                     scene.ego.length, scene.ego.width)
    _draw_poly(d, ego_poly, EGO_EDGE, width=2)
    d.polygon([tuple(p) for p in _to_px(ego_poly)], fill=EGO, outline=EGO_EDGE)

    # ---- readout ----------------------------------------------------------
    x0, y = PANEL + 14, 12
    d.line([PANEL, 0, PANEL, PANEL], fill=(50, 75, 75))
    d.text((x0, y), 'CLOSED LOOP', font=fb, fill=TEXT); y += 22
    d.text((x0, y), f'step {step + 1}/{n_steps}   t={rec.t:.1f}s', font=f,
           fill=DIM); y += 24

    d.text((x0, y), 'EGO', font=fb, fill=TEXT); y += 18
    d.text((x0, y), f'speed {rec.ego_v:5.2f} m/s', font=f, fill=DIM); y += 16
    d.text((x0, y), f'accel {rec.accel:+5.2f} m/s^2   steer {rec.steer:+.3f} rad',
           font=f, fill=DIM); y += 24

    if intent_line:
        d.text((x0, y), 'VLM INTENT', font=fb, fill=TEXT); y += 18
        for ln in intent_line.split('\n'):
            d.text((x0, y), ln[:46], font=f, fill=DIM); y += 16
        y += 8

    d.text((x0, y), 'SAFETY FILTER', font=fb, fill=TEXT); y += 18
    status = ('EMERGENCY BRAKE' if result.emergency
              else f'chose candidate {result.chosen_index}')
    d.text((x0, y), f'{result.feasible_count}/{len(result.verdicts)} feasible'
                    f'   {status}',
           font=f, fill=(230, 90, 90) if result.emergency else FEASIBLE); y += 20

    for v in result.verdicts[:7]:
        col = FEASIBLE if v.feasible else (185, 110, 110)
        why = ','.join(v.reasons)[:30] if v.reasons else 'ok'
        d.text((x0, y), f'  c{v.index} clr{v.min_clearance:4.1f}m  {why}',
               font=f, fill=col); y += 15
    y += 10

    s, rt, c = metrics['safety'], metrics['route'], metrics['comfort']
    d.text((x0, y), 'METRICS (so far)', font=fb, fill=TEXT); y += 18
    mc = s['min_clearance_m']
    rows = [f"collisions   {s['n_collision_steps']}",
            f"min clear    {mc:.2f} m" if mc is not None else 'min clear    n/a',
            f"brakes       {s['emergency_brakes']}"]
    if rt.get('completion') is not None:
        rows.append(f"route        {rt['completion']:.1%}")
    if not c.get('insufficient_data'):
        rows.append(f"jerk rms     {c['jerk_rms']:.2f} m/s^3")
    for r in rows:
        d.text((x0, y), r, font=f, fill=DIM); y += 16

    d.text((x0, PANEL - 46), 'green = feasible   dark red = rejected',
           font=f, fill=(90, 115, 115))
    d.text((x0, PANEL - 30), 'yellow = chosen    teal = drivable',
           font=f, fill=(90, 115, 115))
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default='/Users/trish/Downloads/nuScenes_miniV1.0')
    ap.add_argument('--scene', type=int, default=0)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--anchors',
                    default='diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy')
    ap.add_argument('--frame-ms', type=int, default=500)
    ap.add_argument('--out', default='e2e_pipeline/assets/closed_loop.gif')
    args = ap.parse_args()

    from nuscenes import NuScenes

    from .closed_loop import constant_velocity_planner, diffusiondrive_anchor_planner

    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)
    world = GTWorldModel(nusc, scene_idx=args.scene)
    cfg = LoopConfig(max_steps=args.steps, initial_speed=5.0)

    planner = (diffusiondrive_anchor_planner(args.anchors, cfg.dt)
               if Path(args.anchors).exists()
               else constant_velocity_planner(cfg.horizon, cfg.dt))

    # Re-run the loop capturing the per-step scene and filter result, which the
    # runner does not retain (records hold metrics inputs, not renderables).
    frames, captured = [], []
    orig_call = SafetyFilter.__call__

    def capturing(self, candidates, scene, scores=None, risk=None):
        res = orig_call(self, candidates, scene, scores, risk)
        res.candidates = np.asarray(candidates)
        captured.append((scene, res))
        return res

    SafetyFilter.__call__ = capturing
    try:
        runner = ClosedLoopRunner(world, planner, cfg)
        records, metrics = runner.run(command=2)
    finally:
        SafetyFilter.__call__ = orig_call

    name = nusc.scene[args.scene]['name']
    print(f'[INFO] {name}: {len(records)} steps, {len(captured)} captured')

    for k, rec in enumerate(records):
        if k >= len(captured):
            break
        scene, result = captured[k]
        partial = evaluate(records[:k + 1], world.route(), cfg.dt)
        frames.append(render_frame(rec, scene, result, '', k, len(records),
                                   partial))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pf = [f.quantize(colors=200, method=Image.MEDIANCUT, dither=Image.NONE)
          for f in frames]
    pf[0].save(out, save_all=True, append_images=pf[1:],
               duration=args.frame_ms, loop=0, disposal=2)
    print(f'[DONE] {len(frames)} frames -> {out}  ({out.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
