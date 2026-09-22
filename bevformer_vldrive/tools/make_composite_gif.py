#!/usr/bin/env python3
"""
Build animated GIFs matching bev_outputs/latest_bev_grid.jpg.

Per-frame composite layout (identical to vis_infer.py _save_bev_dual):

    ┌──────────────┬──────────────┐
    │  pred BEV    │  GT traj     │   512 × 1026   (top)
    ├──────────────┴──────────────┤
    │  FL  │   F  │  FR           │   384 × 1026   (camera grid)
    │  BL  │   B  │  BR           │
    └──────────────────────────────┘

…except the VLM REASONING / DECISION block is drawn on the **TOP** of the BACK
camera cell (row 1, centre column) instead of the bottom of the FRONT cell.

BEVFormer-Tiny is re-run over the chosen scene to render the pred-BEV / GT-
trajectory / camera panels (cameras rendered via vis_infer._make_cam_grid, which
does an explicit BGR→RGB conversion — colours are correct). Per-frame reasoning /
decision comes from Qwen2.5VL via Ollama (--vl, default) or a decisions log
(--no-vl --decisions <file>).

Two GIFs are written:
    bev_outputs/composite_bev_vlm.gif   full composite (BEV + cameras)
    bev_outputs/composite_cam_vlm.gif   camera grid only

Usage (from repo root):
    conda run -n simple_bev_vldrive python tools/make_composite_gif.py --scene 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

TOOLS_DIR = Path(__file__).resolve().parent
ROOT      = TOOLS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

# Model + rendering + styling + Ollama helpers from the live visualiser.
from vis_infer import (
    BEVFormerTiny, NuScenesMiniLoader, build_scene_canvas, make_trajectory_canvas,
    _make_cam_grid, _get_ego_pose, _load_nusc_map, _build_remap, _get_device,
    _encode_image, _query_ollama_streaming, _parse_response, _wrap,
    detections_to_text, detection_items, front_camera_b64, build_prompt,
    query_light, light_crop_b64, DecisionSmoother,
    _FONT_BODY, _BG_RGB, _WHITE, _GRAY, _DECISION_COLORS, _LABELS, _LIGHT_COLORS,
)
from dataroot import default_dataroot

OUT_DIR  = ROOT / 'bev_outputs'
OUT_BEV  = OUT_DIR / 'composite_bev_vlm.gif'
OUT_CAM  = OUT_DIR / 'composite_cam_vlm.gif'
DEC_DIR  = TOOLS_DIR / 'reasoning_decisions'   # per-scene VLM decision logs



# ── DiffusionDrive planning panel ──────────────────────────────────────────────

# Palette matched to diffusiondrive_planner/assets/demo_scene-0916.gif so the
# two projects read as one system: near-black field, forward-up, layered
# trajectories distinguished by colour with an explicit legend.
_DD_BG        = ( 13,  13,  13)
_DD_RING      = ( 48,  48,  48)
_DD_EGO       = (245, 245, 245)
_DD_AGENT     = (228, 128,  92)
_DD_AGENT_FAR = ( 92, 168, 160)
_DD_ANCHOR    = ( 96,  96,  96)     # kmeans anchors
_DD_MODE      = ( 78, 190, 172)     # scored modes (candidates)
_DD_TOP1      = (238, 126,  46)     # top-1 plan (chosen)
_DD_LOGGED    = (232, 232, 232)     # logged human path
_DD_REJECT    = ( 96,  58,  52)
_DD_LABEL     = (150, 160, 160)

# The VLM emits a coarse decision, not a speed. Mapping it here keeps the GIF to
# two VLM calls per frame; the panel says "intent from decision" so it is not
# mistaken for a directly-queried DrivingIntent (e2e_pipeline.vlm_planner does
# that properly).
_DECISION_SPEED = {'PROCEED': 1.0, 'SLOW_DOWN': 0.5, 'YIELD': 0.3, 'STOP': 0.0,
                   'UNKNOWN': 0.5}


def _logged_future(nusc, sample_token, ego_xy, ego_yaw, n=6):
    """The human's actual next n keyframe positions, in the current ego frame."""
    import numpy as _np
    pts, tok = [], sample_token
    for _ in range(n):
        nxt = nusc.get('sample', tok)['next']
        if not nxt:
            break
        tok = nxt
        ep = _get_ego_pose(nusc, tok)
        pts.append(ep['translation'][:2])
    if not pts:
        return _np.zeros((0, 2))
    c, sn = _np.cos(-ego_yaw), _np.sin(-ego_yaw)
    R = _np.array([[c, -sn], [sn, c]])
    return (_np.asarray(pts, float) - ego_xy) @ R.T


def _dd_panel(size, dd_planner, safety_filter, out, lidar2ego_yaw, score_thr,
              decision, ego_speed, nusc=None, sample_token=None,
              ego_xy=None, ego_yaw=0.0, rng_m=40.0):
    """DiffusionDrive planning view, styled after the upstream demo.

    Four trajectory layers, because they answer different questions: the anchor
    vocabulary (what the head denoises FROM), the scored modes (what it proposes
    HERE), the top-1 plan (what survived the safety filter), and the logged human
    path (what actually happened). The BEV panels beside this say what is out
    there; this says what the car intends to do about it.
    """
    import numpy as _np
    from e2e_pipeline.freespace import FREE_CLASS, FreeSpaceExtractor, GridConfig
    from e2e_pipeline.scene import EgoState, SceneRepresentation
    from e2e_pipeline.vlm_planner import DrivingIntent

    img = _np.full((size, size, 3), _DD_BG, _np.uint8)
    s_px = size / (2 * rng_m)
    cx = cy = size // 2

    def to_px(xy):
        xy = _np.atleast_2d(_np.asarray(xy, float))
        return _np.stack([cx - xy[:, 1] * s_px, cy - xy[:, 0] * s_px],
                         axis=1).astype(_np.int32)

    def polyline(pts_m, colour, width=1):
        if len(pts_m) < 2:
            return
        cv2.polylines(img, [to_px(pts_m).reshape(-1, 1, 2)], False, colour,
                      width, cv2.LINE_AA)

    # Range rings + crosshair, dim enough to read as grid not content.
    for r in (10, 20, 30, 40):
        cv2.circle(img, (cx, cy), int(r * s_px), _DD_RING, 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0), (cx, size), _DD_RING, 1)
    cv2.line(img, (0, cy), (size, cy), _DD_RING, 1)

    items = detection_items(out['cls_logits'][0].cpu(), out['reg_preds'][0].cpu(),
                            out['ref_pts'][0].cpu(), score_thr=score_thr,
                            lidar2ego_yaw=lidar2ego_yaw)
    for _n, x, y, vx, vy, _sc in items:
        rng = float(_np.hypot(x, y))
        col = _DD_AGENT if rng < 20 else _DD_AGENT_FAR
        yaw = float(_np.arctan2(vy, vx)) if _np.hypot(vx, vy) > 0.5 else 0.0
        c_, s_ = _np.cos(yaw), _np.sin(yaw)
        loc = _np.array([[-2.2, -0.9], [2.2, -0.9], [2.2, 0.9], [-2.2, 0.9]])
        box = loc @ _np.array([[c_, -s_], [s_, c_]]).T + _np.array([x, y])
        cv2.polylines(img, [to_px(box).reshape(-1, 1, 2)], True, col, 1,
                      cv2.LINE_AA)

    grid = GridConfig(x=(-10.0, 50.0, 0.5), y=(-25.0, 25.0, 0.5), z=(-1.0, 5.4, 0.4))
    sem = _np.full(grid.shape, FREE_CLASS, dtype=_np.int64)
    sem[:, :, grid.z_band_indices(-0.4, 0.2)[0]] = 11
    k_lo, k_hi = grid.z_band_indices(0.2, 2.2)
    for _n, x, y, _vx, _vy, _sc in items:
        ix = int((x - grid.x[0]) / grid.x[2]); iy = int((y - grid.y[0]) / grid.y[2])
        if 2 <= ix < grid.shape[0] - 2 and 2 <= iy < grid.shape[1] - 2:
            sem[ix - 2:ix + 3, iy - 2:iy + 3, k_lo:k_hi] = 4
    fs = FreeSpaceExtractor(grid)(sem)

    v_target = ego_speed * _DECISION_SPEED.get(decision, 0.5)
    ego = EgoState(speed=ego_speed)
    scene = SceneRepresentation(agents=[], freespace=fs, ego=ego)

    # Layer 1 -- the anchor vocabulary, all three command sets. A commanded stop
    # re-times every candidate to zero length, so without this the panel goes
    # blank exactly when the decision matters most.
    for cmd in ('left', 'right', 'straight'):
        alt, _ = dd_planner(scene, DrivingIntent(
            command=cmd, target_speed_mps=max(ego_speed, 4.0)))
        for r in alt:
            polyline(_np.vstack([[0.0, 0.0], r]), _DD_ANCHOR, 1)

    # Layer 2 -- scored modes: what the planner PROPOSES, re-timed to the speed
    # the car is actually doing. Conditioning these to the intent instead would
    # collapse all six to zero length on a commanded stop, rendering an empty
    # panel precisely when the decision is most interesting. The stop shows up
    # where it belongs -- as a short top-1 against long modes.
    modes, mode_scores = dd_planner(scene, DrivingIntent(
        command='straight', target_speed_mps=max(ego_speed, 2.0)))
    mode_res = safety_filter(modes, scene, mode_scores)
    for i, v in enumerate(mode_res.verdicts):
        polyline(_np.vstack([[0.0, 0.0], modes[i]]),
                 _DD_MODE if v.feasible else _DD_REJECT, 2)

    # Layer 4 input -- the executed plan, conditioned to the VLM's intent.
    cands, scores = dd_planner(scene, DrivingIntent(command='straight',
                                                    target_speed_mps=v_target))
    res = safety_filter(cands, scene, scores)

    # Layer 3 -- the logged human path, as a reference the plan can be read against.
    if nusc is not None and sample_token is not None and ego_xy is not None:
        fut = _logged_future(nusc, sample_token, ego_xy, ego_yaw)
        if len(fut):
            polyline(_np.vstack([[0.0, 0.0], fut]), _DD_LOGGED, 1)

    # Layer 4 -- top-1 plan.
    if not res.emergency:
        polyline(_np.vstack([[0.0, 0.0], res.trajectory]), _DD_TOP1, 3)
        for p in to_px(res.trajectory):
            cv2.circle(img, (int(p[0]), int(p[1])), 3, _DD_TOP1, -1, cv2.LINE_AA)

    cv2.rectangle(img, (cx - 4, cy - 9), (cx + 4, cy + 9), _DD_EGO, -1)

    F, SC = cv2.FONT_HERSHEY_SIMPLEX, 0.35
    cv2.putText(img, f'BEV forward is up   {len(items)} agents   '
                     f'{len(modes)} denoised modes', (10, 18), F, SC, _DD_LABEL, 1, cv2.LINE_AA)
    head = (f'{decision} -> intent {v_target:.1f} m/s    ego {ego_speed:.1f} m/s    '
            f'{mode_res.feasible_count}/{len(modes)} modes feasible')
    cv2.putText(img, head, (10, 34), F, SC,
                (228, 110, 110) if res.emergency else _DD_LABEL, 1, cv2.LINE_AA)

    legend = [('top-1 plan', _DD_TOP1), ('scored modes', _DD_MODE),
              ('kmeans anchors', _DD_ANCHOR), ('logged path', _DD_LOGGED)]
    for j, (name, col) in enumerate(legend):
        lx = 12 + (j % 2) * 150
        ly = size - 52 + (j // 2) * 16
        cv2.line(img, (lx, ly - 4), (lx + 18, ly - 4), col, 2, cv2.LINE_AA)
        cv2.putText(img, name, (lx + 24, ly), F, 0.32, _DD_LABEL, 1, cv2.LINE_AA)
    return img


# ── Overlay: VLM block anchored to the TOP of a panel ───────────────────────────
def _overlay_vl_text_top(cell, reasoning, decision, light='none'):
    """Draw the VLM reasoning/decision block on the TOP of *cell* (RGB uint8)."""
    H, W   = cell.shape[:2]
    h_box  = min(160, H // 2)
    wrap_cols = max(18, W // 9)
    result = cell.copy()

    result[:h_box] = (
        result[:h_box].astype(np.float32) * 0.35
        + np.array(_BG_RGB, np.float32) * 0.65
    ).clip(0, 255).astype(np.uint8)

    img  = Image.fromarray(result)
    draw = ImageDraw.Draw(img)
    pad  = 8

    draw.line([(0, h_box), (W, h_box)], fill=(60, 100, 100), width=1)
    draw.text((pad, 4), "REASONING", font=_FONT_BODY, fill=_WHITE)
    # Light state comes only from the forward camera — see vis_infer.build_prompt.
    if light in _LIGHT_COLORS:
        chip = f"LIGHT {light.upper()}"
        cw = int(draw.textlength(chip, font=_FONT_BODY))
        draw.text((W - pad - cw, 4), chip, font=_FONT_BODY, fill=_LIGHT_COLORS[light])
    y = 22
    for ln in _wrap(reasoning or '', wrap_cols)[:3]:
        draw.text((pad, y), ln, font=_FONT_BODY, fill=_GRAY)
        y += 16

    if decision in _DECISION_COLORS and decision != "UNKNOWN":
        dec_color = _DECISION_COLORS[decision]
        dec_label = _LABELS.get(decision, decision)
        y_box = h_box - 28
        draw.rectangle([(pad, y_box), (W - pad, y_box + 22)],
                       fill=tuple(max(0, c // 5) for c in dec_color),
                       outline=dec_color, width=2)
        draw.text((pad + 6, y_box + 4), dec_label, font=_FONT_BODY, fill=dec_color)

    return np.array(img, dtype=np.uint8)


def _overlay_back_top(cam_grid, reasoning, decision, light='none'):
    """Overlay the VLM block on the TOP of the BACK cell (row 1, centre col)."""
    out    = cam_grid.copy()
    cell_w = out.shape[1] // 3
    cell_h = out.shape[0] // 2
    back   = out[cell_h:2 * cell_h, cell_w:2 * cell_w]
    out[cell_h:2 * cell_h, cell_w:2 * cell_w] = _overlay_vl_text_top(
        back, reasoning, decision, light)
    return out


def _find_plan_anchors() -> Path | None:
    """Locate kmeans_plan_6.npy, or None.

    A vendored copy lives in this project's assets/ (992 bytes) so a fresh clone
    renders the planning panel without first having to generate anything. The
    diffusiondrive_planner path is checked second for anyone regenerating them
    from full nuScenes -- mini-derived anchors cluster only 22 right and 30 left
    turns, which is fine for pipeline validation and not for anything else.
    """
    for c in (ROOT / 'assets' / 'kmeans_plan_6.npy',
              ROOT.parent / 'diffusiondrive_planner/data/kmeans/kmeans_plan_6.npy'):
        if c.exists():
            return c
    return None


def _load_planner():
    """(planner, safety_filter), or (None, None) if the panel cannot be built."""
    sys.path.insert(0, str(ROOT.parent))
    try:
        from e2e_pipeline.safety_filter import SafetyFilter
        from e2e_pipeline.vlm_planner import intent_conditioned_planner
    except ImportError as exc:
        print(f'[WARN] e2e_pipeline unavailable ({exc}) — planning panel skipped.')
        return None, None

    anchors = _find_plan_anchors()
    if anchors is None:
        print('[WARN] kmeans_plan_6.npy not found — planning panel skipped.\n'
              '       Regenerate with:\n'
              '         python ../diffusiondrive_planner/tools/gen_plan_anchors.py \\\n'
              '             --dataroot <nuScenes> --version v1.0-mini \\\n'
              '             --out ../diffusiondrive_planner/data/kmeans --mode per_command')
        return None, None
    return intent_conditioned_planner(str(anchors)), SafetyFilter()


def _label_panel(panel, text):
    """Draw a camera-cell-style label (dark box + light text) at bottom-left."""
    out = panel.copy()
    H   = out.shape[0]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(out, (2, H - th - 10), (tw + 8, H - 2), (0, 0, 0), -1)
    cv2.putText(out, text, (5, H - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return out


def _latest_complete_run(rows, n_frames):
    """Return the last contiguous run that contains frames 0..n_frames-1."""
    runs, cur = [], []
    for r in rows:
        if r['frame'] == 0 and cur:
            runs.append(cur); cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    complete = [run for run in runs if len(run) >= n_frames]
    if not complete:
        raise SystemExit(f"No complete {n_frames}-frame run found")
    return complete[-1]


def _save_gif(frames, path, frame_ms):
    """Write a GIF with per-frame adaptive (local) palettes.

    A single shared/optimised palette is dominated by the flat teal BEV region
    and visibly distorts the photographic camera tones, so each frame gets its
    own median-cut palette instead.
    """
    pframes = [f.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)
               for f in frames]
    pframes[0].save(path, save_all=True, append_images=pframes[1:],
                    duration=frame_ms, loop=0, disposal=2)


def _render_scene(model, nusc, loader, scene_idx, args, device, log_path=None):
    """Render one scene → (comp_frames, cam_frames) as lists of PIL images."""
    scene = nusc.scene[scene_idx]
    loc   = nusc.get('log', scene['log_token'])['location']
    print(f'[INFO] Scene {scene_idx:2d} : {scene["name"]}  ({loc})')
    print(f'[INFO]   desc   : {scene["description"]}')

    nusc_map = _load_nusc_map(args.dataroot, loc)
    _first   = nusc.get('sample', scene['first_sample_token'])
    _lid_cs  = nusc.get('calibrated_sensor',
                        nusc.get('sample_data', _first['data']['LIDAR_TOP'])
                        ['calibrated_sensor_token'])
    lidar2ego_yaw = Quaternion(_lid_cs['rotation']).yaw_pitch_roll[0]

    # Resolve reasoning source. With --vl we query Ollama, but reuse a complete
    # per-scene log if one already exists (resumable; avoids re-querying).
    by_frame, querying = {}, args.vl
    if args.vl and log_path and Path(log_path).exists():
        rows = [json.loads(l) for l in Path(log_path).read_text().splitlines() if l.strip()]
        if sum(1 for r in rows if r.get('reasoning')) >= args.max_frames:
            by_frame = {r['frame']: r for r in rows}
            querying = False
            print(f'[INFO]   reuse cached decisions ({log_path})')
    if not args.vl:
        dec_path = Path(args.decisions or (DEC_DIR / 'decisions.jsonl'))
        rows = [json.loads(l) for l in dec_path.read_text().splitlines() if l.strip()]
        by_frame = {r['frame']: r for r in _latest_complete_run(rows, args.max_frames)}

    log_fh = open(log_path, 'w') if (querying and log_path) else None
    prev_bev, frame_idx, patch_origin = None, 0, None
    _prev_xy, ego_speed = None, 5.0
    # Per-scene temporal state: latching a stop across a scene cut is meaningless.
    smoother = DecisionSmoother()
    prev_decision: dict | None = None

    # DiffusionDrive anchors + the safety filter, built once per scene. Missing
    # anchors degrade to the two-panel layout rather than crashing: the tool's
    # primary job is the BEV + VLM composite, and the planning panel is an extra.
    dd_planner, safety_filter = _load_planner()
    ego_history, comp_frames, cam_frames = [], [], []

    with torch.no_grad():
        for sample in loader.iter_scene(scene_idx=scene_idx):
            if frame_idx >= args.max_frames:
                break
            imgs, img_metas = sample['imgs'], sample['img_metas']

            t0  = time.perf_counter()
            out = model(imgs, img_metas, prev_bev=prev_bev)
            if device.type == 'mps':
                torch.mps.synchronize()
            bev_ms   = (time.perf_counter() - t0) * 1000
            prev_bev = out['bev_feat'].detach()

            sample_token = img_metas[0]['sample_token']
            # Ego speed from consecutive logged poses; the anchors are re-timed
            # against it, so a standing start is not offered a highway anchor.
            _ep = _get_ego_pose(nusc, sample_token)
            _now = np.array(_ep['translation'][:2])
            ego_speed = (float(np.linalg.norm(_now - _prev_xy)) / 0.5
                         if _prev_xy is not None else 5.0)
            _prev_xy = _now
            ego_pose     = _get_ego_pose(nusc, sample_token)
            ego_tx = float(ego_pose['translation'][0])
            ego_ty = float(ego_pose['translation'][1])
            ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]
            ego_history.append((ego_tx, ego_ty, ego_yaw))
            if patch_origin is None:
                patch_origin = (ego_tx, ego_ty)

            canvas = build_scene_canvas(
                out, ego_pose, nusc_map, patch_origin=patch_origin,
                patch_range=args.range, canvas_size=args.canvas,
                score_thr=args.score_thr, lidar2ego_yaw=lidar2ego_yaw,
                heading_up=True)
            trail_canvas = make_trajectory_canvas(
                ego_history, nusc_map, patch_origin, patch_range=args.range,
                canvas_size=args.canvas, nusc=nusc, sample_token=sample_token,
                heading_up=True,
                ego_yaw=Quaternion(_ep['rotation']).yaw_pitch_roll[0])
            bev_total_w = canvas.shape[0] * 2 + 2
            cam_grid = _make_cam_grid(
                nusc, sample_token, args.dataroot,
                out['cls_logits'][0].cpu(), out['reg_preds'][0].cpu(),
                out['ref_pts'][0].cpu(), args.score_thr, total_w=bev_total_w)

            # ── Reasoning / decision ────────────────────────────────────────────
            vl_ms = 0.0
            light = 'none'
            vl_stats: dict = {}
            if querying:
                vis_path = str(OUT_DIR / '_vlm_query.png')
                cv2.imwrite(vis_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

                # Geometry as authoritative text, semantics as the forward camera.
                det_text = detections_to_text(
                    out['cls_logits'][0].cpu(), out['reg_preds'][0].cpu(),
                    out['ref_pts'][0].cpu(), score_thr=args.score_thr,
                    lidar2ego_yaw=lidar2ego_yaw) if args.det_text else None
                payload_images = [_encode_image(vis_path)]
                cam_b64 = (front_camera_b64(nusc, sample_token, args.dataroot,
                                            max_width=args.cam_width)
                           if args.front_cam else None)
                if cam_b64:
                    payload_images.append(cam_b64)

                t1 = time.perf_counter()
                # Stage 1: the camera alone (plus a map-projected zoom on the
                # fixture) reads the light. It must be its own call — any
                # detection text in context suppresses the reading entirely.
                light_pre = 'none' if cam_b64 else None
                if cam_b64:
                    crop_b64 = (light_crop_b64(nusc, sample_token, args.dataroot)
                                if args.light_crop else None)
                    try:
                        light_pre = query_light(cam_b64, args.ollama_model,
                                                args.ollama_url, args.ollama_timeout,
                                                vl_stats, crop_b64=crop_b64)
                    except Exception as exc:
                        print(f'[WARN] light query failed ({exc})')
                try:
                    # Stage 2: decision, with the light state supplied as text.
                    rawtxt = _query_ollama_streaming(
                        payload_images, args.ollama_model, args.ollama_url,
                        args.ollama_timeout, on_update=lambda _t: None,
                        prompt=build_prompt(det_text,
                                            with_front_cam=cam_b64 is not None,
                                            light_state=light_pre,
                                            prev=prev_decision),
                        stats=vl_stats)
                    reasoning, decision, light = _parse_response(rawtxt)
                    if light_pre is not None:
                        light = light_pre       # stage 1 outranks stage 2's echo
                    decision = smoother.update(decision)
                    prev_decision = {'decision': decision, 'light': light,
                                     'reasoning': reasoning}
                except Exception as exc:
                    reasoning, decision = f'[VLM error: {exc}]', 'UNKNOWN'
                vl_ms = (time.perf_counter() - t1) * 1000
            else:
                rec = by_frame.get(frame_idx, {})
                reasoning = rec.get('reasoning', '').strip()
                decision  = rec.get('decision', 'UNKNOWN')
                light     = rec.get('light', 'none')

            if log_fh:
                log_fh.write(json.dumps({
                    'frame': frame_idx, 'token': sample_token,
                    'decision': decision, 'reasoning': reasoning, 'light': light,
                    'bev_ms': round(bev_ms, 1), 'vl_ms': round(vl_ms, 1),
                    'prefill_ms': round(vl_stats.get('prefill_ms', 0.0), 1),
                    'decode_ms': round(vl_stats.get('decode_ms', 0.0), 1),
                    'prompt_tokens': vl_stats.get('prompt_tokens', 0)}) + '\n')
                log_fh.flush()

            # ── Assemble: cameras on top, three planning panels beneath ────────
            pred_panel = _label_panel(canvas, 'pred BEV')
            traj_panel = _label_panel(trail_canvas, 'GT trajectory')
            sep = np.full((canvas.shape[0], 2, 3), 255, dtype=np.uint8)
            panels = [pred_panel, sep, traj_panel]
            if dd_planner is not None:
                dd = _dd_panel(canvas.shape[0], dd_planner, safety_filter, out,
                               lidar2ego_yaw, args.score_thr, decision, ego_speed,
                               nusc=nusc, sample_token=sample_token,
                               ego_xy=_now, ego_yaw=Quaternion(_ep['rotation'])
                               .yaw_pitch_roll[0])
                panels += [sep, _label_panel(dd, 'DiffusionDrive plan + safety filter')]
            bottom = np.concatenate(panels, axis=1)

            grid_out = _overlay_back_top(cam_grid, reasoning, decision, light)
            if grid_out.shape[1] != bottom.shape[1]:
                new_h = int(round(grid_out.shape[0] * bottom.shape[1]
                                  / grid_out.shape[1]))
                grid_out = cv2.resize(grid_out, (bottom.shape[1], new_h),
                                      interpolation=cv2.INTER_AREA)
            h_sep     = np.full((4, bottom.shape[1], 3), 40, dtype=np.uint8)
            composite = np.concatenate([grid_out, h_sep, bottom], axis=0)

            comp_frames.append(Image.fromarray(composite.astype(np.uint8), 'RGB'))
            cam_frames.append(Image.fromarray(grid_out.astype(np.uint8), 'RGB'))
            print(f'  frame {frame_idx}: bev {bev_ms:6.1f} ms  vl {vl_ms:6.0f} ms  '
                  f'{decision:9s} | {reasoning[:42]}')
            frame_idx += 1

    if log_fh:
        log_fh.close()
    return comp_frames, cam_frames


def _intersection_scenes(nusc):
    """Scene indices whose description mentions an intersection / crosswalk."""
    keys = ('intersection', 'crosswalk', 'cross ')
    return [i for i, sc in enumerate(nusc.scene)
            if any(k in sc['description'].lower() for k in keys)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot',   default=default_dataroot())
    ap.add_argument('--scenes',     type=int, nargs='+', default=None,
                    help='Scene indices (default: all intersection scenes)')
    ap.add_argument('--max-frames', type=int,   default=10)
    ap.add_argument('--score-thr',  type=float, default=0.25)
    ap.add_argument('--canvas',     type=int,   default=512)
    ap.add_argument('--range',      type=float, default=150.)
    ap.add_argument('--checkpoint',
                    default='model/checkpoints/bevformer_tiny_fp16_epoch_24.pth')
    ap.add_argument('--frame-ms',   type=int,   default=800)
    ap.add_argument('--vl', action=argparse.BooleanOptionalAction, default=True,
                    help='Query Qwen2.5VL via Ollama per frame (default on)')
    ap.add_argument('--cam-gif', action=argparse.BooleanOptionalAction, default=False,
                    help='Also write a camera-only GIF per scene')
    ap.add_argument('--ollama-url',     default='http://localhost:11434')
    ap.add_argument('--ollama-model',   default='qwen2.5vl:7b')
    ap.add_argument('--ollama-timeout', type=int, default=120)
    ap.add_argument('--front-cam', dest='front_cam', action='store_true', default=True,
                    help='Send CAM_FRONT as a second image (default on)')
    ap.add_argument('--no-front-cam', dest='front_cam', action='store_false')
    ap.add_argument('--cam-width', type=int, default=640,
                    help='Downscale CAM_FRONT to this width. Values below 640 '
                         'do not reduce tokens further (processor rescales)')
    ap.add_argument('--det-text', dest='det_text', action='store_true', default=True,
                    help='Hand detections to the VLM as authoritative metric text')
    ap.add_argument('--no-det-text', dest='det_text', action='store_false')
    ap.add_argument('--light-crop', dest='light_crop', action='store_true',
                    default=True, help='Map-projected zoom on the traffic light')
    ap.add_argument('--no-light-crop', dest='light_crop', action='store_false')
    ap.add_argument('--decisions',  default=None,
                    help='With --no-vl: decisions.jsonl to read reasoning from')
    args = ap.parse_args()

    device = _get_device()
    print(f'[INFO] Device : {device}   VLM: {"on" if args.vl else "off"}')

    model = BEVFormerTiny(pretrained_backbone=False)
    model.eval()
    ckpt  = torch.load(args.checkpoint, map_location='cpu')
    remap = _build_remap(ckpt.get('state_dict', ckpt))
    res   = model.load_state_dict(remap, strict=False)
    print(f'[INFO] Checkpoint : {args.checkpoint}  '
          f'({len(remap) - len(res.unexpected_keys)}/{len(remap)} keys)')

    loader = NuScenesMiniLoader(args.dataroot)
    nusc   = loader.nusc

    scenes = args.scenes if args.scenes is not None else _intersection_scenes(nusc)
    gif_dir = OUT_DIR / 'scene_gifs'
    gif_dir.mkdir(parents=True, exist_ok=True)
    print(f'[INFO] Scenes : {scenes}  →  {gif_dir}/')

    written = []
    for s in scenes:
        print('─' * 72)
        name = nusc.scene[s]['name']
        DEC_DIR.mkdir(parents=True, exist_ok=True)
        log  = DEC_DIR / f'decisions_scene{s}.jsonl'
        comp, cam = _render_scene(model, nusc, loader, s, args, device, log_path=str(log))
        if not comp:
            print(f'[WARN] scene {s} produced no frames'); continue
        bev_path = gif_dir / f'scene{s:02d}_{name}_bev.gif'
        _save_gif(comp, str(bev_path), args.frame_ms)
        written.append(bev_path)
        if args.cam_gif:
            cam_path = gif_dir / f'scene{s:02d}_{name}_cam.gif'
            _save_gif(cam, str(cam_path), args.frame_ms)
            written.append(cam_path)

    print('═' * 72)
    print(f'Wrote {len(written)} GIF(s):')
    for p in written:
        print('  ', p)


if __name__ == '__main__':
    main()
