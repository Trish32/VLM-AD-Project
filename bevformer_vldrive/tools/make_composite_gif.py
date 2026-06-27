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
    _FONT_BODY, _BG_RGB, _WHITE, _GRAY, _DECISION_COLORS, _LABELS,
)

OUT_DIR  = ROOT / 'bev_outputs'
OUT_BEV  = OUT_DIR / 'composite_bev_vlm.gif'
OUT_CAM  = OUT_DIR / 'composite_cam_vlm.gif'
DEC_DIR  = TOOLS_DIR / 'reasoning_decisions'   # per-scene VLM decision logs


# ── Overlay: VLM block anchored to the TOP of a panel ───────────────────────────
def _overlay_vl_text_top(cell, reasoning, decision):
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


def _overlay_back_top(cam_grid, reasoning, decision):
    """Overlay the VLM block on the TOP of the BACK cell (row 1, centre col)."""
    out    = cam_grid.copy()
    cell_w = out.shape[1] // 3
    cell_h = out.shape[0] // 2
    back   = out[cell_h:2 * cell_h, cell_w:2 * cell_w]
    out[cell_h:2 * cell_h, cell_w:2 * cell_w] = _overlay_vl_text_top(
        back, reasoning, decision)
    return out


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
                score_thr=args.score_thr, lidar2ego_yaw=lidar2ego_yaw)
            trail_canvas = make_trajectory_canvas(
                ego_history, nusc_map, patch_origin, patch_range=args.range,
                canvas_size=args.canvas, nusc=nusc, sample_token=sample_token)
            bev_total_w = canvas.shape[0] * 2 + 2
            cam_grid = _make_cam_grid(
                nusc, sample_token, args.dataroot,
                out['cls_logits'][0].cpu(), out['reg_preds'][0].cpu(),
                out['ref_pts'][0].cpu(), args.score_thr, total_w=bev_total_w)

            # ── Reasoning / decision ────────────────────────────────────────────
            vl_ms = 0.0
            if querying:
                vis_path = str(OUT_DIR / '_vlm_query.png')
                cv2.imwrite(vis_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                t1 = time.perf_counter()
                try:
                    rawtxt = _query_ollama_streaming(
                        _encode_image(vis_path), args.ollama_model, args.ollama_url,
                        args.ollama_timeout, on_update=lambda _t: None)
                    reasoning, decision = _parse_response(rawtxt)
                except Exception as exc:
                    reasoning, decision = f'[VLM error: {exc}]', 'UNKNOWN'
                vl_ms = (time.perf_counter() - t1) * 1000
            else:
                rec = by_frame.get(frame_idx, {})
                reasoning = rec.get('reasoning', '').strip()
                decision  = rec.get('decision', 'UNKNOWN')

            if log_fh:
                log_fh.write(json.dumps({
                    'frame': frame_idx, 'token': sample_token,
                    'decision': decision, 'reasoning': reasoning,
                    'bev_ms': round(bev_ms, 1), 'vl_ms': round(vl_ms, 1)}) + '\n')
                log_fh.flush()

            # ── Assemble (mirror _save_bev_dual, BACK-top overlay) ──────────────
            pred_panel = _label_panel(canvas, 'pred BEV')
            traj_panel = _label_panel(trail_canvas, 'GT trajectory')
            sep = np.full((canvas.shape[0], 2, 3), 255, dtype=np.uint8)
            top = np.concatenate([pred_panel, sep, traj_panel], axis=1)

            grid_out = _overlay_back_top(cam_grid, reasoning, decision)
            if grid_out.shape[1] != top.shape[1]:
                new_h = int(round(grid_out.shape[0] * top.shape[1] / grid_out.shape[1]))
                grid_out = cv2.resize(grid_out, (top.shape[1], new_h),
                                      interpolation=cv2.INTER_AREA)
            h_sep     = np.full((4, top.shape[1], 3), 40, dtype=np.uint8)
            composite = np.concatenate([top, h_sep, grid_out], axis=0)

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
    ap.add_argument('--dataroot',   default='/Users/trish/Downloads/nuScenes_miniV1.0')
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
