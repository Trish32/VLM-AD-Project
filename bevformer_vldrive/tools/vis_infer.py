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
_BEV_LEGEND = (
    "  RED shapes     — vehicles (car, truck, bus, motorcycle, bicycle)\n"
    "  BLUE shapes    — pedestrians\n"
    "  CYAN shapes    — barriers / static obstacles\n"
    "  MAGENTA shapes — traffic cones\n"
    "  BLUE circle + white arrow — ego vehicle; arrow = heading direction.\n"
    "  Teal surface   — drivable road.\n"
)

# The LIGHT line is only offered when the forward camera is actually in the payload.
# Asking for it BEV-only invites fabrication, and it does: with --no-front-cam the
# model confidently answered "LIGHT: red / DECISION: STOP" on a frame it had no
# camera view of at all.  Never ask a model for a field it has no input for.
_OUTPUT_CONTRACT_CAM = (
    "Reply in EXACTLY this format (no extra lines):\n"
    "LIGHT: <red | yellow | green | none>   <- read from IMAGE 2 only; if no\n"
    "       traffic light is visible in IMAGE 2, you MUST answer none\n"
    "REASONING: <one sentence>\n"
    "DECISION: <PROCEED | SLOW_DOWN | YIELD | STOP>"
)

_OUTPUT_CONTRACT_BEV = (
    "Reply in EXACTLY this format (no extra lines):\n"
    "REASONING: <one sentence>\n"
    "DECISION: <PROCEED | SLOW_DOWN | YIELD | STOP>"
)


# Stage 1 of the two-stage query: the camera alone, asked one thing only.
#
# WHY TWO STAGES.  Measured on scene-0757 (red light, empty road): with the BEV
# and the camera in ONE call, adding even a SINGLE row of detection text flips
# the answer from "LIGHT: red / STOP" to "LIGHT: none / PROCEED".  The sweep is
# unambiguous — rows=0 reads the light, rows=1..5 all miss it, at nearly
# identical token counts.  So this is not a context-length limit; authoritative
# text redirects the model off the visual task wholesale.  One call cannot do
# perception-from-pixels and reasoning-over-numbers at the same time, so we stop
# asking it to.
_LIGHT_PROMPT = (
    "This is the forward-facing camera of a car.\n"
    "Is there a traffic light governing this lane? If yes, what colour is lit?\n"
    "Reply with EXACTLY one word: red, yellow, green, or none."
)


# Cache of per-location fixture geometry; the map JSONs are a few MB each and the
# same location repeats across every frame of a scene.
_FIXTURE_CACHE: dict = {}


def light_crop_b64(nusc, sample_token: str, dataroot: str,
                   crop_out: int = 384, ref_range: float = 25.0,
                   base_half: int = 170, max_range: float = 60.0) -> str | None:
    """A tight crop of CAM_FRONT centred on the traffic light, or None.

    The map expansion knows where every fixture is, so there is no need to make
    the VLM hunt for a signal head in a downscaled wide shot. Projecting the
    fixture with the ego pose and cropping around it recovers exactly the detail
    that range destroys: asked in isolation the model reads a tight crop
    correctly every time, while the same light at 32 m inside a 640 px frame was
    missed.

    The window scales as 1/range so a light at 50 m ends up the same apparent
    size as one at 15 m; with a fixed window the far ones stay unreadable, which
    defeats the purpose.

    Returns None when no fixture projects into view — the caller should then fall
    back to the full frame rather than sending a meaningless crop.
    """
    try:
        from make_light_annotation import fixture_records, project

        scene = nusc.get('scene', nusc.get('sample', sample_token)['scene_token'])
        loc = nusc.get('log', scene['log_token'])['location']
        if loc not in _FIXTURE_CACHE:
            _FIXTURE_CACHE[loc] = fixture_records(dataroot, loc)
        fixtures = _FIXTURE_CACHE[loc]
        if not fixtures:
            return None

        sd = nusc.get('sample_data',
                      nusc.get('sample', sample_token)['data']['CAM_FRONT'])
        ep = nusc.get('ego_pose', sd['ego_pose_token'])
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

        best = None
        for fx in fixtures:
            p = np.array([fx['xy'][0], fx['xy'][1], ep['translation'][2] + fx['z']])
            pr = project(p, ep, cs)
            if pr is None:
                continue
            u, v, depth = pr
            if not (60 <= u < 1600 - 60 and 60 <= v < 900 - 60):
                continue
            if depth > max_range:
                continue
            if best is None or depth < best[2]:
                best = (u, v, depth)
        if best is None:
            return None

        img = cv2.imread(str(Path(dataroot) / sd['filename']))
        if img is None:
            return None
        u, v, depth = best
        half = int(np.clip(base_half * ref_range / max(depth, 1.0), 70, 300))
        x0 = int(np.clip(u - half, 0, 1600 - 2 * half))
        y0 = int(np.clip(v - half, 0, 900 - 2 * half))
        crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
        if crop.shape[0] < 40 or crop.shape[1] < 40:
            return None
        crop = cv2.resize(crop, (crop_out, crop_out), interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else None
    except Exception as exc:
        print(f'[WARN] light crop unavailable ({exc}) — using the wide frame only.')
        return None


_LIGHT_PROMPT_CROPPED = (
    "Two views from a car's forward camera.\n"
    "IMAGE 1 — the wide view: use it to judge whether a signal governs THIS lane.\n"
    "IMAGE 2 — a zoomed crop centred on the traffic light in that scene.\n"
    "What colour is lit on the light governing this lane?\n"
    "Reply with EXACTLY one word: red, yellow, green, or none."
)


def query_light(cam_b64: str, model: str, base_url: str, timeout: int,
                stats: dict | None = None, crop_b64: str | None = None) -> str:
    """Stage 1 — read the traffic light from the camera, nothing else in context.

    With `crop_b64` the wide frame and a map-projected zoom go in together: the
    wide view carries lane context (which signal is mine), the crop carries the
    state that range destroys. Without it this is the single-image query.
    """
    images = [cam_b64] + ([crop_b64] if crop_b64 else [])
    payload = json.dumps({
        "model": model,
        "prompt": _LIGHT_PROMPT_CROPPED if crop_b64 else _LIGHT_PROMPT,
        "images": images,
        "stream": False, "options": {"temperature": 0.0, "num_predict": 8},
    }).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read())
    if stats is not None:
        stats.update(light_prefill_ms=obj.get("prompt_eval_duration", 0) / 1e6,
                     light_tokens=obj.get("prompt_eval_count", 0))
    tok = obj.get("response", "").strip().lower()
    return next((c for c in _VALID_LIGHTS if c in tok), "none")


def build_prompt(det_text: str | None = None, with_front_cam: bool = False,
                 light_state: str | None = None,
                 prev: dict | None = None) -> str:
    """Assemble the VLM prompt for the enabled input channels.

    The division of labour is the point of this function.  A rendered BEV is a
    lossy, synthetic picture of geometry the pipeline has *already measured* — so
    asking the VLM to read distances off it is asking the one component that is bad
    at metric estimation to redo work the detector did exactly.  Meanwhile traffic
    light state, brake lights, signage and pedestrian intent exist nowhere in the
    BEV at all, which makes a correct decision at a red light with clear road ahead
    structurally impossible.

    So: detections go in as text and are declared authoritative, the front camera
    goes in as a second image and is declared the only source of semantics.
    """
    parts = []

    if with_front_cam:
        parts.append(
            "You are a driving assistant. You are given TWO images.\n\n"
            "IMAGE 1 — top-down Bird's Eye View (BEV) map, global-frame, north-up.\n"
            f"{_BEV_LEGEND}\n"
            "IMAGE 2 — the forward-facing camera (CAM_FRONT).\n"
        )
    else:
        parts.append(
            "Analyze this top-down Bird's Eye View (BEV) map of a driving scene.\n"
            "Canvas is global-frame, north-up.  Colour legend:\n"
            f"{_BEV_LEGEND}"
        )

    # Stage-1 result arrives as TEXT, not as a picture to re-read.  The camera has
    # already been looked at by a dedicated call; asking this call to look again
    # is what fails (see the note on _LIGHT_PROMPT).
    if light_state and light_state != "none":
        parts.append(
            f"\nTRAFFIC LIGHT GOVERNING YOUR LANE: **{light_state.upper()}**\n"
            "(read from the forward camera; treat this as fact).\n"
        )
    elif light_state == "none":
        parts.append("\nTRAFFIC LIGHT: none visible in the forward camera.\n")

    if with_front_cam:
        parts.append(
            "\nAlso use IMAGE 2 for what the BEV physically cannot represent:\n"
            "  brake lights, turn signals, road signs, construction markings,\n"
            "  and whether a pedestrian is about to step out.\n"
        )

    if det_text:
        parts.append(
            "\nTHEN use these MEASURED DETECTIONS (authoritative — they come from\n"
            "the 3-D detector, not from the pictures. Do NOT estimate distances or\n"
            "speeds from the images; use these numbers). Forward hemisphere only:\n"
            f"{det_text}\n"
        )

    # Temporal context. Every frame was previously an independent query, which is
    # why the decision stream jitters on scenes that barely change. Stating what
    # was decided a moment ago costs a handful of text tokens and gives the model
    # a reason to stay put unless something actually changed.  Deliberately framed
    # as evidence, not as an instruction to agree — anchoring it too hard would
    # trade jitter for an inability to react.
    if prev and prev.get('decision') and prev['decision'] != 'UNKNOWN':
        parts.append(f"\nPREVIOUS FRAME (0.5 s ago): decided {prev['decision']}")
        if prev.get('light') and prev['light'] != 'none':
            parts.append(f", light was {prev['light']}")
        if prev.get('reasoning'):
            parts.append(f'\n  because: "{prev["reasoning"]}"')
        parts.append("\nKeep that decision unless the scene has actually changed.\n")

    parts.append("\nDetermine if there is a blocking hazard ahead and decide:\n"
                 "[PROCEED, SLOW_DOWN, YIELD, STOP]\n")
    if light_state in ("red", "yellow"):
        parts.append(
            f"A {light_state} light governing your lane means STOP even if the\n"
            "road ahead is completely clear. This overrides the detections.\n")
    # The LIGHT field is only requested when stage 1 supplied one; asking for it
    # with no camera evidence produced confident fabrication ("LIGHT: red" on a
    # BEV-only payload).
    parts.append(f"\n{_OUTPUT_CONTRACT_CAM if light_state is not None else _OUTPUT_CONTRACT_BEV}")
    return "".join(parts)


# Kept for callers that want the plain BEV-only prompt (and for the cached-decision
# path in make_composite_gif.py, which must not change behaviour when --no-vl).
_SYSTEM_PROMPT = build_prompt()

_VALID_DECISIONS = {"PROCEED", "SLOW_DOWN", "YIELD", "STOP"}
_VALID_LIGHTS = ("red", "yellow", "green", "none")
# "none" is deliberately absent — an unlit chip would just be visual noise on the
# overwhelming majority of frames that have no signal in view.
_LIGHT_COLORS = {
    "red":    (230,  60,  60),
    "yellow": (230, 200,  60),
    "green":  ( 70, 220, 110),
}
# ── Temporal hysteresis on the decision stream ─────────────────────────────────

# Ordered by how much caution each implies. UNKNOWN sits at SLOW_DOWN: a parse
# failure should not read as "proceed".
_SEVERITY = {'PROCEED': 0, 'SLOW_DOWN': 1, 'UNKNOWN': 1, 'YIELD': 2, 'STOP': 3}


class DecisionSmoother:
    """Latch severity across frames so one noisy call cannot flip the output.

    The need is measured, not theoretical.  scene-0757 frame 1 returned STOP on
    one run and PROCEED on the next from identical inputs (decision temperature
    is 0.1, so the call is not deterministic), and the original logs show
    STOP -> PROCEED -> SLOW_DOWN inside 1.5 s on a barely-changing scene.  No
    prompt change fixes that; it is sampling noise on a per-frame independent
    query, and it belongs outside the model.

    The asymmetry is deliberate and is the whole point:

      * ESCALATION is immediate.  The first frame that says STOP, we stop.  Making
        caution wait for confirmation would be the one failure mode worth avoiding.
      * DE-ESCALATION needs `release_frames` consecutive quieter frames before it
        takes effect, so a single optimistic sample cannot release a stop.
      * STOP additionally holds for `stop_latch` frames regardless, because at
        2 Hz a one-frame stop is not physically actionable anyway.

    Pure post-processing: no extra model calls, no added latency.
    """

    def __init__(self, release_frames: int = 2, stop_latch: int = 2) -> None:
        self.release_frames = int(release_frames)
        self.stop_latch = int(stop_latch)
        self.held: str | None = None
        self._quieter_run = 0
        self._stop_age = 0

    def update(self, raw: str) -> str:
        """Feed one raw decision, get the decision to act on.

        A parse failure resolves to SLOW_DOWN rather than propagating UNKNOWN:
        the output of this class is meant to be executed, and "UNKNOWN" is not a
        thing a controller can do.  SLOW_DOWN is the conservative reading at the
        same severity, so an unparseable frame slows the car instead of either
        stalling it or — worse — reading as permission to proceed.
        """
        if raw not in _SEVERITY or raw == 'UNKNOWN':
            raw = 'SLOW_DOWN'
        if self.held is None:
            self.held = raw
            self._stop_age = 0 if raw != 'STOP' else 1
            return self.held

        if _SEVERITY[raw] > _SEVERITY[self.held]:
            self.held = raw                      # escalate at once
            self._quieter_run = 0
            self._stop_age = 1 if raw == 'STOP' else 0
            return self.held

        if _SEVERITY[raw] == _SEVERITY[self.held]:
            self._quieter_run = 0
            if self.held == 'STOP':
                self._stop_age += 1
            return self.held

        # Quieter than what we are holding — require sustained agreement, and
        # never release a STOP before its latch has expired.
        self._quieter_run += 1
        if self.held == 'STOP':
            self._stop_age += 1
            if self._stop_age < self.stop_latch:
                return self.held
        if self._quieter_run >= self.release_frames:
            self.held = raw
            self._quieter_run = 0
            self._stop_age = 0
        return self.held

    def reset(self) -> None:
        """Clear between scenes — holding a stop across a cut is meaningless."""
        self.held = None
        self._quieter_run = 0
        self._stop_age = 0


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

    reasoning, decision, light = _parse_response(text)
    if not reasoning:
        reasoning = text.strip()

    draw.line([(0, y0), (W, y0)], fill=(60, 100, 100), width=1)
    draw.text((pad, y0 + 4), "REASONING", font=_FONT_BODY, fill=_WHITE)

    # Traffic-light chip, right-aligned on the header row.  This is the one piece
    # of information that only exists because the forward camera is in the payload,
    # so showing it is what makes the modality fix visible in the rendered output.
    if light in _LIGHT_COLORS:
        chip = f"LIGHT {light.upper()}"
        cw = int(draw.textlength(chip, font=_FONT_BODY))
        draw.text((W - pad - cw, y0 + 4), chip, font=_FONT_BODY,
                  fill=_LIGHT_COLORS[light])

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


# ── Metric state as text ───────────────────────────────────────────────────────

# Same PC_RANGE the BEV grid and the visualiser use — keep in sync or the text and
# the picture will disagree about where things are.
_PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


def _bearing_label(x: float, y: float) -> str:
    """Human-readable bearing in the ego frame (+x forward, +y left)."""
    deg = math.degrees(math.atan2(y, x))
    if -15 <= deg <= 15:
        return "ahead"
    if 15 < deg <= 60:
        return "front-left"
    if 60 < deg <= 120:
        return "left"
    if deg > 120:
        return "behind-left"
    if -60 <= deg < -15:
        return "front-right"
    if -120 <= deg < -60:
        return "right"
    return "behind-right"


def _format_detection_rows(items, max_rows: int = 5,
                           forward_only: bool = True) -> str:
    """Render (class, x, y, vx, vy, score) ego-frame tuples as the metric block.

    Shared by the predicted and ground-truth paths so an ablation between them
    measures DETECTION QUALITY and not text layout. Any change here lands on both
    arms simultaneously, which is the point.

    x forward, y left, metres; vx/vy in m/s in the same frame.
    """
    rows = []
    for name, x, y, vx, vy, score in items:
        rng = math.hypot(x, y)
        if rng < 1e-3 or rng > 60.0:
            continue
        # Drop anything behind the ego.  Two reasons, and the second is the one
        # that actually bit: objects to the rear cannot be a *blocking hazard
        # ahead*, and — measured on scene-0757 — a long detection list crowds the
        # camera out of the model's attention entirely.  With 8 rows (5 of them
        # "behind-*") it answered "no traffic lights visible" on a frame with a
        # plainly visible red light; trimmed to the forward hemisphere it reads
        # the light correctly.  Text tokens compete with visual ones.
        if forward_only and abs(math.degrees(math.atan2(y, x))) > 100.0:
            continue
        speed = math.hypot(vx, vy)
        # Radial component along the ego->object ray; negative means closing.
        v_radial = (vx * x + vy * y) / rng
        if speed < 0.5:
            motion = "stationary"
        elif v_radial < -0.5:
            motion = f"closing {abs(v_radial):.1f} m/s"
        elif v_radial > 0.5:
            motion = f"receding {v_radial:.1f} m/s"
        else:
            motion = f"crossing {speed:.1f} m/s"
        rows.append((rng, f"  {name:<12s} {rng:5.1f} m "
                          f"{_bearing_label(x, y):<12s} "
                          f"| {motion:<18s} | conf {score:.2f}"))
    if not rows:
        return "  (none above threshold)"
    rows.sort(key=lambda t: t[0])
    out = [line for _, line in rows[:max_rows]]
    if len(rows) > max_rows:
        out.append(f"  ... and {len(rows) - max_rows} more further away")
    return "\n".join(out)


def detections_to_text(cls_logits: torch.Tensor,
                       reg_preds: torch.Tensor,
                       ref_pts: torch.Tensor,
                       score_thr: float = 0.25,
                       lidar2ego_yaw: float = 0.0,
                       max_rows: int = 5,
                       forward_only: bool = True) -> str:
    """Compact metric summary of the detections, in the EGO frame.

    Mirrors `visualizer._draw_detections` exactly — same top-200-then-threshold
    selection, same log-encoded size decode, same reference-point denormalisation —
    so the text and the rendered canvas can never disagree.  The one deliberate
    difference is the frame: the canvas draws in global/north-up because it overlays
    the map, while the VLM needs ego-relative ("12 m ahead") to reason about *my*
    path.  Rotating by lidar2ego_yaw is what converts between them.

    Sorted by range so truncation at `max_rows` drops the least relevant objects.
    """
    scores, labels = cls_logits.float().sigmoid().max(-1)
    order = scores.argsort(descending=True)[:200]
    idxs = [int(i) for i in order if float(scores[i]) > score_thr]
    if not idxs:
        return "  (none above threshold)"

    return _format_detection_rows(
        detection_items(cls_logits, reg_preds, ref_pts, score_thr, lidar2ego_yaw),
        max_rows=max_rows, forward_only=forward_only)


def detection_items(cls_logits: torch.Tensor, reg_preds: torch.Tensor,
                    ref_pts: torch.Tensor, score_thr: float = 0.25,
                    lidar2ego_yaw: float = 0.0) -> list[tuple]:
    """Ego-frame (class, x, y, vx, vy, score) tuples behind `detections_to_text`.

    Split out so an ablation can replace one field — swapping GT velocities onto
    predicted positions, say — without duplicating the LiDAR->ego rotation and
    risking the two paths drifting apart.
    """
    scores, labels = cls_logits.float().sigmoid().max(-1)
    order = scores.argsort(descending=True)[:200]
    idxs = [int(i) for i in order if float(scores[i]) > score_thr]

    c_l2e, s_l2e = math.cos(lidar2ego_yaw), math.sin(lidar2ego_yaw)
    items = []
    for i in idxs:
        r, p = reg_preds[i].float(), ref_pts[i].float()
        x_lid = float(p[0]) * (_PC_RANGE[3] - _PC_RANGE[0]) + _PC_RANGE[0]
        y_lid = float(p[1]) * (_PC_RANGE[4] - _PC_RANGE[1]) + _PC_RANGE[1]
        vx_lid, vy_lid = float(r[8]), float(r[9])
        # LiDAR -> ego: +x forward, +y left.
        items.append((
            CLASS_NAMES[int(labels[i])],
            c_l2e * x_lid - s_l2e * y_lid,
            s_l2e * x_lid + c_l2e * y_lid,
            c_l2e * vx_lid - s_l2e * vy_lid,
            s_l2e * vx_lid + c_l2e * vy_lid,
            float(scores[i]),
        ))
    return items


# ── Front camera ───────────────────────────────────────────────────────────────

def front_camera_b64(nusc, sample_token: str, dataroot: str,
                     max_width: int = 640) -> str | None:
    """CAM_FRONT as a base64 JPEG, downscaled.

    Qwen2.5-VL uses dynamic resolution, so token count scales with pixel area —
    the image size is the single biggest lever on prefill latency.  1600x900 is far
    more than the task needs: traffic-light *state* is a coloured blob that survives
    aggressive downscaling, and brake lights and construction markings likewise.
    Sign *text* does not survive, but sign *presence* does.  640 px wide is roughly
    a 6x reduction in tokens against the native frame.
    """
    try:
        sd_token = nusc.get('sample', sample_token)['data']['CAM_FRONT']
        img_path = Path(dataroot) / nusc.get('sample_data', sd_token)['filename']
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > max_width:
            img = cv2.resize(img, (max_width, int(round(h * max_width / w))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode('utf-8')
    except Exception as exc:                      # missing file, bad token, ...
        print(f'[WARN] front camera unavailable ({exc}) — BEV-only this frame.')
        return None


def _query_ollama_streaming(b64, model: str, base_url: str,
                              timeout: int, on_update, update_every: int = 6,
                              prompt: str | None = None,
                              stats: dict | None = None,
                              temperature: float = 0.1,
                              seed: int | None = None,
                              ) -> str:
    """
    Stream from Ollama /api/generate.
    Calls on_update(accumulated_text) every *update_every* tokens and on done.
    Returns the full response string.

    `b64` accepts a single base64 image or a list of them (BEV first, then the
    forward camera).  `stats`, if given, is filled from the final chunk with the
    prefill/decode split — without it the per-frame latency is one opaque number
    and you cannot tell whether adding the second image cost anything, since
    image tokens land entirely in prefill.
    """
    images = [b64] if isinstance(b64, str) else list(b64)
    payload = json.dumps({
        "model":  model,
        # System text first, image payload after: Ollama caches the prompt prefix,
        # and the instructions are identical every frame while the images are not.
        "prompt": prompt if prompt is not None else _SYSTEM_PROMPT,
        "images": images,
        "stream": True,
        # temperature defaults to 0.1 for live/demo use, where a little variation
        # reads more naturally. ABLATIONS MUST PASS 0: at 0.1 the same prompt and
        # images return different decisions run to run, which silently turned an
        # A/B into a coin flip (an arm compared against ITSELF agreed 0/6 across
        # two runs). Measured: temperature 0 gives 3/3 identical responses.
        "options": ({"temperature": temperature, "num_predict": 140}
                    | ({"seed": seed} if seed is not None else {})),
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
                    if stats is not None:
                        # Ollama reports durations in nanoseconds.
                        stats.update(
                            prompt_tokens=obj.get("prompt_eval_count", 0),
                            eval_tokens=obj.get("eval_count", 0),
                            prefill_ms=obj.get("prompt_eval_duration", 0) / 1e6,
                            decode_ms=obj.get("eval_duration", 0) / 1e6,
                        )
                    break

    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            msg = json.loads(raw).get("error", raw)
        except Exception:
            msg = raw
        raise RuntimeError(f"HTTP {exc.code}: {msg}") from None

    return accumulated


def _parse_response(text: str) -> tuple[str, str, str]:
    """-> (reasoning, decision, light).

    `light` is the traffic-light state read from the forward camera, or "none".
    It is parsed as its own field rather than left inside the prose because it is
    the one claim that is directly checkable against a human label, which makes it
    the first thing in this stage that can actually be scored.
    """
    reasoning, decision, light = "", "UNKNOWN", "none"
    for line in text.strip().splitlines():
        s = line.strip()
        if s.startswith("REASONING:"):
            reasoning = s[len("REASONING:"):].strip()
        elif s.startswith("LIGHT:"):
            tok = s[len("LIGHT:"):].strip().lower().rstrip(".")
            light = next((c for c in _VALID_LIGHTS if c in tok), "none")
        elif s.startswith("DECISION:"):
            tok = s[len("DECISION:"):].strip().upper().rstrip(".")
            # Ordered longest-first so "SLOW_DOWN" is not shadowed by a substring,
            # and negations ("not STOP") do not silently match.
            decision = tok if tok in _VALID_DECISIONS else next(
                (d for d in sorted(_VALID_DECISIONS, key=len, reverse=True)
                 if d in tok), "UNKNOWN"
            )
    return reasoning, decision, light


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
    ap.add_argument('--front-cam', dest='front_cam', action='store_true', default=True,
                    help='Send CAM_FRONT as a second image (default on) — the only '
                         'source of traffic lights, brake lights and signage')
    ap.add_argument('--no-front-cam', dest='front_cam', action='store_false')
    ap.add_argument('--cam-width', type=int, default=640,
                    help='Downscale CAM_FRONT to this width before encoding. '
                         'Measured floor: 224/384/640 all cost ~2600 prompt '
                         'tokens; only native 1600 costs more (~3399). Below '
                         '640 buys nothing (default 640)')
    ap.add_argument('--det-text', dest='det_text', action='store_true', default=True,
                    help='Hand the decoder detections to the VLM as authoritative '
                         'metric text (default on)')
    ap.add_argument('--no-det-text', dest='det_text', action='store_false')
    ap.add_argument('--release-frames', type=int, default=2,
                    help='Consecutive quieter frames required before de-escalating '
                         'a decision (0 disables hysteresis)')
    ap.add_argument('--stop-latch', type=int, default=2,
                    help='Minimum frames a STOP is held before it can be released')
    ap.add_argument('--prev-context', dest='prev_context', action='store_true',
                    default=True, help='Put the previous decision in the prompt')
    ap.add_argument('--no-prev-context', dest='prev_context', action='store_false')
    ap.add_argument('--light-crop', dest='light_crop', action='store_true',
                    default=True,
                    help='Add a map-projected zoom on the traffic light to the '
                         'stage-1 query, alongside the wide frame (default on)')
    ap.add_argument('--no-light-crop', dest='light_crop', action='store_false')
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
    # Hysteresis state. release_frames=0 disables smoothing entirely, which is the
    # A/B baseline for measuring what the latch actually buys.
    smoother = DecisionSmoother(release_frames=args.release_frames,
                                stop_latch=args.stop_latch)
    prev_decision: dict | None = None
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
            reasoning = decision = raw_decision = ""
            light = "none"
            vl_ms = 0.0
            vl_stats: dict = {}
            if args.vl:
                def _on_token(text, _c=canvas, _t=trail_canvas, _g=cam_grid):
                    _save_bev_dual(_c, _t, vl_text=text, cam_grid=_g)

                # Geometry as text, semantics as a second image. The detector owns
                # metric state; the camera is the only channel carrying light state.
                det_text = detections_to_text(
                    out['cls_logits'][0].cpu(), out['reg_preds'][0].cpu(),
                    out['ref_pts'][0].cpu(), score_thr=args.score_thr,
                    lidar2ego_yaw=lidar2ego_yaw) if args.det_text else None

                payload_images = [_encode_image(out_path)]
                cam_b64 = (front_camera_b64(nusc, sample_token, args.dataroot,
                                            max_width=args.cam_width)
                           if args.front_cam else None)
                if cam_b64:
                    payload_images.append(cam_b64)

                t1 = time.monotonic()

                # Stage 1: camera alone reads the light. Must happen in its own
                # call — any detection text in context suppresses it entirely.
                light_pre = "none" if cam_b64 else None
                if cam_b64:
                    # Map-projected zoom on the signal head. The wide frame keeps
                    # lane context and the other semantics a BEV cannot carry
                    # (brake lights, construction, pedestrian intent); the crop
                    # restores the state that range destroys. Measured on
                    # scene-0757: 4/5 -> 5/5, recovering the 32 m miss.
                    crop_b64 = (light_crop_b64(nusc, sample_token, args.dataroot)
                                if args.light_crop else None)
                    try:
                        light_pre = query_light(cam_b64, args.ollama_model,
                                                args.ollama_url, args.ollama_timeout,
                                                vl_stats, crop_b64=crop_b64)
                    except Exception as exc:
                        print(f'[WARN] light query failed ({exc}) — assuming none.')

                # Stage 2: decision, with the light state supplied as text.
                prompt = build_prompt(det_text, with_front_cam=cam_b64 is not None,
                                      light_state=light_pre, prev=prev_decision)
                try:
                    raw = _query_ollama_streaming(
                        payload_images,
                        args.ollama_model, args.ollama_url,
                        args.ollama_timeout, _on_token,
                        update_every=args.update_every,
                        prompt=prompt, stats=vl_stats,
                    )
                    reasoning, decision, light = _parse_response(raw)
                    # Stage 1 saw the camera with nothing competing for
                    # attention, so it outranks stage 2's echo.
                    if light_pre is not None:
                        light = light_pre
                    raw_decision = decision
                    # Hysteresis is applied AFTER logging the raw value, so the
                    # smoother's effect stays measurable rather than hidden.
                    decision = smoother.update(decision)
                    prev_decision = {'decision': decision, 'light': light,
                                     'reasoning': reasoning}
                    _save_bev_dual(canvas, trail_canvas, vl_text=raw, cam_grid=cam_grid)
                except urllib.error.URLError as exc:
                    raw = f"[OLLAMA UNREACHABLE: {exc.reason}]"
                    _save_bev_dual(canvas, trail_canvas, vl_text=raw, cam_grid=cam_grid)
                except Exception as exc:
                    raw = f"[VLM ERROR: {exc}]"
                    _save_bev_dual(canvas, trail_canvas, vl_text=raw, cam_grid=cam_grid)
                vl_ms = (time.monotonic() - t1) * 1000

            # ── Console ────────────────────────────────────────────────────────
            split = ''
            if vl_stats:
                split = (f' (prefill {vl_stats["prefill_ms"]:.0f} ms /'
                         f' {vl_stats["prompt_tokens"]} tok,'
                         f' decode {vl_stats["decode_ms"]:.0f} ms)')
            print(f'  frame {frame_idx:3d} | bev {bev_ms:6.1f} ms'
                  + (f' | vl {vl_ms:5.0f} ms{split}' if args.vl else '')
                  + f' | {_top3_str(out["cls_logits"][0])}'
                  + f' | {sample_token[:8]}')
            if reasoning:
                print(f'           reasoning : {reasoning}')
            if decision:
                label = _LABELS.get(decision, decision)
                held = ('' if raw_decision == decision
                        else f'   [raw {raw_decision} -> held {decision}]')
                print(f'           decision  : {label}   light: {light}{held}')

            if log_fh and args.vl:
                log_fh.write(json.dumps({
                    "frame": frame_idx, "token": sample_token,
                    "decision": decision, "raw_decision": raw_decision,
                    "reasoning": reasoning, "light": light,
                    "bev_ms": round(bev_ms, 1), "vl_ms": round(vl_ms, 1),
                    "prefill_ms": round(vl_stats.get("prefill_ms", 0.0), 1),
                    "decode_ms": round(vl_stats.get("decode_ms", 0.0), 1),
                    "prompt_tokens": vl_stats.get("prompt_tokens", 0),
                    "front_cam": bool(args.front_cam and cam_b64),
                    "light_crop": bool(args.light_crop and cam_b64 and crop_b64),
                    "det_text": bool(args.det_text),
                }) + '\n')
                log_fh.flush()

            frame_idx += 1

    if log_fh:
        log_fh.close()
    print(f'\n[DONE] {frame_idx} frames → {OUT_DIR}/')


if __name__ == '__main__':
    main()
