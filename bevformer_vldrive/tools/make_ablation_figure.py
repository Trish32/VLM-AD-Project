#!/usr/bin/env python3
"""Render the red-light ablation figure used in the README.

One frame, four input configurations, side by side with what each answered. The
frame is scene-0757 #0 — a red light with a completely empty road ahead, which is
the case a BEV-only planner cannot get right even in principle.

Usage:
    conda run -n simple_bev_vldrive python tools/make_ablation_figure.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from nuscenes import NuScenes
from PIL import Image, ImageDraw

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from vis_infer import _FONT_BODY, _load_fonts

DATAROOT = '/Users/trish/Downloads/nuScenes_miniV1.0'
OUT = ROOT / 'bev_outputs' / 'redlight_ablation.png'
SCENE, FRAME = 'scene-0757', 0

_FONT_TITLE, _ = _load_fonts(size_body=15, size_title=20)

# Measured on 2026-09-22; see README "Red-light test".
CASES = [
    ("BEV only", "no camera in payload",
     "SLOW_DOWN", "none", (220, 220, 50),
     "Cannot see the light — not in its input at all."),
    ("BEV + camera + detections\n(single call)", "one call, 8 detection rows",
     "PROCEED", "none", (80, 220, 80),
     "Camera IS in the payload, but the detection\ntext suppresses it: \"no traffic lights visible\"."),
    ("BEV + camera, detections removed", "one call, 0 detection rows",
     "STOP", "red", (220, 50, 50),
     "Same images, text removed -> reads the light."),
    ("Two-stage (shipped)", "light query, then decision",
     "STOP", "red", (220, 50, 50),
     "Stage 1 reads the camera alone; stage 2 gets\nthe state as text. Both jobs succeed."),
]

PANEL_W, PANEL_H = 430, 300
TEXT_H = 150


def main() -> None:
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    sc = [s for s in nusc.scene if s['name'] == SCENE][0]
    tok = sc['first_sample_token']
    for _ in range(FRAME):
        tok = nusc.get('sample', tok)['next']
    sd = nusc.get('sample_data', nusc.get('sample', tok)['data']['CAM_FRONT'])
    cam = cv2.cvtColor(cv2.imread(str(Path(DATAROOT) / sd['filename'])),
                       cv2.COLOR_BGR2RGB)

    bev_path = ROOT / 'bev_outputs' / 'vis_000.png'
    bev = (cv2.cvtColor(cv2.imread(str(bev_path)), cv2.COLOR_BGR2RGB)
           if bev_path.exists() else np.full((512, 512, 3), 20, np.uint8))

    cam_small = cv2.resize(cam, (PANEL_W, int(PANEL_W * 9 / 16)))
    bev_small = cv2.resize(bev, (PANEL_W, int(PANEL_W * 9 / 16)))

    cols = []
    for title, sub, decision, light, colour, note in CASES:
        panel = np.full((PANEL_H + TEXT_H, PANEL_W, 3), 18, np.uint8)
        shows_cam = "camera" in title.lower() or "two-stage" in title.lower()
        img = cam_small if shows_cam else bev_small
        panel[34:34 + img.shape[0], :, :] = img
        if not shows_cam:
            # Make the absence explicit rather than implied by a missing tile.
            cv2.putText(panel, "(camera not sent)", (10, 34 + img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        pil = Image.fromarray(panel)
        d = ImageDraw.Draw(pil)
        for i, line in enumerate(title.split('\n')):
            d.text((8, 4 + i * 15), line, font=_FONT_BODY, fill=(235, 235, 235))
        y = 34 + img.shape[0] + 12
        d.text((8, y), sub, font=_FONT_BODY, fill=(140, 155, 155))
        y += 24
        d.rectangle([(8, y), (PANEL_W - 8, y + 26)],
                    fill=tuple(max(0, c // 5) for c in colour), outline=colour, width=2)
        d.text((14, y + 5), f"LIGHT: {light}   ->   {decision}",
               font=_FONT_BODY, fill=colour)
        y += 36
        for line in note.split('\n'):
            d.text((8, y), line, font=_FONT_BODY, fill=(175, 185, 185))
            y += 16
        cols.append(np.array(pil))

    sep = np.full((cols[0].shape[0], 3, 3), 60, np.uint8)
    body = cols[0]
    for c in cols[1:]:
        body = np.hstack([body, sep, c])

    header = np.full((44, body.shape[1], 3), 12, np.uint8)
    pil = Image.fromarray(header)
    ImageDraw.Draw(pil).text(
        (10, 12),
        "scene-0757 frame 0 — red light, empty road ahead.  "
        "Same frame, four input configurations.",
        font=_FONT_TITLE, fill=(235, 235, 235))
    out = np.vstack([np.array(pil), body])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f'[DONE] {out.shape} -> {OUT}')


if __name__ == '__main__':
    main()
