"""Visualise BEVFusion-PP detections → GIF: 6 surround cameras (with projected
3-D boxes) + a LiDAR-frame BEV (points + boxes), over one nuScenes-mini scene.

Example:
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
        python visualize.py --max-frames 15
"""
import argparse
import os
import sys

import torch

_PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))           # parent → bev_viz
import config as C
import bev_viz
from model.bevfusion_pp import BEVF_FasterRCNN
from model.checkpoint import load_bevfusion_pp
from data.loader import NuScenesPPLoader


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=C.CHECKPOINT)
    ap.add_argument('--dataroot', default=C.DATAROOT)
    ap.add_argument('--split', default='mini_val')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--max-frames', type=int, default=15)
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--out', default='viz_out')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    model = BEVF_FasterRCNN(C, device=device)
    load_bevfusion_pp(model, args.checkpoint, map_location="cpu")
    model.to(device).eval()

    loader = NuScenesPPLoader(args.dataroot, C.VERSION)
    nusc = loader.nusc
    tokens = loader.sample_tokens(args.split)[args.start:args.start + args.max_frames]
    print(f"{len(tokens)} frames")

    os.makedirs(args.out, exist_ok=True)
    frames = []
    for i, tok in enumerate(tokens):
        fr = loader.get_frame(tok)
        bboxes, scores, labels = model.simple_test(
            [fr['points'].to(device)], fr['img'].to(device), [fr['lidar2img']])
        comp = bev_viz.composite(nusc, tok, args.dataroot, fr['points'],
                                 bboxes, scores, labels, C.POINT_CLOUD_RANGE,
                                 C.CLASS_NAMES, "BEVFusion-PP", args.score_thr)
        frames.append(comp)
        n = int((scores >= args.score_thr).sum()) if scores.numel() else 0
        print(f"  frame {i:02d}: {n} boxes (>= {args.score_thr})")

    gif = os.path.join(args.out, "bevfusion_pp_scene.gif")
    bev_viz.make_gif(frames, gif)
    print(f"wrote {gif}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
