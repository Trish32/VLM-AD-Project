"""Visualise MIT-BEVFusion detections → GIF: 6 surround cameras (with projected
3-D boxes) + a LiDAR-frame BEV (points + boxes), over one nuScenes-mini scene.

Example:
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
        python visualize.py --max-frames 15 --device cpu
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
from model.bevfusion import BEVFusion
from data.loader import NuScenesMITLoader


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=C.CKPT_DET)
    ap.add_argument('--dataroot', default=C.DATAROOT)
    ap.add_argument('--split', default='mini_val')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--max-frames', type=int, default=15)
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--out', default='viz_out')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    print("device:", args.device)

    model = BEVFusion(C.DetCfg)
    info = model.load_state_dict(
        torch.load(args.checkpoint, map_location='cpu')['state_dict'], strict=False)
    print(f"[checkpoint] missing {len(info.missing_keys)}  unexpected {len(info.unexpected_keys)}")
    model.to(args.device).eval()

    ld = NuScenesMITLoader(args.dataroot)
    nusc = ld.nusc
    tokens = ld.sample_tokens(args.split)[args.start:args.start + args.max_frames]
    print(f"{len(tokens)} frames")

    os.makedirs(args.out, exist_ok=True)
    frames = []
    for i, tok in enumerate(tokens):
        frame = ld.get_frame(tok)
        boxes, scores, labels = model(frame)[0]
        comp = bev_viz.composite(nusc, tok, args.dataroot, frame['points'],
                                 boxes, scores, labels, C.DetCfg.POINT_CLOUD_RANGE,
                                 C.OBJECT_CLASSES, "BEVFusion (MIT)", args.score_thr)
        frames.append(comp)
        n = int((scores >= args.score_thr).sum()) if scores.numel() else 0
        print(f"  frame {i:02d}: {n} boxes (>= {args.score_thr})")

    gif = os.path.join(args.out, "bevfusion_mit_scene.gif")
    bev_viz.make_gif(frames, gif)
    print(f"wrote {gif}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
