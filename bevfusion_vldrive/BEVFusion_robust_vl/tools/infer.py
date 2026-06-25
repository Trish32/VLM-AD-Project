"""End-to-end inference for the BEVFusion-PP port on nuScenes mini."""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from model.bevfusion_pp import BEVF_FasterRCNN
from model.checkpoint import load_bevfusion_pp
from data.loader import NuScenesPPLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=C.CHECKPOINT)
    ap.add_argument('--dataroot', default=C.DATAROOT)
    ap.add_argument('--split', default='mini_val')
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    model = BEVF_FasterRCNN(C, device=device)
    load_bevfusion_pp(model, args.checkpoint, map_location="cpu")
    model.to(device).eval()

    loader = NuScenesPPLoader(args.dataroot, C.VERSION)
    tokens = loader.sample_tokens(args.split)
    print(f"{len(tokens)} samples in {args.split}")
    token = tokens[args.frame]
    frame = loader.get_frame(token)

    points = [frame['points'].to(device)]
    img = frame['img'].to(device)
    lidar2img = [frame['lidar2img']]

    t0 = time.time()
    bboxes, scores, labels = model.simple_test(points, img, lidar2img)
    dt = time.time() - t0
    print(f"sample {token}  ({dt:.2f}s)")
    print(f"  detections: {bboxes.shape[0]}")
    if scores.numel():
        print(f"  score range: {float(scores.min()):.3f} .. {float(scores.max()):.3f}")
        keep = scores > 0.3
        print(f"  score>0.3  : {int(keep.sum())}")
        for c in range(C.NUM_CLASSES):
            n = int((labels[keep] == c).sum())
            if n:
                print(f"    {C.CLASS_NAMES[c]:20s}: {n}")
        # show a few high-score boxes
        order = torch.argsort(scores, descending=True)[:5]
        print("  top-5 boxes [x y z w l h yaw vx vy] score label:")
        for i in order.tolist():
            b = bboxes[i].tolist()
            print("   ", " ".join(f"{v:6.2f}" for v in b),
                  f"{float(scores[i]):.3f}", C.CLASS_NAMES[int(labels[i])])


if __name__ == "__main__":
    main()
