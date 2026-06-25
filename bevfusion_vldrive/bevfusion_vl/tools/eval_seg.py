"""
BEV map segmentation eval for MIT BEVFusion port on nuScenes mini_val.
Runs the model, rasterizes GT map masks via NuScenesMap, thresholds at 0.5,
and reports per-class IoU + mIoU (official protocol).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from nuscenes.map_expansion.map_api import NuScenesMap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from model.bevfusion import BEVFusion
from data.loader import NuScenesMITLoader

XBOUND = [-50.0, 50.0, 0.5]
YBOUND = [-50.0, 50.0, 0.5]
CANVAS = (int((YBOUND[1] - YBOUND[0]) / YBOUND[2]),
          int((XBOUND[1] - XBOUND[0]) / XBOUND[2]))  # (200,200)
PATCH = (YBOUND[1] - YBOUND[0], XBOUND[1] - XBOUND[0])


def gt_masks(nmap, frame, classes):
    lidar2global = frame['ego2global'].numpy() @ frame['lidar2ego'].numpy()  # point2lidar=I
    pose = lidar2global[:2, 3]
    patch_box = (pose[0], pose[1], PATCH[0], PATCH[1])
    rot = lidar2global[:3, :3]
    v = rot @ np.array([1.0, 0, 0])
    patch_angle = np.arctan2(v[1], v[0]) / np.pi * 180
    mappings = {}
    for name in classes:
        if name == 'divider':
            mappings[name] = ['road_divider', 'lane_divider']
        elif name == 'drivable_area':
            mappings[name] = ['drivable_area']
        else:
            mappings[name] = [name]
    layer_names = list(set(l for v in mappings.values() for l in v))
    masks = nmap.get_map_mask(patch_box=patch_box, patch_angle=patch_angle,
                              layer_names=layer_names, canvas_size=CANVAS)
    masks = masks.transpose(0, 2, 1).astype(bool)
    labels = np.zeros((len(classes), *CANVAS), dtype=bool)
    for k, name in enumerate(classes):
        for ln in mappings[name]:
            labels[k, masks[layer_names.index(ln)]] = True
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    model = BEVFusion(C.SegCfg)
    ck = torch.load(C.CKPT_SEG, map_location='cpu', weights_only=False)['state_dict']
    info = model.load_state_dict(ck, strict=False)
    print("load missing/unexpected:", len(info.missing_keys), len(info.unexpected_keys))
    model.to(args.device).eval()

    ld = NuScenesMITLoader(C.DATAROOT)
    nmaps = {loc: NuScenesMap(C.DATAROOT, loc) for loc in
             ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']}
    tokens = ld.sample_tokens('mini_val')
    if args.limit:
        tokens = tokens[:args.limit]

    classes = C.MAP_CLASSES
    nc = len(classes)
    thresholds = np.array([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    nt = len(thresholds)
    # accumulate tp/fp/fn per (class, threshold) for identity & flipped GT (A/B)
    stats = {v: [np.zeros((nc, nt)) for _ in range(3)] for v in ['id', 'gtflipH', 'gtflipW']}

    skipped = 0
    for idx, tok in enumerate(tokens):
        frame = ld.get_frame(tok)
        try:
            gt0 = gt_masks(nmaps[frame['location']], frame, classes)
        except Exception as e:
            skipped += 1
            print(f"  skip {idx} (GT gen failed: {type(e).__name__})", flush=True)
            continue
        with torch.no_grad():
            prob = model(frame)[0].cpu().numpy()   # (6,200,200) sigmoid
        pred = prob[:, :, :, None] >= thresholds   # (6,200,200,nt)
        lab = gt0[:, :, :, None]
        tp, fp, fn = stats['id']
        tp += (pred & lab).reshape(nc, -1, nt).sum(1)
        fp += (pred & ~lab).reshape(nc, -1, nt).sum(1)
        fn += (~pred & lab).reshape(nc, -1, nt).sum(1)
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(tokens)}]", flush=True)
    print(f"evaluated {len(tokens)-skipped}/{len(tokens)} frames ({skipped} skipped)")

    for vname in ['id']:
        tp, fp, fn = stats[vname]
        ious = tp / (tp + fp + fn + 1e-7)           # (nc, nt)
        iou_max = ious.max(axis=1)                  # per class
        tag = 'GT identity'
        print(f"\n=== {tag}: per-class iou@max ===")
        for k, name in enumerate(classes):
            print(f"  {name:16s} {iou_max[k]*100:.2f}")
        print(f"  --> mIoU(iou@max): {iou_max.mean()*100:.2f}")


if __name__ == "__main__":
    main()
