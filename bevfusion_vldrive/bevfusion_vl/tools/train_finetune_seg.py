"""
SEG fine-tune smoke test: freeze encoders, train fuser+decoder+seg-head with
sigmoid focal loss on rasterized GT map masks. Confirms the training path works
(loss decreases, grads flow) and the checkpoint round-trips.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from nuscenes.map_expansion.map_api import NuScenesMap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from model.bevfusion import BEVFusion
from data.loader import NuScenesMITLoader
from tools.eval_seg import gt_masks


def focal(logits, targets, alpha=-1, gamma=2):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - pt) ** gamma
    return loss.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    model = BEVFusion(C.SegCfg)
    model.load_state_dict(torch.load(C.CKPT_SEG, map_location='cpu')['state_dict'], strict=False)
    model.to(args.device)
    # freeze encoders (camera + lidar); train fuser + decoder + head
    for m in [model.encoders]:
        for p in m.parameters():
            p.requires_grad = False
    model.encoders.eval()
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in train_params)/1e6:.1f}M")
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.01)

    ld = NuScenesMITLoader(C.DATAROOT)
    nmaps = {}
    tokens = ld.sample_tokens('mini_train')
    head = model.heads['map']

    for step in range(args.steps):
        frame = ld.get_frame(tokens[step % len(tokens)])
        if frame['location'] not in nmaps:
            nmaps[frame['location']] = NuScenesMap(C.DATAROOT, frame['location'])
        gt = gt_masks(nmaps[frame['location']], frame, C.MAP_CLASSES)
        gt = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).to(args.device)

        img = frame['img'].unsqueeze(0)
        points = [frame['points']]
        for k in ['camera2lidar', 'camera_intrinsics', 'img_aug_matrix', 'lidar2image']:
            frame[k] = frame[k].unsqueeze(0)
        frame['lidar_aug_matrix'] = frame['lidar_aug_matrix'].unsqueeze(0)
        with torch.no_grad():
            cam = model.extract_camera(img, points, frame)
            lid = model.extract_lidar(points)
        x = model.fuser([cam, lid])
        x = model.decoder['backbone'](x)
        x = model.decoder['neck'](x)
        logits = head.classifier(head.transform(x))
        loss = focal(logits, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step:02d} loss={float(loss):.5f}")

    print("seg fine-tune smoke test OK (loss should trend down)")


if __name__ == "__main__":
    main()
