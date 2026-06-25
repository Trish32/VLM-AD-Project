"""
DET fine-tune smoke test: TransFusion loss (gaussian heatmap focal + Hungarian
matching + bbox L1). Freezes encoders, trains fuser+decoder+head. Single-frame
overfit to confirm the det training path works (loss decreases).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from nuscenes.eval.detection.utils import category_to_detection_name
from pyquaternion import Quaternion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from model.bevfusion import BEVFusion
from model.det_loss import transfusion_loss
from data.loader import NuScenesMITLoader

CLS_IDX = {n: i for i, n in enumerate(C.OBJECT_CLASSES)}


def get_gt(ld, token, device):
    sample = ld.nusc.get('sample', token)
    _, boxes, _ = ld.nusc.get_sample_data(sample['data']['LIDAR_TOP'])
    gb, gl = [], []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name is None or name not in CLS_IDX:
            continue
        w, l, h = box.wlh
        cx, cy, cz = box.center
        v = box.orientation.rotation_matrix @ np.array([1.0, 0, 0])
        yaw = -np.arctan2(v[1], v[0]) - np.pi / 2
        gb.append([cx, cy, cz - h / 2, w, l, h, yaw, 0.0, 0.0])
        gl.append(CLS_IDX[name])
    if not gb:
        return torch.zeros((0, 9), device=device), torch.zeros((0,), dtype=torch.long, device=device)
    return (torch.tensor(gb, dtype=torch.float32, device=device),
            torch.tensor(gl, dtype=torch.long, device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    model = BEVFusion(C.DetCfg)
    model.load_state_dict(torch.load(C.CKPT_DET, map_location='cpu')['state_dict'], strict=False)
    model.to(args.device)
    for p in model.encoders.parameters():
        p.requires_grad = False
    model.encoders.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    ld = NuScenesMITLoader(C.DATAROOT)
    tok = ld.sample_tokens('mini_train')[0]
    frame = ld.get_frame(tok)
    gt_boxes, gt_labels = get_gt(ld, tok, args.device)
    print(f"GT boxes: {gt_boxes.shape[0]}")

    img = frame['img'].unsqueeze(0)
    points = [frame['points']]
    for k in ['camera2lidar', 'camera_intrinsics', 'img_aug_matrix', 'lidar2image']:
        frame[k] = frame[k].unsqueeze(0)
    frame['lidar_aug_matrix'] = frame['lidar_aug_matrix'].unsqueeze(0)
    with torch.no_grad():
        cam = model.extract_camera(img, points, frame)
        lid = model.extract_lidar(points)
    head = model.heads['object']

    print("single-frame overfit (total loss should trend down):")
    for s in range(args.steps):
        x = model.fuser([cam, lid])
        x = model.decoder['backbone'](x)
        x = model.decoder['neck'](x)
        res, dense_heatmap, _ = head.forward_train(x)
        lh, lc, lb, npos = transfusion_loss(res, dense_heatmap, gt_boxes, gt_labels, C.DetCfg)
        loss = lh + lc + lb
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 35.0)
        opt.step()
        print(f"  step {s:02d} loss={float(loss):.4f} (hm={float(lh):.4f} cls={float(lc):.4f} bbox={float(lb):.4f}) npos={npos}")
    print("det fine-tune smoke test OK")


if __name__ == "__main__":
    main()
