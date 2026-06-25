"""
Fine-tune smoke test for the BEVFusion-PP port: freeze the camera stream, train
the LiDAR stream + fusion + head on nuScenes mini for a few steps, and confirm
the loss decreases. Saves a checkpoint that can be re-evaluated.

Run with PYTORCH_ENABLE_MPS_FALLBACK=1 (some 3D ops lack MPS kernels).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from model.bevfusion_pp import BEVF_FasterRCNN
from model.checkpoint import load_bevfusion_pp
from model.losses import (assign_targets, sigmoid_focal_loss, smooth_l1,
                          add_sin_difference)
from data.finetune_loader import NuScenesPPFinetuneLoader


def freeze_camera(model):
    for mod in [model.img_backbone, model.img_neck, model.lift_splat_shot_vis]:
        for p in mod.parameters():
            p.requires_grad = False
    model.img_backbone.eval()
    model.img_neck.eval()
    model.lift_splat_shot_vis.eval()


def compute_loss(model, feats, gt_boxes, gt_labels, device):
    head = model.pts_bbox_head
    cls, reg, dir_ = head(feats[0])
    B = cls.shape[0]
    anchors = head.anchor_generator.grid_anchors(cls.shape[-2:], device)
    cls = cls[0].permute(1, 2, 0).reshape(-1, head.num_classes)
    reg = reg[0].permute(1, 2, 0).reshape(-1, head.box_code_size)
    dir_ = dir_[0].permute(1, 2, 0).reshape(-1, 2)

    labels, lw, bt, bw, dt, dw, pos_inds = assign_targets(
        anchors, gt_boxes.to(device), gt_labels.to(device), head.num_classes,
        C.TRAIN_CFG['pos_iou_thr'], C.TRAIN_CFG['neg_iou_thr'], C.TRAIN_CFG['min_pos_iou'])

    num_pos = max(pos_inds.numel(), 1)
    loss_cls = sigmoid_focal_loss(cls, labels, lw, head.num_classes, avg_factor=num_pos)

    if pos_inds.numel() > 0:
        code_w = torch.tensor(C.TRAIN_CFG['code_weight'], device=device)
        pos_reg = reg[pos_inds]
        pos_bt = bt[pos_inds]
        pos_bw = bw[pos_inds] * code_w
        pos_reg, pos_bt = add_sin_difference(pos_reg, pos_bt)
        loss_bbox = smooth_l1(pos_reg, pos_bt, pos_bw, avg_factor=num_pos)
        loss_dir = (torch.nn.functional.cross_entropy(
            dir_[pos_inds], dt[pos_inds], reduction='none') * dw[pos_inds]).sum() / num_pos * 0.2
    else:
        loss_bbox = reg.sum() * 0
        loss_dir = dir_.sum() * 0
    return loss_cls, loss_bbox, loss_dir, num_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=C.CHECKPOINT)
    ap.add_argument('--dataroot', default=C.DATAROOT)
    ap.add_argument('--split', default='mini_train')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--save', default='/Users/trish/VLMProjects/bevfusion_vldrive/BEVFusion_vl/model/checkpoints/finetune_pp.pth')
    args = ap.parse_args()

    device = torch.device(args.device)
    model = BEVF_FasterRCNN(C, device=device)
    load_bevfusion_pp(model, args.checkpoint, map_location="cpu")
    model.to(device)
    model.train()
    freeze_camera(model)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)

    loader = NuScenesPPFinetuneLoader(args.dataroot, C.VERSION)
    tokens = loader.sample_tokens(args.split)
    print(f"{len(tokens)} train samples")

    for step in range(args.steps):
        token = tokens[step % len(tokens)]
        frame = loader.get_train_frame(token)
        if frame['gt_boxes'].shape[0] == 0:
            continue
        points = [frame['points'].to(device)]
        img = frame['img'].to(device)
        l2i = [frame['lidar2img']]

        feats, _ = model.extract_feat(points, img, l2i)
        lc, lb, ld, npos = compute_loss(model, feats, frame['gt_boxes'],
                                        frame['gt_labels'], device)
        loss = lc + lb + ld
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 35.0)
        opt.step()
        print(f"step {step:02d} npos={npos:4d} loss={float(loss):.3f} "
              f"(cls={float(lc):.3f} bbox={float(lb):.3f} dir={float(ld):.3f})")

    torch.save({'state_dict': model.state_dict()}, args.save)
    print("saved", args.save)


if __name__ == "__main__":
    main()
