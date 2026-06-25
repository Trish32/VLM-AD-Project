"""
Full training of Sparse4D-v2 on nuScenes mini (Apple MPS) with the backbone
UNFROZEN and DENSE DEPTH SUPERVISION enabled.

Difference from train_finetune.py
----------------------------------
  train_finetune.py : freezes ResNet-50 + FPN, trains the head only, low LR.
  train_v2.py       : trains the backbone too (except stem + layer1, which stay
                      frozen per the Sparse4D frozen_stages=1 config; BN stays in
                      eval mode per norm_eval=True), and adds an auxiliary dense
                      depth branch supervised by projected LiDAR points.

Total loss = detection (per-stage Hungarian) + depth_weight * masked-L1 depth.

The depth branch is a TRAINING-ONLY crutch that shapes the backbone's geometric
features; it is discarded at eval (tools/eval.py builds the model without it).

Usage
-----
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
        python train_v2.py \
        --dataroot /Users/trish/Downloads/nuScenes_miniV1.0 \
        --checkpoint sparse4d_vl/model/checkpoints/sparse4dv2_r50_HInf_256x704.pth \
        --epochs 6 --lr 2e-5 --depth_weight 0.05 \
        --save_dir checkpoints/train_v2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from sparse4d_vl.data.finetune_loader import NuScenesFinetuneLoader
from sparse4d_vl.data.lidar_depth     import depth_targets_for_sample
from sparse4d_vl.model.sparse4d_v2    import Sparse4Dv2
from sparse4d_vl.model.checkpoint     import load_checkpoint
from sparse4d_vl.model.loss           import Sparse4DLoss
from sparse4d_vl.model.depth_head     import depth_l1_loss


def build_model(checkpoint_path: str, device: torch.device) -> Sparse4Dv2:
    """Sparse4Dv2 with the depth branch; backbone trainable except frozen_stages."""
    model = Sparse4Dv2(with_depth=True).to(device)
    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint(model, checkpoint_path, version='v2')
        print(f"[init] Checkpoint loaded: {checkpoint_path}")
    else:
        print("[init] No checkpoint — training from ImageNet backbone only")

    # NOTE: we do NOT freeze the backbone here. ResNet50 already keeps stem +
    # layer1 frozen (frozen_stages=1) and all BN in eval (norm_eval=True) via
    # its own .train() override — matching the official Sparse4D config.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[init] Params — total: {total/1e6:.1f}M  trainable: {trainable/1e6:.1f}M  "
          f"(stem+layer1 + all BN frozen)")
    return model


def build_optimizer(model, lr, weight_decay, total_steps, warmup_steps, backbone_lr_mult):
    """Lower LR for the backbone (it is converged) than for the head/depth."""
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith('backbone.') else other_params).append(p)

    optimizer = torch.optim.AdamW(
        [
            {'params': other_params,    'lr': lr},
            {'params': backbone_params, 'lr': lr * backbone_lr_mult},
        ],
        weight_decay=weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return float(0.5 * (1.0 + np.cos(progress * np.pi)))   # python float (not np scalar)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def save_checkpoint(path, model, optimizer, epoch, step, loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_state = {k: v for k, v in model.state_dict().items() if '_cached_' not in k}
    torch.save({'epoch': epoch, 'step': step, 'loss': loss,
                'model': model_state, 'optimizer': optimizer.state_dict()}, path)
    print(f"[ckpt] Saved → {path}")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    epoch, global_step, depth_weight, log_every, grad_clip):
    model.train()
    model.backbone.eval()   # keep BN frozen even though backbone params train

    n_cam = model.NUM_CAMS
    total_loss = total_det = total_dep = 0.0
    n_frames = 0
    t0 = time.time()

    for frame in loader:
        if frame['is_first_frame']:
            model.reset_state()

        imgs      = frame['imgs'].to(device)
        img_metas = frame['img_metas']
        gt_boxes  = frame['gt_boxes'].to(device)
        gt_labels = frame['gt_labels'].to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(imgs, img_metas)

        # ---- Detection loss (per-stage Hungarian) ----
        det_loss = criterion.forward_multi(output['stage_preds'], gt_boxes, gt_labels)

        # ---- Depth loss (build GT at the head's output resolution) ----
        dep_loss = imgs.sum() * 0.0
        if 'depth_pred' in output:
            H_f, W_f = output['depth_pred'].shape[-2:]
            depth_gt, depth_mask = depth_targets_for_sample(
                loader.nusc, img_metas['sample_token'],
                img_metas['projection_mat'], img_metas['img_wh'], (H_f, W_f),
            )
            dep_loss = depth_l1_loss(
                output['depth_pred'], depth_gt.to(device), depth_mask.to(device)
            )

        loss = det_loss + depth_weight * dep_loss
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        if device.type == 'mps':
            torch.mps.empty_cache()

        total_loss += loss.item(); total_det += det_loss.item(); total_dep += float(dep_loss)
        n_frames += 1; global_step += 1

        if global_step % log_every == 0:
            lr_now = scheduler.get_last_lr()[0]
            fps = n_frames / max(time.time() - t0, 1e-6)
            print(f"  epoch {epoch:02d} | step {global_step:05d} | loss {loss.item():.4f} "
                  f"(det {det_loss.item():.4f}  depth {float(dep_loss):.4f}) "
                  f"| lr {lr_now:.2e} | {fps:.2f} fr/s")

    return total_loss / max(n_frames, 1), global_step


def parse_args():
    p = argparse.ArgumentParser(description='Train Sparse4Dv2 (unfrozen backbone + depth)')
    p.add_argument('--dataroot',   default='/Users/trish/Downloads/nuScenes_miniV1.0')
    p.add_argument('--checkpoint', default='sparse4d_vl/model/checkpoints/sparse4dv2_r50_HInf_256x704.pth')
    p.add_argument('--save_dir',   default='checkpoints/train_v2')
    p.add_argument('--resume',     default=None)
    p.add_argument('--epochs',     type=int,   default=6)
    p.add_argument('--lr',         type=float, default=2e-5)
    p.add_argument('--backbone_lr_mult', type=float, default=0.1,
                   help='Backbone LR = lr * this (backbone is already converged)')
    p.add_argument('--weight_decay',  type=float, default=0.01)
    p.add_argument('--warmup_epochs', type=float, default=0.5)
    p.add_argument('--depth_weight',  type=float, default=0.05)
    p.add_argument('--grad_clip',  type=float, default=35.0)
    p.add_argument('--log_every',  type=int,   default=20)
    p.add_argument('--version',    default='v1.0-mini')
    return p.parse_args()


def main():
    args = parse_args()
    device = (torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"[init] Device: {device}")

    loader = NuScenesFinetuneLoader(dataroot=args.dataroot, version=args.version)
    total_frames = len(loader)
    print(f"[init] Dataset: {total_frames} keyframes across {len(loader.nusc.scene)} scenes")

    warmup_steps = int(args.warmup_epochs * total_frames)
    total_steps  = args.epochs * total_frames

    model = build_model(args.checkpoint, device)
    optimizer, scheduler = build_optimizer(
        model, args.lr, args.weight_decay, total_steps, warmup_steps, args.backbone_lr_mult)
    criterion = Sparse4DLoss()

    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, global_step = ckpt['epoch'] + 1, ckpt['step']
        print(f"[resume] from epoch {ckpt['epoch']}")

    print(f"\n[train] {args.epochs} epochs ({total_steps} steps), depth_weight={args.depth_weight}\n")
    for epoch in range(start_epoch, args.epochs):
        print(f"{'='*60}\nEpoch {epoch:02d} / {args.epochs - 1}\n{'='*60}")
        mean_loss, global_step = train_one_epoch(
            model, loader, criterion, optimizer, scheduler, device,
            epoch, global_step, args.depth_weight, args.log_every, args.grad_clip)
        print(f"\n[epoch {epoch:02d}] mean_loss = {mean_loss:.4f}\n")
        save_checkpoint(os.path.join(args.save_dir, f"epoch_{epoch:02d}.pt"),
                        model, optimizer, epoch, global_step, mean_loss)

    print(f"\n[done] Checkpoints in {args.save_dir}")


if __name__ == '__main__':
    main()
