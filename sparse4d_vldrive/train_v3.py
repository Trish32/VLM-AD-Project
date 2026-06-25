"""
Full training of Sparse4D-v3 on nuScenes mini (Apple MPS) with the backbone
UNFROZEN, plus the three v3 training techniques wired in:

  • Temporal Instance Denoising (DN)   — model/denoising.py
  • Quality-branch supervision (cns/yns)— Sparse4DLoss / DNLoss (matches the
                                          official Sparse4Dv3 losses.py targets)
  • Dense depth supervision             — model/depth_head.py + data/lidar_depth.py

This is the v3 analogue of train_v2.py: the backbone trains (except stem +
layer1, frozen per frozen_stages=1; BN stays in eval per norm_eval=True).

Total loss = detection (per-stage Hungarian, cls+box+cns+yns)
           + DN (direct, cls+box+cns+yns over noised GT copies)
           + depth_weight * masked-L1 depth

All three additions are TRAINING-ONLY: tools/eval.py builds a plain Sparse4Dv3
(no depth head, no DN, normal forward), so the eval path is unchanged.

Usage
-----
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
        python train_v3.py \
        --checkpoint sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth \
        --epochs 6 --lr 2e-5 --dn_groups 5 --depth_weight 0.05 \
        --save_dir checkpoints/train_v3
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
from sparse4d_vl.model.sparse4d_v3    import Sparse4Dv3
from sparse4d_vl.model.checkpoint     import load_checkpoint
from sparse4d_vl.model.loss           import Sparse4DLoss
from sparse4d_vl.model.denoising      import DNLoss
from sparse4d_vl.model.depth_head     import depth_l1_loss
from sparse4d_vl.model.motion_head    import MotionLoss, MotionMeter
from sparse4d_vl.model.motion_planning import (
    build_anchors, AgentMotionLoss, PlanLoss, PlanMeter)


def build_model(checkpoint_path: str, device: torch.device, with_depth: bool,
                with_motion: bool = False, motion_modes: int = 6,
                motion_steps: int = 12, with_planning: bool = False,
                ego_steps: int = 6, with_map: bool = False) -> Sparse4Dv3:
    model = Sparse4Dv3(with_depth=with_depth, with_motion=with_motion,
                       motion_modes=motion_modes, motion_steps=motion_steps,
                       with_planning=with_planning, ego_steps=ego_steps,
                       with_map=with_map).to(device)
    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint(model, checkpoint_path, version='v3')
        print(f"[init] Checkpoint loaded: {checkpoint_path}")
    else:
        print("[init] No checkpoint — training from ImageNet backbone only")

    # Backbone NOT frozen (stem+layer1 + all BN stay frozen via ResNet50.train()).
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[init] Params — total: {total/1e6:.1f}M  trainable: {trainable/1e6:.1f}M  "
          f"(stem+layer1 + all BN frozen)")
    return model


def build_optimizer(model, lr, weight_decay, total_steps, warmup_steps, backbone_lr_mult):
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
        return float(0.5 * (1.0 + np.cos(progress * np.pi)))   # python float

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def save_checkpoint(path, model, optimizer, epoch, step, loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_state = {k: v for k, v in model.state_dict().items() if '_cached_' not in k}
    torch.save({'epoch': epoch, 'step': step, 'loss': loss,
                'model': model_state, 'optimizer': optimizer.state_dict()}, path)
    print(f"[ckpt] Saved → {path}")


def train_one_epoch(model, loader, criterion, dn_criterion, optimizer, scheduler,
                    device, epoch, global_step, dn_groups, depth_weight,
                    log_every, grad_clip, motion_criterion=None, motion_weight=1.0,
                    agent_motion_criterion=None, plan_criterion=None,
                    motion_weight_sd=1.0, plan_weight=1.0):
    model.train()
    model.backbone.eval()   # keep BN frozen even though backbone params train

    total_loss = total_det = total_dn = total_dep = total_mot = total_plan = 0.0
    motion_meter = MotionMeter() if motion_criterion is not None else None
    plan_meter = PlanMeter() if plan_criterion is not None else None
    sd_motion_meter = MotionMeter() if agent_motion_criterion is not None else None
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
        output = model.forward_train(imgs, img_metas, gt_boxes, gt_labels,
                                     dn_groups=dn_groups)

        # Detection loss (now includes cns/yns via 3-tuple stage_preds)
        det_loss = criterion.forward_multi(output['stage_preds'], gt_boxes, gt_labels)

        # DN loss (cls+box+cns+yns, known correspondence)
        dn_loss = imgs.sum() * 0.0
        if output.get('dn_stage_preds'):
            dn_loss = dn_criterion(output['dn_stage_preds'],
                                   output['dn_labels'], output['dn_gt_boxes'])

        # Depth loss
        dep_loss = imgs.sum() * 0.0
        if 'depth_pred' in output:
            H_f, W_f = output['depth_pred'].shape[-2:]
            dgt, dmask = depth_targets_for_sample(
                loader.nusc, img_metas['sample_token'],
                img_metas['projection_mat'], img_metas['img_wh'], (H_f, W_f))
            dep_loss = depth_l1_loss(output['depth_pred'], dgt.to(device), dmask.to(device))

        # Motion forecasting loss (QCNet-style; matched final-stage queries)
        mot_loss = imgs.sum() * 0.0
        if motion_criterion is not None and 'trajectories' in output:
            gtf = frame['gt_futures'].to(device)
            gtm = frame['gt_future_mask'].to(device)
            if gtf.shape[0] > 0:
                pidx, gidx = criterion._match(
                    output['final_anchor'][0], output['final_cls'][0], gt_boxes, gt_labels)
                if len(pidx) > 0:
                    traj = output['trajectories'][0][pidx.to(device)]
                    ml   = output['traj_mode_logits'][0][pidx.to(device)]
                    mot_loss = motion_criterion(traj, ml, gtf[gidx.to(device)], gtm[gidx.to(device)])
                    motion_meter.update(traj.detach(), ml.detach(),
                                        gtf[gidx.to(device)], gtm[gidx.to(device)])

        # SparseDrive motion (anchored agents) + ego planning
        sd_mot_loss = imgs.sum() * 0.0
        plan_loss = imgs.sum() * 0.0
        if agent_motion_criterion is not None and 'trajectories' in output:
            gtf = frame['gt_futures'].to(device); gtm = frame['gt_future_mask'].to(device)
            if gtf.shape[0] > 0:
                pidx, gidx = criterion._match(
                    output['final_anchor'][0], output['final_cls'][0], gt_boxes, gt_labels)
                if len(pidx) > 0:
                    traj = output['trajectories'][0][pidx.to(device)]
                    ml   = output['traj_mode_logits'][0][pidx.to(device)]
                    sd_mot_loss = agent_motion_criterion(traj, ml, gtf[gidx.to(device)], gtm[gidx.to(device)])
                    sd_motion_meter.update(traj.detach(), ml.detach(), gtf[gidx.to(device)], gtm[gidx.to(device)])
            if plan_criterion is not None and 'ego_future' in frame:
                ego_f = frame['ego_future'].unsqueeze(0).to(device)        # (1,T,2)
                ego_m = frame['ego_future_mask'].unsqueeze(0).to(device)   # (1,T)
                cmd   = frame['command'].view(1).to(device)
                plan_loss = plan_criterion(
                    output['ego_traj'], output['ego_logits'], cmd, ego_f, ego_m,
                    agent_future=gtf, agent_mask=gtm)
                plan_meter.update(output['ego_traj'].detach(), output['ego_logits'].detach(),
                                  cmd, ego_f, ego_m, agent_future=gtf, agent_mask=gtm)

        loss = (det_loss + dn_loss + depth_weight * dep_loss + motion_weight * mot_loss
                + motion_weight_sd * sd_mot_loss + plan_weight * plan_loss)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        if device.type == 'mps':
            torch.mps.empty_cache()

        total_loss += loss.item(); total_det += det_loss.item()
        total_dn += float(dn_loss); total_dep += float(dep_loss); total_mot += float(mot_loss)
        total_plan += float(plan_loss) + float(sd_mot_loss)
        n_frames += 1; global_step += 1

        if global_step % log_every == 0:
            lr_now = scheduler.get_last_lr()[0]
            fps = n_frames / max(time.time() - t0, 1e-6)
            extra = ""
            if agent_motion_criterion is not None:
                extra = f"  motion {float(sd_mot_loss):.3f}  plan {float(plan_loss):.3f}"
            print(f"  epoch {epoch:02d} | step {global_step:05d} | loss {loss.item():.4f} "
                  f"(det {det_loss.item():.4f}  dn {float(dn_loss):.4f}  depth {float(dep_loss):.4f}"
                  f"  motion {float(mot_loss):.4f}{extra}) | lr {lr_now:.2e} | {fps:.2f} fr/s")

    if motion_meter is not None and motion_meter.n > 0:
        m = motion_meter.compute()
        print(f"  [motion train] minADE {m['minADE']:.3f}  minFDE {m['minFDE']:.3f}  "
              f"brier-minFDE {m['brier_minFDE']:.3f}  MR {m['MR']:.3f}")
    if sd_motion_meter is not None and sd_motion_meter.n > 0:
        m = sd_motion_meter.compute()
        print(f"  [SD motion] minADE {m['minADE']:.3f}  minFDE {m['minFDE']:.3f}  MR {m['MR']:.3f}")
    if plan_meter is not None and plan_meter.n > 0:
        p = plan_meter.compute()
        print(f"  [plan] L2@1s {p['L2@1s']:.3f} @2s {p['L2@2s']:.3f} @3s {p['L2@3s']:.3f}  "
              f"col@1s {p['col@1s']:.3f} @2s {p['col@2s']:.3f} @3s {p['col@3s']:.3f}")
    return total_loss / max(n_frames, 1), global_step


def parse_args():
    p = argparse.ArgumentParser(description='Train Sparse4Dv3 (unfrozen backbone + DN + quality + depth)')
    p.add_argument('--dataroot',   default='/Users/trish/Downloads/nuScenes_miniV1.0')
    p.add_argument('--checkpoint', default='sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth')
    p.add_argument('--save_dir',   default='checkpoints/train_v3')
    p.add_argument('--resume',     default=None)
    p.add_argument('--epochs',     type=int,   default=6)
    p.add_argument('--lr',         type=float, default=2e-5)
    p.add_argument('--backbone_lr_mult', type=float, default=0.1)
    p.add_argument('--freeze_backbone', action='store_true',
                   help='freeze the whole backbone (finetune mode vs full train)')
    p.add_argument('--planning_only', action='store_true',
                   help='freeze the entire detector and train ONLY the motion/ego heads '
                        '(avoids degrading detection when adding planning)')
    p.add_argument('--weight_decay',  type=float, default=0.01)
    p.add_argument('--warmup_epochs', type=float, default=0.5)
    p.add_argument('--dn_groups',     type=int,   default=5, help='DN groups (0 = off)')
    p.add_argument('--depth_weight',  type=float, default=0.05, help='0 disables depth')
    p.add_argument('--motion',        action='store_true', help='QCNet-Laplace trajectory head')
    p.add_argument('--motion_modes',  type=int,   default=6)
    p.add_argument('--motion_steps',  type=int,   default=12, help='future keyframes (12=6s@2Hz)')
    p.add_argument('--motion_weight', type=float, default=0.5)
    p.add_argument('--planning',      action='store_true',
                   help='SparseDrive motion planner: anchored agent motion + ego planner')
    p.add_argument('--ego_steps',     type=int,   default=6, help='ego horizon (6=3s@2Hz)')
    p.add_argument('--with_map',      action='store_true',
                   help='agent–map cross-attention over HD-map polylines')
    p.add_argument('--motion_weight_sd', type=float, default=1.0)
    p.add_argument('--plan_weight',   type=float, default=1.0)
    p.add_argument('--anchor_cache',  default='sparse4d_vl/data/motion_anchors.npz')
    p.add_argument('--grad_clip',  type=float, default=35.0)
    p.add_argument('--log_every',  type=int,   default=20)
    p.add_argument('--version',    default='v1.0-mini')
    # Loss weights
    p.add_argument('--weight_cls', type=float, default=2.0)
    p.add_argument('--weight_reg', type=float, default=0.25)
    p.add_argument('--weight_vel', type=float, default=0.2)
    p.add_argument('--weight_cns', type=float, default=1.0)
    p.add_argument('--weight_yns', type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    device = (torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"[init] Device: {device}")

    need_future = args.motion or args.planning
    loader = NuScenesFinetuneLoader(
        dataroot=args.dataroot, version=args.version,
        future_steps=args.motion_steps if need_future else 0,
        plan=args.planning, with_map=args.with_map)
    total_frames = len(loader)
    print(f"[init] Dataset: {total_frames} keyframes across {len(loader.nusc.scene)} scenes")

    warmup_steps = int(args.warmup_epochs * total_frames)
    total_steps  = args.epochs * total_frames

    with_depth = args.depth_weight > 0
    model = build_model(args.checkpoint, device, with_depth=with_depth,
                        with_motion=args.motion, motion_modes=args.motion_modes,
                        motion_steps=args.motion_steps, with_planning=args.planning,
                        ego_steps=args.ego_steps, with_map=args.with_map)
    if args.freeze_backbone:                       # finetune mode
        for prm in model.backbone.parameters():
            prm.requires_grad = False
        print("[init] backbone FROZEN (finetune mode)")

    if args.planning_only:                         # train ONLY the new heads
        head_prefixes = ('agent_motion.', 'ego_planner.', 'motion_head.', 'map_encoder.')
        for name, prm in model.named_parameters():
            prm.requires_grad = name.startswith(head_prefixes)
        ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[init] PLANNING-ONLY: detector frozen, training motion/ego heads "
              f"({ntrain/1e6:.2f}M params)")

    # SparseDrive: build/cache kmeans anchors and load them into the heads
    agent_motion_criterion = plan_criterion = None
    if args.planning:
        print("[init] building motion/ego anchors (kmeans over GT futures)...")
        agent_a, ego_a = build_anchors(loader, num_modes=args.motion_modes,
                                       future_steps=args.motion_steps,
                                       ego_steps=args.ego_steps, cache=args.anchor_cache)
        model.agent_motion.anchors.copy_(agent_a.to(device))
        model.ego_planner.anchors.copy_(ego_a.to(device))
        agent_motion_criterion = AgentMotionLoss()
        plan_criterion = PlanLoss()

    optimizer, scheduler = build_optimizer(
        model, args.lr, args.weight_decay, total_steps, warmup_steps, args.backbone_lr_mult)

    loss_kw = dict(weight_cls=args.weight_cls, weight_reg=args.weight_reg,
                   weight_vel=args.weight_vel, weight_cns=args.weight_cns,
                   weight_yns=args.weight_yns)
    criterion    = Sparse4DLoss(**loss_kw)
    dn_criterion = DNLoss(**loss_kw)
    motion_criterion = MotionLoss() if args.motion else None
    print(f"[init] DN groups={args.dn_groups}  depth={'on' if with_depth else 'off'}  "
          f"motion={'on' if args.motion else 'off'}  planning={'on' if args.planning else 'off'}  "
          f"quality cns/yns weights={args.weight_cns}/{args.weight_yns}")

    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, global_step = ckpt['epoch'] + 1, ckpt['step']
        print(f"[resume] from epoch {ckpt['epoch']}")

    print(f"\n[train] {args.epochs} epochs ({total_steps} steps)\n")
    for epoch in range(start_epoch, args.epochs):
        print(f"{'='*60}\nEpoch {epoch:02d} / {args.epochs - 1}\n{'='*60}")
        mean_loss, global_step = train_one_epoch(
            model, loader, criterion, dn_criterion, optimizer, scheduler, device,
            epoch, global_step, args.dn_groups, args.depth_weight,
            args.log_every, args.grad_clip,
            motion_criterion=motion_criterion, motion_weight=args.motion_weight,
            agent_motion_criterion=agent_motion_criterion, plan_criterion=plan_criterion,
            motion_weight_sd=args.motion_weight_sd, plan_weight=args.plan_weight)
        print(f"\n[epoch {epoch:02d}] mean_loss = {mean_loss:.4f}\n")
        save_checkpoint(os.path.join(args.save_dir, f"epoch_{epoch:02d}.pt"),
                        model, optimizer, epoch, global_step, mean_loss)

    print(f"\n[done] Checkpoints in {args.save_dir}")


if __name__ == '__main__':
    main()
