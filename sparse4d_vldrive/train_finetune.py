"""
Fine-tuning script for Sparse4D v2 on nuScenes mini (Apple MPS).

Strategy
--------
  - Load official pretrained checkpoint (sparse4dv2_r50_HInf_256x704.pth)
  - Freeze backbone (ResNet-50 + FPN): only the Sparse4D head is trained
  - Iterate nuScenes mini keyframes scene-by-scene (temporal state reset per scene)
  - Optimise with AdamW + cosine LR schedule
  - Save a checkpoint after each epoch

Usage
-----
    conda run -n simple_bev_vldrive python train_finetune.py \
        --dataroot /Users/trish/Downloads/nuScenes_miniV1.0 \
        --checkpoint sparse4d_vl/model/checkpoints/sparse4dv2_r50_HInf_256x704.pth \
        --epochs 10 \
        --lr 2e-4 \
        --save_dir checkpoints/finetune

Resume from a saved epoch:
    python train_finetune.py ... --resume checkpoints/finetune/epoch_03.pt

MPS out-of-memory:
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python train_finetune.py ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from sparse4d_vl.data.finetune_loader import NuScenesFinetuneLoader
from sparse4d_vl.model.sparse4d_v2   import Sparse4Dv2
from sparse4d_vl.model.sparse4d_v3   import Sparse4Dv3
from sparse4d_vl.model.checkpoint    import load_checkpoint
from sparse4d_vl.model.loss          import Sparse4DLoss
from sparse4d_vl.model.denoising     import DNLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(checkpoint_path: str, device: torch.device,
                version: str = 'v2') -> Sparse4Dv2 | Sparse4Dv3:
    """Load Sparse4Dv2/v3 and freeze backbone weights."""
    model_cls = Sparse4Dv3 if version == 'v3' else Sparse4Dv2
    model = model_cls().to(device)
    load_checkpoint(model, checkpoint_path, version=version)
    print(f"[init] Checkpoint loaded: {checkpoint_path}")

    # Freeze backbone (ResNet-50 + 4-level FPN)
    for param in model.backbone.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frozen    = total - trainable
    print(f"[init] Params — total: {total/1e6:.1f}M  "
          f"trainable (head): {trainable/1e6:.1f}M  "
          f"frozen (backbone): {frozen/1e6:.1f}M")
    return model


def build_optimizer(
    model: Sparse4Dv2,
    lr: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    # This build_optimizer passes only requires_grad params to AdamW  — 
    # so the frozen backbone isn't even in the optimizer state.
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(head_params, lr=lr,
                                     weight_decay=weight_decay)
    # Cosine schedule with linear warmup
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def save_checkpoint(
    path: str,
    model: Sparse4Dv2,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Exclude temporal cache buffers (runtime state that must be reset per scene)
    model_state = {k: v for k, v in model.state_dict().items() if '_cached_' not in k}  # The checkpoint stores weights, not one scene's temporal state
    torch.save({
        'epoch':     epoch,
        'step':      step,
        'loss':      loss,
        'model':     model_state,
        'optimizer': optimizer.state_dict(),
    }, path)
    print(f"[ckpt] Saved → {path}")


def load_resume(
    path: str,
    model: Sparse4Dv2,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    """Returns (start_epoch, global_step)."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer'])
    print(f"[resume] Resumed from epoch {ckpt['epoch']}  step {ckpt['step']}")
    return ckpt['epoch'] + 1, ckpt['step']


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     Sparse4Dv2,
    loader:    NuScenesFinetuneLoader,
    criterion: Sparse4DLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device:    torch.device,
    epoch:     int,
    global_step: int,
    log_every:   int = 20,
    grad_clip:   float = 35.0,
    dn_criterion: DNLoss | None = None,
    dn_groups:    int = 0,
) -> tuple[float, int]:
    """
    Iterates all scenes / keyframes for one epoch.

    Returns (mean_loss_for_epoch, updated_global_step).
    """
    model.train()
    # Keep backbone in eval mode — it's frozen but BN layers should stay stable
    # even though the model is in .train(). This keeps BatchNorm in inference mode
    #  — frozen weights aren't enough, because BN would otherwise keep updating its running mean/var from the tiny mini-set and drift. 
    # Freeze the params and the BN statistics.
    model.backbone.eval()  

    total_loss   = 0.0
    total_frames = 0
    t0           = time.time()

    for frame in loader:
        # ---- Scene boundary: reset temporal state ----
        # the temporal cache builds within a scene and must be cleared between scenes, during training too
        if frame['is_first_frame']:
            model.reset_state()  # scene boundary → clear instance bank

        imgs      = frame['imgs'].to(device)             # (1, N_cam, 3, H, W)
        img_metas = frame['img_metas']
        gt_boxes  = frame['gt_boxes'].to(device)         # (M, 11)
        gt_labels = frame['gt_labels'].to(device)        # (M,)

        # ---- Forward ----
        optimizer.zero_grad(set_to_none=True)

        # ---- Forward (+ Temporal Instance Denoising for v3 when enabled) ----
        if dn_criterion is not None and dn_groups > 0:
            output = model.forward_train(imgs, img_metas, gt_boxes, gt_labels,
                                         dn_groups=dn_groups)
        else:
            output = model(imgs, img_metas)  # full 6-stage forward

        # ---- Loss: per-stage Hungarian supervision over all 6 decoder stages ----
        loss = criterion.forward_multi(output['stage_preds'], gt_boxes, gt_labels)

        # ---- DN loss: direct (known correspondence) over DN groups ----
        if dn_criterion is not None and output.get('dn_stage_preds'):
            loss = loss + dn_criterion(
                output['dn_stage_preds'], output['dn_labels'], output['dn_gt_boxes'])

        # ---- Backward ----
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],  # only over head params
            max_norm=grad_clip,
        )  # Stability: guards against the occasional huge gradient from a bad match early on
        optimizer.step()
        scheduler.step()

        # MPS: release unused cached memory each step to avoid fragmentation(OOM over a long run)
        if device.type == 'mps':
            torch.mps.empty_cache()

        total_loss   += loss.item()
        total_frames += 1
        global_step  += 1

        if global_step % log_every == 0:
            mem_str = ""
            if device.type == 'mps':
                mem_gb = torch.mps.current_allocated_memory() / 1e9
                mem_str = f"  MPS {mem_gb:.2f} GB"
            elif device.type == 'cuda':
                mem_gb = torch.cuda.memory_allocated() / 1e9
                mem_str = f"  CUDA {mem_gb:.2f} GB"

            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            fps     = total_frames / elapsed if elapsed > 0 else 0.0
            print(
                f"  epoch {epoch:02d} | step {global_step:05d} "
                f"| loss {loss.item():.4f} "
                f"| lr {lr_now:.2e} "
                f"| {fps:.2f} fr/s"
                f"{mem_str}"
            )

    mean_loss = total_loss / max(total_frames, 1)
    return mean_loss, global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Fine-tune Sparse4Dv2 on nuScenes mini (MPS)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dataroot',   default='/Users/trish/Downloads/nuScenes_miniV1.0',
                   help='Path to nuScenes mini dataset root')
    p.add_argument('--checkpoint',
                   default='sparse4d_vl/model/checkpoints/sparse4dv2_r50_HInf_256x704.pth',
                   help='Path to pretrained Sparse4Dv2 checkpoint')
    p.add_argument('--save_dir',   default='checkpoints/finetune',
                   help='Directory to save fine-tuned checkpoints')
    p.add_argument('--resume',     default=None,
                   help='Path to a previous fine-tune checkpoint to resume from')
    p.add_argument('--epochs',     type=int,   default=10)
    p.add_argument('--lr',         type=float, default=1e-5,
                   help='Peak learning rate for AdamW (low: fine-tuning a converged model)')
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_epochs', type=float, default=0.5,
                   help='Fraction of first epoch used for LR warmup')
    p.add_argument('--grad_clip',  type=float, default=35.0)
    p.add_argument('--log_every',  type=int,   default=20,
                   help='Print log every N gradient steps')
    p.add_argument('--version',    default='v1.0-mini',
                   help='nuScenes dataset version string')
    p.add_argument('--model_version', default='v2', choices=['v2', 'v3'],
                   help='Sparse4D model version to fine-tune')
    p.add_argument('--dn_groups', type=int, default=0,
                   help='Temporal Instance Denoising groups (v3 only; 0 = off)')
    # Loss weights
    p.add_argument('--weight_cls', type=float, default=2.0)
    p.add_argument('--weight_reg', type=float, default=0.25)
    p.add_argument('--weight_vel', type=float, default=0.2)
    return p.parse_args()


def main():
    """
    The flow is: build_model → build_optimizer → loop epochs → save
    """
    args   = parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[init] Device: {device}")

    # Data
    print(f"[init] Loading nuScenes mini from {args.dataroot} ...")
    loader = NuScenesFinetuneLoader(
        dataroot=args.dataroot,
        version=args.version,
    )
    total_frames = len(loader)
    print(f"[init] Dataset: {total_frames} keyframes across "
          f"{len(loader.nusc.scene)} scenes")

    # Estimate total steps for scheduler
    warmup_steps = int(args.warmup_epochs * total_frames)
    total_steps  = args.epochs * total_frames

    # Model
    model     = build_model(args.checkpoint, device, version=args.model_version)
    optimizer, scheduler = build_optimizer(
        model, args.lr, args.weight_decay, total_steps, warmup_steps
    )

    # Loss
    criterion = Sparse4DLoss(
        weight_cls=args.weight_cls,
        weight_reg=args.weight_reg,
        weight_vel=args.weight_vel,
    )

    # Temporal Instance Denoising (v3 only)
    dn_criterion = None
    if args.dn_groups > 0:
        if args.model_version != 'v3':
            raise ValueError('--dn_groups requires --model_version v3')
        dn_criterion = DNLoss(
            weight_cls=args.weight_cls,
            weight_reg=args.weight_reg,
            weight_vel=args.weight_vel,
        )
        print(f"[init] Temporal Instance Denoising ON — {args.dn_groups} groups")

    # Resume
    start_epoch  = 0
    global_step  = 0
    if args.resume:
        start_epoch, global_step = load_resume(args.resume, model, optimizer)

    # Training
    print(f"\n[train] Starting fine-tuning for {args.epochs} epochs "
          f"({total_steps} total steps)\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"{'='*60}")
        print(f"Epoch {epoch:02d} / {args.epochs - 1}")
        print(f"{'='*60}")

        mean_loss, global_step = train_one_epoch(
            model        = model,
            loader       = loader,
            criterion    = criterion,
            optimizer    = optimizer,
            scheduler    = scheduler,
            device       = device,
            epoch        = epoch,
            global_step  = global_step,
            log_every    = args.log_every,
            grad_clip    = args.grad_clip,
            dn_criterion = dn_criterion,
            dn_groups    = args.dn_groups,
        )

        print(f"\n[epoch {epoch:02d}] mean_loss = {mean_loss:.4f}\n")

        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch:02d}.pt")
        save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, mean_loss)

    print("\n[done] Fine-tuning complete.")
    print(f"       Checkpoints saved in: {args.save_dir}")


if __name__ == '__main__':
    main()
