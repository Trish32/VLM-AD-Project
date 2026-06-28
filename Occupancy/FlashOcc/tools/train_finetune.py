"""FlashOcc fine-tuning on nuScenes-mini, pure PyTorch / MPS.

Reproduces the occupancy loss of ``BEVOCCHead2D.loss`` with ``use_mask=True``,
``class_balance=False``: a per-voxel cross-entropy (ignore_index=255) over the
(B,Dx,Dy,Dz,18) logits, masked to camera-visible voxels and normalised by the
number of visible voxels.  Optimiser/clip match the config (AdamW lr=1e-4,
wd=1e-2, grad-clip max_norm=5).

Requires Occ3D occupancy GT -- see data/occ_gt.py for download.

Example (8-step smoke run):
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
        python tools/train_finetune.py --occ-root /path/to/gts \
        --device mps --steps 8
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import BEVDetOCC, load_flashocc_checkpoint
from data.loader import NuScenesOccLoader
from data.occ_gt import coverage, load_label, DOWNLOAD_HINT
from tools.infer import GRID_CONFIG, to_device


def occ_loss(logits, voxel_semantics, mask_camera, num_classes=18):
    """Masked cross-entropy. logits (B,Dx,Dy,Dz,C); others (B,Dx,Dy,Dz)."""
    labels = voxel_semantics.reshape(-1).long()
    preds = logits.reshape(-1, num_classes)
    mask = mask_camera.reshape(-1).float()
    per = F.cross_entropy(preds, labels, reduction='none', ignore_index=255)
    return (per * mask).sum() / mask.sum().clamp(min=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/Users/trish/Downloads/nuScenes_miniV1.0')
    ap.add_argument('--occ-root', required=True)
    ap.add_argument('--ckpt', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'model/checkpoints/flashocc-r50-256x704.pth'))
    ap.add_argument('--device', default='mps')
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--save', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'model/checkpoints/finetune_flashocc.pth'))
    args = ap.parse_args()

    device = torch.device(args.device if (
        args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')
    print(f'device = {device}')

    loader = NuScenesOccLoader(args.dataroot)
    found = coverage(loader, args.occ_root, list(range(len(loader))))
    if not found:
        print('\n' + DOWNLOAD_HINT)
        print('\nNo occ GT found -- cannot fine-tune.')
        return
    print(f'GT coverage: {len(found)} mini samples have occ GT')
    idx = sorted(found)

    model = BEVDetOCC(grid_config=GRID_CONFIG, input_size=(256, 704),
                      numC_Trans=64, num_classes=18, Dz=16)
    load_flashocc_checkpoint(model, args.ckpt)
    model.train().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    for step in range(args.steps):
        i = idx[step % len(idx)]
        img_inputs = to_device(loader.get_batched(i), device)
        sem, _, mask_cam = load_label(found[i])
        sem = torch.from_numpy(sem.astype(np.int64))[None].to(device)
        mask_cam = torch.from_numpy(mask_cam)[None].to(device)

        logits = model(img_inputs)              # (1,200,200,16,18)
        loss = occ_loss(logits, sem, mask_cam)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5,
                                       norm_type=2)
        opt.step()
        print(f'step {step:3d}  sample {i:3d}  loss {loss.item():.4f}')

    torch.save({'state_dict': model.state_dict()}, args.save)
    print('saved fine-tuned checkpoint ->', args.save)


if __name__ == '__main__':
    main()
