"""FlashOcc mIoU evaluation on nuScenes-mini, pure PyTorch / MPS.

Reproduces ``Metric_mIoU`` (use_image_mask=True, num_classes=18) from
``projects/mmdet3d_plugin/core/evaluation/occ_metrics.py``: a class confusion
matrix over camera-visible voxels, reporting per-class IoU and mIoU over
classes 0..16 (the 'free' class 17 is excluded from the mIoU average).

Requires Occ3D occupancy GT -- see flashOcc_vl/data/occ_gt.py for download.

Example:
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
        python flashOcc_vl/tools/eval.py --occ-root /path/to/gts --device mps
    
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
  python flashOcc_vl/tools/eval.py --occ-root /Users/trish/Downloads/Occ3D_nuScenes/gts \
  --device mps --metric ray-iou --scene scene-0103

"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import BEVDetOCC, load_flashocc_checkpoint
from data.loader import NuScenesOccLoader
from data.occ_gt import coverage, load_label, DOWNLOAD_HINT
from tools.infer import GRID_CONFIG, OCC_CLASSES, to_device
from tools.ray_metrics import (generate_lidar_rays, raycast,
                               RayIoUAccumulator)


class MetricMIoU:
    """Confusion-matrix mIoU, faithful to the official Metric_mIoU."""

    def __init__(self, num_classes=18):
        self.n = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.cnt = 0

    def _hist(self, pred, gt):
        k = (gt >= 0) & (gt < self.n)
        return np.bincount(self.n * gt[k].astype(int) + pred[k].astype(int),
                           minlength=self.n ** 2).reshape(self.n, self.n)

    def add(self, pred, gt, mask_camera):
        self.cnt += 1
        self.hist += self._hist(pred[mask_camera].flatten(),
                                gt[mask_camera].flatten())

    def per_class_iou(self):
        h = self.hist
        return np.diag(h) / (h.sum(1) + h.sum(0) - np.diag(h))

    def report(self):
        iou = self.per_class_iou()
        print(f'\n===> per class IoU of {self.cnt} samples:')
        for c in range(self.n - 1):     # exclude 'free'
            print(f'===> {OCC_CLASSES[c]:20s} - IoU = {round(iou[c]*100, 2)}')
        miou = round(float(np.nanmean(iou[:self.n - 1])) * 100, 2)
        print(f'===> mIoU of {self.cnt} samples: {miou}')
        return miou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/Users/trish/Downloads/nuScenes_miniV1.0')
    ap.add_argument('--occ-root', required=True,
                    help='dir with scene-*/<token>/labels.npz')
    ap.add_argument('--ckpt', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'model/checkpoints/flashocc-r50-256x704.pth'))
    ap.add_argument('--device', default='mps')
    ap.add_argument('--metric', default='miou', choices=['miou', 'ray-iou'],
                    help='miou = voxel IoU (default); ray-iou = surface RayIoU')
    ap.add_argument('--ray-step', type=float, default=0.2,
                    help='ray marching stride in metres (ray-iou only)')
    ap.add_argument('--scene', default=None,
                    help='restrict to one scene, e.g. scene-0103')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if (
        args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')
    print(f'device = {device}')

    loader = NuScenesOccLoader(args.dataroot)
    indices = (loader.scene_indices(args.scene) if args.scene
               else list(range(len(loader))))

    found = coverage(loader, args.occ_root, indices)
    if not found:
        print('\n' + DOWNLOAD_HINT)
        print(f'\nChecked {len(indices)} mini samples under {args.occ_root} '
              '-- 0 had labels.npz. Nothing to evaluate.')
        return
    print(f'GT coverage: {len(found)}/{len(indices)} mini samples have occ GT')

    eval_indices = sorted(found)
    if args.limit:
        eval_indices = eval_indices[:args.limit]

    model = BEVDetOCC(grid_config=GRID_CONFIG, input_size=(256, 704),
                      numC_Trans=64, num_classes=18, Dz=16)
    load_flashocc_checkpoint(model, args.ckpt)
    model.eval().to(device)

    if args.metric == 'miou':
        metric = MetricMIoU(num_classes=18)
        for i in tqdm(eval_indices, desc='miou'):
            img_inputs = to_device(loader.get_batched(i), device)
            pred = model.predict_occ(img_inputs)[0].numpy()    # (200,200,16)
            sem, _, mask_cam = load_label(found[i])
            metric.add(pred, sem, mask_cam)
        metric.report()
    else:   # ray-iou
        rays = torch.from_numpy(generate_lidar_rays())          # (R, 3)
        acc = RayIoUAccumulator()
        for i in tqdm(eval_indices, desc='ray-iou'):
            img_inputs = to_device(loader.get_batched(i), device)
            pred = model.predict_occ(img_inputs)[0]            # (200,200,16) cpu
            sem, _, _ = load_label(found[i])                   # GT semantics
            sem = torch.from_numpy(sem.astype('int64'))
            origins = loader.get_lidar_origins(i)              # (T, 3)
            pl, pd = raycast(pred.long(), origins, rays, device,
                             step=args.ray_step)
            gl, gd = raycast(sem, origins, rays, device, step=args.ray_step)
            acc.add(pl, pd, gl, gd)
        acc.report()


if __name__ == '__main__':
    main()
