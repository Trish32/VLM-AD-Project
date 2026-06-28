"""FlashOcc (BEVDetOCC) inference on nuScenes-mini, pure PyTorch / MPS.

Example:
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
        python tools/infer.py --frame 0 --device mps
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import BEVDetOCC, load_flashocc_checkpoint
from data.loader import NuScenesOccLoader

GRID_CONFIG = {'x': [-40, 40, 0.4], 'y': [-40, 40, 0.4],
               'z': [-1, 5.4, 6.4], 'depth': [1.0, 45.0, 0.5]}

OCC_CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free']


def build_model(ckpt, device):
    model = BEVDetOCC(grid_config=GRID_CONFIG, input_size=(256, 704),
                      numC_Trans=64, num_classes=18, Dz=16)
    load_flashocc_checkpoint(model, ckpt)
    model.eval().to(device)
    return model


def to_device(img_inputs, device):
    return tuple(t.to(device) for t in img_inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/Users/trish/Downloads/nuScenes_miniV1.0')
    ap.add_argument('--ckpt', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'model/checkpoints/flashocc-r50-256x704.pth'))
    ap.add_argument('--device', default='mps')
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--out', default=None, help='optional .npy to save (Dx,Dy,Dz)')
    args = ap.parse_args()

    device = torch.device(args.device if (
        args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')
    print(f'device = {device}')

    model = build_model(args.ckpt, device)
    loader = NuScenesOccLoader(args.dataroot)
    print(f'{len(loader)} samples in nuScenes-mini')

    img_inputs = to_device(loader.get_batched(args.frame), device)
    t0 = time.time()
    occ = model.predict_occ(img_inputs)[0].numpy()      # (200, 200, 16) uint8
    dt = time.time() - t0

    print(f'frame {args.frame}  token={loader.sample_token(args.frame)}')
    print(f'occ shape {occ.shape}  dtype {occ.dtype}  '
          f'({dt*1000:.0f} ms)')
    free = 17
    occupied = occ != free
    print(f'occupied voxels: {occupied.sum()} / {occ.size} '
          f'({100*occupied.mean():.1f}%)')
    ids, counts = np.unique(occ, return_counts=True)
    print('class histogram (non-free):')
    for i, c in zip(ids, counts):
        if i == free:
            continue
        print(f'  {OCC_CLASSES[i]:20s} {c}')

    if args.out:
        np.save(args.out, occ)
        print('saved', args.out)


if __name__ == '__main__':
    main()
