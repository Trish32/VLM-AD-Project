"""Run FlashOcc over every nuScenes-mini keyframe and cache the occupancy.

One model load, N frames, saved as compressed int8 by sample token. The
e2e_pipeline free-space adapter then replays these the way the detector's
results_nusc.json is replayed -- so the closed loop gets real occupancy without
paying inference per step.

Stored as argmax class labels rather than logits: (200, 200, 16) int8 is 640 kB
per frame against 46 MB for the 18-channel float volume, and `freespace.py`
consumes labels anyway.

Usage:
    python Occupancy/FlashOcc/tools/batch_infer.py --out data/occ_cache.npz
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tools.infer import build_model, to_device          # noqa: E402
from data.loader import NuScenesOccLoader               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot',
                    default=os.environ.get('NUSCENES_DATAROOT') or
                    os.path.expanduser('~/Downloads/nuScenes_miniV1.0'))
    ap.add_argument('--ckpt', default=str(HERE.parent / 'model/checkpoints/flashocc-r50-256x704.pth'))
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', default=str(HERE.parent / 'occ_outputs/occ_cache.npz'))
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if (
        args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')
    model = build_model(args.ckpt, device)
    loader = NuScenesOccLoader(args.dataroot)
    n = args.limit or len(loader)
    print(f'[INFO] {n} frames on {device}')

    out, t0 = {}, time.time()
    for i in range(n):
        with torch.no_grad():
            pred = model(to_device(loader.get_batched(i), device))
        arr = pred[0] if isinstance(pred, (list, tuple)) else pred
        arr = arr.detach().cpu().numpy()
        # Squeeze any leading batch dim, then reduce the CLASS axis to labels.
        # FlashOcc emits (1, X, Y, Z, C) here -- an earlier version of this
        # script tested `ndim == 4` and matched nothing, so float logits were
        # cast straight to int8 and silently wrapped to -128..127. The cache
        # looked plausible (404 keys, right key format) and was garbage; only
        # checking the value range caught it.
        while arr.ndim > 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 4:
            # Class axis is LAST: FlashOcc emits (X, Y, Z, C) = (200,200,16,18).
            # A previous version inferred it as the smallest axis, which picked
            # the 16 height bins over the 18 classes and produced an argmax over
            # HEIGHT -- shape (200,200,18), values 0..15. It looked like labels
            # and was z-indices. Hard-code the known layout rather than guess.
            arr = arr.argmax(axis=-1)
        assert arr.ndim == 3, f'expected (X,Y,Z) labels, got {arr.shape}'
        out[loader.sample_token(i) if hasattr(loader, 'sample_token')
            else str(i)] = arr.astype(np.int8)
        if i % 25 == 0:
            el = time.time() - t0
            print(f'  [{i + 1}/{n}] {el / (i + 1):.2f}s/frame  eta {(n - i - 1) * el / (i + 1) / 60:.1f} min')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    mb = Path(args.out).stat().st_size / 1e6
    print(f'[DONE] {len(out)} frames -> {args.out} ({mb:.1f} MB)')


if __name__ == '__main__':
    main()
