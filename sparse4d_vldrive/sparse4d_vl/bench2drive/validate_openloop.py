#!/usr/bin/env python3
"""
Local open-loop validation of the full Sparse4D-v3 driving stack:

    6-camera multi-view → detect → TRACK → MOTION → PLAN → CONTROL

Runs on nuScenes (no CARLA needed) using the SAME plan→control path as the
Bench2Drive agent (bench2drive/agent.py:plan_to_control), so it exercises the
entire chain end-to-end on this machine and reports, per frame:
  #tracks, #agents-with-motion, driving command, ego-plan endpoint, and the
  emitted steer/throttle/brake — plus open-loop planning L2 @1/2/3 s vs GT ego.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python sparse4d_vl/bench2drive/validate_openloop.py \
        --checkpoint checkpoints/train_v3_plan3/epoch_05.pt --scene 1 --max-frames 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sparse4d_vl.data.finetune_loader import NuScenesFinetuneLoader
from sparse4d_vl.model.sparse4d_v3 import Sparse4Dv3
from sparse4d_vl.model.checkpoint import load_checkpoint
from sparse4d_vl.model.motion_planning import PlanMeter
from sparse4d_vl.bench2drive.controller import TrajectoryController
from sparse4d_vl.bench2drive.agent import plan_to_control

_CMD = {0: 'right', 1: 'straight', 2: 'left'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/train_v3_plan3/epoch_05.pt')
    p.add_argument('--dataroot', default='/Users/trish/Downloads/nuScenes_miniV1.0')
    p.add_argument('--version', default='v1.0-mini')
    p.add_argument('--scene', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=8)
    args = p.parse_args()

    dev = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    # auto-detect map support from the checkpoint
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    with_map = any('map_encoder' in k for k in ck.get('model', ck))
    print(f"[validate] checkpoint={args.checkpoint}  with_map={with_map}")

    model = Sparse4Dv3(with_planning=True, with_map=with_map, ego_steps=6).to(dev)
    model.eval()
    load_checkpoint(model, args.checkpoint, version='v3')
    loader = NuScenesFinetuneLoader(args.dataroot, version=args.version,
                                    future_steps=12, plan=True, with_map=with_map)
    ctrl = TrajectoryController()
    meter = PlanMeter()

    print(f"\n[validate] scene {args.scene} — full multi-view→track→motion→plan→control\n")
    model.reset_state(); ctrl.reset()
    prev_t = prev_xy = None
    for i, frame in enumerate(loader.iter_scene(args.scene)):
        if i >= args.max_frames:
            break
        metas = frame['img_metas']
        # current speed from consecutive ego-global translations
        t = metas['timestamp']; xy = metas['ego2global'][:2, 3]
        speed = 0.0
        if prev_t is not None and t > prev_t:
            speed = float(np.linalg.norm(xy - prev_xy) / (t - prev_t))
        prev_t, prev_xy = t, xy

        with torch.no_grad():
            out = model(frame['imgs'].float(), metas)
        det = out['detections'][0]
        cmd = int(frame['command'])
        c = plan_to_control(out, cmd, speed, ctrl)

        n_box = det['boxes_3d'].shape[0]
        n_trk = int((det['track_ids'] >= 0).sum()) if 'track_ids' in det else 0
        ego_end = out['ego_traj'][0, cmd, -1].cpu().numpy()
        print(f"  f{i:02d} | boxes={n_box:3d} tracks={n_trk:3d} motion_agents={n_box:3d} "
              f"| cmd={_CMD[cmd]:8s} ego→({ego_end[0]:5.1f},{ego_end[1]:5.1f})m "
              f"| speed={speed:4.1f} → steer={c.steer:+.2f} thr={c.throttle:.2f} brk={c.brake:.2f}")

        # open-loop planning metric vs GT ego
        meter.update(out['ego_traj'], out['ego_logits'],
                     frame['command'].view(1).to(dev),
                     frame['ego_future'].unsqueeze(0).to(dev),
                     frame['ego_future_mask'].unsqueeze(0).to(dev),
                     agent_future=frame['gt_futures'].to(dev),
                     agent_mask=frame['gt_future_mask'].to(dev))

    m = meter.compute()
    print(f"\n[validate] open-loop planning  L2@1/2/3s = "
          f"{m['L2@1s']:.2f}/{m['L2@2s']:.2f}/{m['L2@3s']:.2f} m  "
          f"col = {m['col@1s']:.2f}/{m['col@2s']:.2f}/{m['col@3s']:.2f}")
    print("[validate] full chain ran end-to-end: multi-view→track→motion→plan→control ✓")


if __name__ == '__main__':
    main()
