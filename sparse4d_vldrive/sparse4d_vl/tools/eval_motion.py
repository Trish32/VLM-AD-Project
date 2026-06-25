#!/usr/bin/env python3
"""
Evaluate Sparse4D-v3 SparseDrive motion + planning on nuScenes mini.

Motion (agents): minADE / minFDE / miss-rate over K modes, on GT-matched agents.
Planning (ego):  L2 @ 1/2/3 s and collision rate, for the GT-command ego mode.

The checkpoint must have been trained with --planning (so it carries the motion
head, ego planner, and the kmeans anchor buffers).

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python sparse4d_vl/tools/eval_motion.py \
        --checkpoint checkpoints/train_v3_plan/epoch_05.pt --eval-set mini_val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sparse4d_vl.data.finetune_loader import NuScenesFinetuneLoader
from sparse4d_vl.model.sparse4d_v3 import Sparse4Dv3
from sparse4d_vl.model.checkpoint import load_checkpoint
from sparse4d_vl.model.loss import Sparse4DLoss
from sparse4d_vl.model.motion_head import MotionMeter
from sparse4d_vl.model.motion_planning import PlanMeter
from sparse4d_vl.tools.eval import _split_tokens


def main():
    p = argparse.ArgumentParser(description='Sparse4D-v3 motion + planning eval')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataroot', default='/Users/trish/Downloads/nuScenes_miniV1.0')
    p.add_argument('--version', default='v1.0-mini')
    p.add_argument('--eval-set', default='mini_val')
    p.add_argument('--motion_steps', type=int, default=12)
    p.add_argument('--ego_steps', type=int, default=6)
    args = p.parse_args()

    dev = (torch.device('mps') if torch.backends.mps.is_available()
           else torch.device('cpu'))

    # Auto-detect whether this checkpoint was trained with agent–map attention
    _ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    _state = _ck.get('model', _ck)
    with_map = any('map_encoder' in k for k in _state)
    print(f"[eval] with_map={with_map} (auto-detected from checkpoint)")

    loader = NuScenesFinetuneLoader(args.dataroot, version=args.version,
                                    future_steps=args.motion_steps, plan=True,
                                    with_map=with_map)
    split = _split_tokens(loader.nusc, args.eval_set)

    model = Sparse4Dv3(with_planning=True, motion_steps=args.motion_steps,
                       ego_steps=args.ego_steps, with_map=with_map).to(dev)
    model.eval()
    load_checkpoint(model, args.checkpoint, version='v3')
    matcher = Sparse4DLoss()
    mm, pm = MotionMeter(), PlanMeter()

    print(f"[eval] motion+planning on {args.eval_set}")
    with torch.no_grad():
        for scene_idx in range(len(loader.nusc.scene)):
            model.reset_state()
            for frame in loader.iter_scene(scene_idx):
                if frame['img_metas']['sample_token'] not in split:
                    continue
                out = model(frame['imgs'].float(), frame['img_metas'])
                gtf = frame['gt_futures'].to(dev); gtm = frame['gt_future_mask'].to(dev)
                gtb = frame['gt_boxes'].to(dev);  gtl = frame['gt_labels'].to(dev)
                # motion: match final preds to GT, gather matched trajectories
                if gtb.shape[0] > 0 and 'trajectories' in out:
                    pidx, gidx = matcher._match(out['anchor'][0], out['cls_logits'][0], gtb, gtl)
                    if len(pidx) > 0:
                        traj = out['trajectories'][0][pidx.to(dev)]
                        ml   = out['traj_mode_logits'][0][pidx.to(dev)]
                        mm.update(traj, ml, gtf[gidx.to(dev)], gtm[gidx.to(dev)])
                # planning: ego metrics for the GT command
                if 'ego_traj' in out:
                    ego_f = frame['ego_future'].unsqueeze(0).to(dev)
                    ego_m = frame['ego_future_mask'].unsqueeze(0).to(dev)
                    cmd   = frame['command'].view(1).to(dev)
                    pm.update(out['ego_traj'], out['ego_logits'], cmd, ego_f, ego_m,
                              agent_future=gtf, agent_mask=gtm)

    m = mm.compute(); pl = pm.compute()
    print(f"\n{'='*52}")
    print(f"  MOTION (agents, n={m['count']})")
    print(f"    minADE       : {m['minADE']:.4f}")
    print(f"    minFDE       : {m['minFDE']:.4f}")
    print(f"    brier-minFDE : {m['brier_minFDE']:.4f}")
    print(f"    miss-rate    : {m['MR']:.4f}")
    print(f"  PLANNING (ego, n={pl['count']})")
    print(f"    L2  @1s/2s/3s: {pl['L2@1s']:.3f} / {pl['L2@2s']:.3f} / {pl['L2@3s']:.3f}")
    print(f"    col @1s/2s/3s: {pl['col@1s']:.3f} / {pl['col@2s']:.3f} / {pl['col@3s']:.3f}")
    print(f"{'='*52}")


if __name__ == '__main__':
    main()
