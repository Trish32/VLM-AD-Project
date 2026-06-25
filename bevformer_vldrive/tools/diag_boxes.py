#!/usr/bin/env python3
"""
Diagnostic: compare top decoded predictions with GT annotations for a single frame.
Prints LiDAR-frame position, global position, w/l sizes, and yaw for both.
"""
import math, sys
import numpy as np
import torch
from pathlib import Path
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from model import BEVFormerTiny
from data  import NuScenesMiniLoader
from eval  import _build_remap

PC_RANGE    = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
CLASS_NAMES = ['car','truck','construction_vehicle','bus','trailer',
               'barrier','motorcycle','bicycle','pedestrian','traffic_cone']

def main():
    model = BEVFormerTiny(pretrained_backbone=False)
    model.eval()
    ckpt  = torch.load('model/checkpoints/bevformer_tiny_fp16_epoch_24.pth',
                       map_location='cpu')
    raw   = ckpt.get('state_dict', ckpt)
    model.load_state_dict(_build_remap(raw), strict=False)

    loader = NuScenesMiniLoader('/Users/trish/Downloads/nuScenes_miniV1.0')
    nusc   = loader.nusc
    sample = next(iter(loader.iter_scene(scene_idx=0)))
    imgs, img_metas = sample['imgs'], sample['img_metas']
    sample_token    = img_metas[0]['sample_token']

    with torch.no_grad():
        out = model(imgs, img_metas)

    cls  = out['cls_logits'][0].cpu()
    reg  = out['reg_preds'][0].cpu()
    ref  = out['ref_pts'][0].cpu()

    scores, labels = cls.float().sigmoid().max(-1)
    order = scores.argsort(descending=True)[:15]

    nusc_s   = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', nusc_s['data']['LIDAR_TOP'])
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    ego_tx, ego_ty = ego_pose['translation'][:2]
    ego_q   = Quaternion(ego_pose['rotation'])
    ego_yaw = ego_q.yaw_pitch_roll[0]
    cos_e, sin_e = math.cos(ego_yaw), math.sin(ego_yaw)

    # lidar2ego for full transform
    lidar_q   = Quaternion(lidar_cs['rotation'])
    lidar_tx  = np.array(lidar_cs['translation'])
    R_l2e     = lidar_q.rotation_matrix
    R_e2g     = ego_q.rotation_matrix
    lidar2global = R_e2g @ R_l2e   # combined rotation

    print(f"\n=== TOP-15 PREDICTIONS (token {sample_token[:8]}) ===")
    print(f"  ego global  tx={ego_tx:.2f}  ty={ego_ty:.2f}"
          f"  yaw_deg={math.degrees(ego_yaw):.1f}")
    print(f"  (LiDAR frame: x=forward, y=left of vehicle)\n")
    print(f"  {'class':<22} {'score':>5}  {'x_lid':>7} {'y_lid':>7}"
          f"  {'gx':>8} {'gy':>8}  {'w':>5} {'l':>5}"
          f"  {'yaw_lid°':>8} {'yaw_glo°':>8}")
    print("  " + "-"*90)

    for i in order:
        r   = reg[i].float()
        p   = ref[i].float()
        sc  = float(scores[i])
        lbl = int(labels[i])
        x_lid = float(p[0]) * (PC_RANGE[3]-PC_RANGE[0]) + PC_RANGE[0]
        y_lid = float(p[1]) * (PC_RANGE[4]-PC_RANGE[1]) + PC_RANGE[1]
        # ego-frame rotation only (matches current visualizer)
        gx_approx = cos_e * x_lid - sin_e * y_lid + ego_tx
        gy_approx = sin_e * x_lid + cos_e * y_lid + ego_ty
        # full lidar2global rotation
        pos_l = np.array([x_lid, y_lid, 0.0])
        pos_g = lidar2global @ pos_l + ego_tx * np.array([1,0,0]) + ego_ty * np.array([0,1,0])
        gx_full = float(pos_g[0])
        gy_full = float(pos_g[1])

        w       = float(r[2].exp())
        l       = float(r[3].exp())
        yaw_lid = math.atan2(float(r[6]), float(r[7]))
        yaw_glo = ego_yaw + yaw_lid      # current visualizer
        yaw_glo_full = yaw_lid + lidar_q.yaw_pitch_roll[0] + ego_yaw  # with l2e yaw

        print(f"  {CLASS_NAMES[lbl]:<22} {sc:>5.3f}"
              f"  {x_lid:>7.2f} {y_lid:>7.2f}"
              f"  {gx_approx:>8.2f} {gy_approx:>8.2f}"
              f"  {w:>5.2f} {l:>5.2f}"
              f"  {math.degrees(yaw_lid):>8.1f} {math.degrees(yaw_glo):>8.1f}")

    print(f"\n=== GT ANNOTATIONS ===")
    print(f"  (nuScenes: size=[width, length, height],"
          f" yaw measured CCW from global +x)\n")
    print(f"  {'category':<40}  {'gx':>8} {'gy':>8}  {'w':>5} {'l':>5}"
          f"  {'yaw_glo°':>8}")
    print("  " + "-"*80)

    for ann_tok in nusc_s['anns']:
        ann   = nusc.get('sample_annotation', ann_tok)
        tx, ty = ann['translation'][:2]
        w_a, l_a = ann['size'][0], ann['size'][1]
        yaw_a = Quaternion(ann['rotation']).yaw_pitch_roll[0]
        print(f"  {ann['category_name']:<40}  {tx:>8.2f} {ty:>8.2f}"
              f"  {w_a:>5.2f} {l_a:>5.2f}  {math.degrees(yaw_a):>8.1f}")

if __name__ == '__main__':
    main()
