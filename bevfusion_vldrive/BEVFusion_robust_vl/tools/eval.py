"""
Evaluate the BEVFusion-PP port on nuScenes mini_val: run inference over all
samples, convert predictions LiDAR -> global, write a nuScenes submission JSON,
and run the official detection metric (mAP / NDS).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from model.bevfusion_pp import BEVF_FasterRCNN
from model.checkpoint import load_bevfusion_pp
from data.loader import NuScenesPPLoader

DEFAULT_ATTR = {
    'car': 'vehicle.parked', 'pedestrian': 'pedestrian.moving',
    'trailer': 'vehicle.parked', 'truck': 'vehicle.parked',
    'bus': 'vehicle.moving', 'motorcycle': 'cycle.without_rider',
    'construction_vehicle': 'vehicle.parked', 'bicycle': 'cycle.without_rider',
    'barrier': '', 'traffic_cone': '',
}


def boxes_to_nusc(bboxes, scores, labels):
    """LiDAR boxes [x,y,z,w,l,h,yaw,vx,vy] -> list[NuScenesBox] (lidar frame)."""
    out = []
    b = bboxes.cpu().numpy()
    s = scores.cpu().numpy()
    lab = labels.cpu().numpy()
    for i in range(b.shape[0]):
        x, y, z, w, l, h, yaw = b[i, :7]
        vx, vy = b[i, 7], b[i, 8]
        gcz = z + h / 2.0
        nus_yaw = -yaw - np.pi / 2
        quat = Quaternion(axis=[0, 0, 1], radians=nus_yaw)
        box = NuScenesBox([x, y, gcz], [w, l, h], quat,
                          label=int(lab[i]), score=float(s[i]),
                          velocity=(vx, vy, 0.0))
        out.append(box)
    return out


def lidar_to_global(boxes, lidar2ego_t, lidar2ego_r, ego2global_t, ego2global_r,
                    eval_cfg, classes):
    out = []
    for box in boxes:
        # to ego frame
        box.rotate(Quaternion(lidar2ego_r))
        box.translate(np.array(lidar2ego_t))
        # filter by class range in EGO frame (matches mmdet3d)
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = eval_cfg.class_range[classes[box.label]]
        if radius > det_range:
            continue
        # to global frame
        box.rotate(Quaternion(ego2global_r))
        box.translate(np.array(ego2global_t))
        out.append(box)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=C.CHECKPOINT)
    ap.add_argument('--dataroot', default=C.DATAROOT)
    ap.add_argument('--eval-set', default='mini_val')
    ap.add_argument('--device', default=None)
    ap.add_argument('--limit', type=int, default=0, help='limit #samples (debug)')
    ap.add_argument('--out', default='/Users/trish/VLMProjects/bevfusion_vldrive/BEVFusion_vl/eval_out')
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    model = BEVF_FasterRCNN(C, device=device)
    load_bevfusion_pp(model, args.checkpoint, map_location="cpu")
    model.to(device).eval()

    loader = NuScenesPPLoader(args.dataroot, C.VERSION)
    nusc = loader.nusc
    tokens = loader.sample_tokens(args.eval_set)
    if args.limit:
        tokens = tokens[:args.limit]
    print(f"{len(tokens)} samples, device {device}")

    eval_cfg = config_factory('detection_cvpr_2019')
    results = {}
    for idx, token in enumerate(tokens):
        frame = loader.get_frame(token)
        points = [frame['points'].to(device)]
        img = frame['img'].to(device)
        l2i = [frame['lidar2img']]
        t0 = time.time()
        bboxes, scores, labels = model.simple_test(points, img, l2i)
        # transforms
        sample = nusc.get('sample', token)
        lsd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs = nusc.get('calibrated_sensor', lsd['calibrated_sensor_token'])
        ep = nusc.get('ego_pose', lsd['ego_pose_token'])
        nus_boxes = boxes_to_nusc(bboxes, scores, labels)
        nus_boxes = lidar_to_global(
            nus_boxes, cs['translation'], cs['rotation'],
            ep['translation'], ep['rotation'], eval_cfg, C.CLASS_NAMES)
        annos = []
        for box in nus_boxes:
            name = C.CLASS_NAMES[box.label]
            attr = DEFAULT_ATTR[name]
            annos.append({
                'sample_token': token,
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': float(box.score),
                'attribute_name': attr,
            })
        results[token] = annos
        print(f"[{idx+1}/{len(tokens)}] {token} dets={len(annos)} ({time.time()-t0:.1f}s)")

    submission = {
        'meta': {'use_camera': True, 'use_lidar': True, 'use_radar': False,
                 'use_map': False, 'use_external': False},
        'results': results,
    }
    res_path = os.path.join(args.out, 'results_nusc.json')
    with open(res_path, 'w') as f:
        json.dump(submission, f)
    print("wrote", res_path)

    ev = DetectionEval(nusc, config=eval_cfg, result_path=res_path,
                       eval_set=args.eval_set, output_dir=args.out, verbose=True)
    metrics = ev.main(render_curves=False)
    print("\n===== mAP:", metrics['mean_ap'], "NDS:", metrics['nd_score'], "=====")


if __name__ == "__main__":
    main()
