"""
3D detection eval for MIT BEVFusion port on nuScenes mini_val.
Boxes [x,y,z_bottom,w,l,h,yaw,vx,vy] (lidar frame) -> nuScenes global -> mAP/NDS.
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
from model.bevfusion import BEVFusion
from data.loader import NuScenesMITLoader

DEFAULT_ATTR = {
    'car': 'vehicle.parked', 'pedestrian': 'pedestrian.moving', 'trailer': 'vehicle.parked',
    'truck': 'vehicle.parked', 'bus': 'vehicle.moving', 'motorcycle': 'cycle.without_rider',
    'construction_vehicle': 'vehicle.parked', 'bicycle': 'cycle.without_rider',
    'barrier': '', 'traffic_cone': '',
}


def to_global(boxes, scores, labels, cs, ep, eval_cfg, classes):
    out = []
    b = boxes.cpu().numpy()
    s = scores.cpu().numpy()
    lab = labels.cpu().numpy()
    for i in range(b.shape[0]):
        x, y, z, w, l, h, yaw = b[i, :7]
        vx, vy = b[i, 7], b[i, 8]
        box = NuScenesBox([x, y, z + h / 2], [w, l, h],
                          Quaternion(axis=[0, 0, 1], radians=-yaw - np.pi / 2),
                          label=int(lab[i]), score=float(s[i]), velocity=(vx, vy, 0.0))
        box.rotate(Quaternion(cs['rotation']))
        box.translate(np.array(cs['translation']))
        radius = np.linalg.norm(box.center[:2])
        if radius > eval_cfg.class_range[classes[box.label]]:
            continue
        box.rotate(Quaternion(ep['rotation']))
        box.translate(np.array(ep['translation']))
        out.append(box)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='/Users/trish/VLMProjects/bevfusion_vldrive/bevfusion_vl/eval_out_det')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = BEVFusion(C.DetCfg)
    info = model.load_state_dict(torch.load(C.CKPT_DET, map_location='cpu')['state_dict'], strict=False)
    print("load missing/unexpected:", len(info.missing_keys), len(info.unexpected_keys))
    model.to(args.device).eval()

    ld = NuScenesMITLoader(C.DATAROOT)
    nusc = ld.nusc
    eval_cfg = config_factory('detection_cvpr_2019')
    tokens = ld.sample_tokens('mini_val')
    if args.limit:
        tokens = tokens[:args.limit]

    results = {}
    for idx, tok in enumerate(tokens):
        frame = ld.get_frame(tok)
        t = time.time()
        boxes, scores, labels = model(frame)[0]
        sample = nusc.get('sample', tok)
        lsd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs = nusc.get('calibrated_sensor', lsd['calibrated_sensor_token'])
        ep = nusc.get('ego_pose', lsd['ego_pose_token'])
        nb = to_global(boxes, scores, labels, cs, ep, eval_cfg, C.OBJECT_CLASSES)
        annos = []
        for box in nb:
            name = C.OBJECT_CLASSES[box.label]
            annos.append({
                'sample_token': tok, 'translation': box.center.tolist(),
                'size': box.wlh.tolist(), 'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(), 'detection_name': name,
                'detection_score': float(box.score), 'attribute_name': DEFAULT_ATTR[name]})
        results[tok] = annos
        print(f"[{idx+1}/{len(tokens)}] {len(annos)} dets ({time.time()-t:.1f}s)", flush=True)

    sub = {'meta': {'use_camera': True, 'use_lidar': True, 'use_radar': False,
                    'use_map': False, 'use_external': False}, 'results': results}
    rp = os.path.join(args.out, 'results_nusc.json')
    json.dump(sub, open(rp, 'w'))
    ev = DetectionEval(nusc, config=eval_cfg, result_path=rp, eval_set='mini_val',
                       output_dir=args.out, verbose=False)
    metrics = ev.main(render_curves=False)
    print(f"\n===== mAP: {metrics['mean_ap']:.4f}  NDS: {metrics['nd_score']:.4f} =====")


if __name__ == "__main__":
    main()
