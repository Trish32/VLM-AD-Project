"""
Fine-tune loader: extends the inference loader with ground-truth boxes in the
LiDAR frame (mmdet3d format [x,y,z_bottom, w,l,h, yaw, vx,vy]) and class labels.

Uses nuscenes-devkit `get_sample_data`, which already returns annotation boxes
in the LIDAR_TOP sensor frame, so we only convert box convention (gravity
center -> bottom center, nusc yaw -> mmdet3d lidar yaw).
"""
from __future__ import annotations

import numpy as np
import torch
from nuscenes.eval.detection.utils import category_to_detection_name

from .loader import NuScenesPPLoader

CLASS_TO_IDX = {
    'car': 0, 'truck': 1, 'trailer': 2, 'bus': 3, 'construction_vehicle': 4,
    'bicycle': 5, 'motorcycle': 6, 'pedestrian': 7, 'traffic_cone': 8, 'barrier': 9,
}


class NuScenesPPFinetuneLoader(NuScenesPPLoader):
    def get_gt(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        _, boxes, _ = self.nusc.get_sample_data(lidar_token)  # boxes in lidar frame
        gt_boxes, gt_labels = [], []
        for box in boxes:
            det_name = category_to_detection_name(box.name)
            if det_name is None or det_name not in CLASS_TO_IDX:
                continue
            w, l, h = box.wlh
            cx, cy, cz = box.center  # gravity center
            # nusc yaw in lidar frame -> mmdet3d lidar yaw
            v = box.orientation.rotation_matrix @ np.array([1.0, 0.0, 0.0])
            yaw_nusc = np.arctan2(v[1], v[0])
            yaw = -yaw_nusc - np.pi / 2
            # velocity (global) -> lidar frame; cheap: use sample_annotation velocity
            try:
                vel_global = self.nusc.box_velocity(box.token)[:2]
            except Exception:
                vel_global = np.array([0.0, 0.0])
            vel = np.nan_to_num(vel_global)
            gt_boxes.append([cx, cy, cz - h / 2.0, w, l, h, yaw, vel[0], vel[1]])
            gt_labels.append(CLASS_TO_IDX[det_name])
        if gt_boxes:
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
            gt_labels = torch.tensor(gt_labels, dtype=torch.long)
        else:
            gt_boxes = torch.zeros((0, 9), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.long)
        return gt_boxes, gt_labels

    def get_train_frame(self, sample_token):
        frame = self.get_frame(sample_token)
        gt_boxes, gt_labels = self.get_gt(sample_token)
        frame['gt_boxes'] = gt_boxes
        frame['gt_labels'] = gt_labels
        return frame
