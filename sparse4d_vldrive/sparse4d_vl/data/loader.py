"""
nuScenes mini loader for Sparse4D.

Per-frame output dict:
  imgs           : (N_cam, 3, H, W)  float32 [0, 255]
  img_metas      : dict with:
    projection_mat : (N_cam, 4, 4)  float32   lidar→pixel (ego frame → image pixel)
    ego2global     : (4, 4)         float32   current lidar-ego → global
    timestamp      : float                     lidar timestamp (seconds, for velocity Δt)
    img_wh         : (N_cam, 2)     [W, H]    after resize, for normalising pts_2d

Camera order (nuScenes convention):
  CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
  CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT

Image size: 256 × 704  (H × W) matching Sparse4D-v2 config.
Intrinsics are rescaled accordingly, so projection_mat is correct for the
resized image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

CAM_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

IMG_H, IMG_W = 256, 704   # target image size (H × W)
NUM_CAMS     = len(CAM_NAMES)

# Test-time resize/crop matching reference ResizeCropFlipImage:
#   resize = max(fH/H, fW/W) = max(256/900, 704/1600) = 0.44
#   resized to (704, 396), then crop rows [140:396] (keep bottom fH rows)
#   Top-crop is deliberate: it discards sky, keeps the road/objects in the bottom 256 rows.
SRC_H, SRC_W = 900, 1600
RESIZE       = max(IMG_H / SRC_H, IMG_W / SRC_W)          # 0.44
NEW_W, NEW_H = int(SRC_W * RESIZE), int(SRC_H * RESIZE)   # 704, 396
CROP_H       = NEW_H - IMG_H                               # 140 (from top)
CROP_W       = (NEW_W - IMG_W) // 2                        # 0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rot_trans_to_mat44(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3,  3] = t
    return M


def _build_ego2global(nusc: NuScenes, lidar_token: str) -> np.ndarray:
    """4×4 lidar-ego → global transform at the lidar timestamp."""
    lsd = nusc.get('sample_data', lidar_token)
    ep  = nusc.get('ego_pose', lsd['ego_pose_token'])
    R   = Quaternion(ep['rotation']).rotation_matrix
    t   = np.array(ep['translation'])
    return _rot_trans_to_mat44(R, t).astype(np.float32)


def _build_projection_mat(
    nusc: NuScenes,
    cam_token: str,
    lidar_token: str,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """
    4×4 projection matrix: lidar frame → image pixel (after resize).

    Chain: lidar → lidar_ego → global → cam_ego → cam → pixel
    This matches the BEVFormer lidar2img convention exactly.
    The 6 cameras and the lidar are each sampled at slightly different timestamps within a keyframe, 
    and the ego vehicle is moving. Each sensor therefore has its own ego_pose. 
    You can't just do lidar→ego→cam with one shared ego — that assumes all sensors fired at the same instant. 
    Going lidar_ego → global → cam_ego uses each sensor's own pose and lets the world frame absorb the time offset.
    """
    # Camera data
    cam_sd = nusc.get('sample_data', cam_token)
    cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cam_ep = nusc.get('ego_pose', cam_sd['ego_pose_token'])

    # Lidar data
    lid_sd = nusc.get('sample_data', lidar_token)
    lid_cs = nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])
    lid_ep = nusc.get('ego_pose', lid_sd['ego_pose_token'])

    # Intrinsic K — uniform resize then crop (matching reference test pipeline):
    #   u' = RESIZE * u - CROP_W,  v' = RESIZE * v - CROP_H
    # Important: If you resize the image but forget to adjust K, projection and pixels disagree by exactly the resize/crop — the DFA samples shifted locations everywhere, and again mAP tanks.
    K = np.eye(4, dtype=np.float64)
    K[:3, :3] = np.array(cam_cs['camera_intrinsic'])
    K[0] *= scale_x          # scale_x == scale_y == RESIZE
    K[1] *= scale_y
    K[0, 2] -= CROP_W
    K[1, 2] -= CROP_H

    # cam → cam_ego
    cam2ego = _rot_trans_to_mat44(
        Quaternion(cam_cs['rotation']).rotation_matrix,
        np.array(cam_cs['translation']),
    )
    # Transform: camera ego → camera sensor, Frame after: camera optical
    ego2cam = np.linalg.inv(cam2ego)

    # cam_ego → global
    cam_ego2global = _rot_trans_to_mat44(
        Quaternion(cam_ep['rotation']).rotation_matrix,
        np.array(cam_ep['translation']),
    )
    # Transform: global -> camera ego, Frame after: ego at cam time
    global2cam_ego = np.linalg.inv(cam_ego2global)

    # Transform: lidar → lidar_ego, Frame after: ego at lidar time
    lid2ego = _rot_trans_to_mat44(
        Quaternion(lid_cs['rotation']).rotation_matrix,
        np.array(lid_cs['translation']),
    )

    # Transform: lidar_ego → global
    lid_ego2global = _rot_trans_to_mat44(
        Quaternion(lid_ep['rotation']).rotation_matrix,
        np.array(lid_ep['translation']),
    )

    # Full chain: lidar → pixel
    proj = K @ ego2cam @ global2cam_ego @ lid_ego2global @ lid2ego
    return proj.astype(np.float32)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class NuScenesSparse4DLoader:
    """
    Iterates over nuScenes mini scene-by-scene, keyframe-by-keyframe.

    Usage:
        loader = NuScenesSparse4DLoader('/path/to/nuScenes_miniV1.0')
        for frame in loader.iter_scene(scene_idx=0):
            imgs      = frame['imgs']        # (1, 6, 3, 256, 704) float32
            metas     = frame['img_metas']   # dict
            timestamp = metas['timestamp']   # float (seconds)
    """
    # NUSCENES_MEAN/STD are defined but not applied in the loader. 
    # The mean/std subtraction happens inside the model (_normalize in sparse4d_base.py)
    NUSCENES_MEAN = np.array([123.675, 116.280, 103.530], dtype=np.float32)  # RGB
    NUSCENES_STD  = np.array([ 58.395,  57.120,  57.375], dtype=np.float32)

    def __init__(self, dataroot: str, version: str = 'v1.0-mini'):
        self.nusc     = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = Path(dataroot)

    def _load_image(self, cam_token: str) -> np.ndarray:
        sd   = self.nusc.get('sample_data', cam_token)
        path = self.dataroot / sd['filename']
        img  = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        # Uniform resize then crop (no aspect-ratio distortion), matching
        # the reference ResizeCropFlipImage test-time pipeline.
        img  = cv2.resize(img, (NEW_W, NEW_H), interpolation=cv2.INTER_LINEAR)
        img  = img[CROP_H:CROP_H + IMG_H, CROP_W:CROP_W + IMG_W]
        return img.astype(np.float32)   # (H, W, 3) RGB float32 [0, 255]

    def _process_sample(self, sample_token: str) -> dict:
        sample      = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lid_sd      = self.nusc.get('sample_data', lidar_token)

        # Uniform scale factor for intrinsics (resize-then-crop pipeline)
        scale_x = RESIZE
        scale_y = RESIZE

        imgs_list   = []
        proj_list   = []

        for cam in CAM_NAMES:
            cam_token = sample['data'][cam]
            img_np    = self._load_image(cam_token)
            imgs_list.append(img_np)
            proj = _build_projection_mat(
                self.nusc, cam_token, lidar_token, scale_x, scale_y
            )  # Lidar -> pixel
            proj_list.append(proj)

        # (N_cam, H, W, 3) → (N_cam, 3, H, W)
        imgs_np = np.stack(imgs_list).transpose(0, 3, 1, 2)   # float32 [0,255]
        imgs_t  = torch.from_numpy(imgs_np).unsqueeze(0)      # Batch is hardcoded to 1 (unsqueeze(0)) (1, N_cam, 3, H, W)

        projection_mat = np.stack(proj_list)                   # (N_cam, 4, 4)
        ego2global     = _build_ego2global(self.nusc, lidar_token)  # (4, 4)

        # lidar sensor → ego transform (needed to convert predictions to ego/global frame)
        lid_cs   = self.nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])
        lidar2ego = _rot_trans_to_mat44(
            Quaternion(lid_cs['rotation']).rotation_matrix,
            np.array(lid_cs['translation']),
        ).astype(np.float32)

        img_wh = np.array([[IMG_W, IMG_H]] * NUM_CAMS, dtype=np.float32)  # (N_cam, 2)

        # The bank needs to reconstruct lidar→global at each timestamp, lidar2global = ego2global @ lidar2ego
        img_metas = {
            'projection_mat': projection_mat,      # (N_cam, 4, 4) float32
            'ego2global':     ego2global,           # (4, 4)        float32
            'lidar2ego':      lidar2ego,            # (4, 4)        float32  lidar sensor → ego
            'timestamp':      lid_sd['timestamp'] / 1e6,  # divided to seconds, the Δt between frames scales the velocity term in temporal projection
            'img_wh':         img_wh,               # (N_cam, 2) [W, H] the post-crop 704×256, used by project_points to normalize pixel coords to [0,1] before grid_sample
            'sample_token':   sample_token,
        }

        return {
            'imgs':      imgs_t,       # (1, N_cam, 3, H, W) float32 [0,255]
            'img_metas': img_metas,
        }

    # walks the first_sample_token → next linked list keyframe by keyframe, in temporal order
    def iter_scene(self, scene_idx: int = 0) -> Iterator[dict]:
        scene = self.nusc.scene[scene_idx]
        token = scene['first_sample_token']
        while token:
            sample = self.nusc.get('sample', token)
            yield self._process_sample(token)
            token = sample['next'] if sample['next'] else None

    def __iter__(self) -> Iterator[dict]:
        for idx in range(len(self.nusc.scene)):
            yield from self.iter_scene(scene_idx=idx)

    def __len__(self) -> int:
        return sum(s['nbr_samples'] for s in self.nusc.scene)
