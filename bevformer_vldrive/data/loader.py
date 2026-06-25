"""
nuScenes mini v1.0 data loader — pure Python + nuscenes-devkit.

Loads each sample as a dict containing:
  imgs       : (num_cams, 3, H, W)  float32 [0, 255]
  img_metas  : list[dict] with keys
                 lidar2img    : list of (4, 4) numpy arrays  (one per cam)
                 img_shape    : list of (H, W, 3) tuples     (one per cam)
                 can_bus      : (18,) numpy array

Camera order matches BEVFormer-Tiny training convention:
  CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
  CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

CAM_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

# Match BEVFormer-Tiny training normalisation (applied *after* loading)
IMG_CONTENT_H = 450              # 900 × 0.5 — pre-pad resize target (official RandomScaleImageMultiViewImage scales=[0.5])
IMG_H, IMG_W  = 480, 800         # padded to multiple of 32 (official PadMultiViewImage size_divisor=32)

PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _get_lidar2img(nusc: NuScenes, sample_data_token: str) -> np.ndarray:
    """Build the 4×4 lidar-to-image projection matrix for one camera."""
    sd   = nusc.get('sample_data', sample_data_token)  # camera sample_data
    cs   = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])  # camera sensor
    ep   = nusc.get('ego_pose', sd['ego_pose_token'])  # ego pose at camera timestamp

    # Camera intrinsic K (3×3 -> 4×4 padded)
    K = np.eye(4, dtype=np.float64)
    K[:3, :3] = np.array(cs['camera_intrinsic'])  # (3, 3)
    # Scale K for resized image: K is scaled for the pre-pad content size (800×450),
    # matching official RandomScaleImageMultiViewImage(scales=[0.5]).
    # img_shape stores the padded size (480) but K must use the content size.
    scale_x = IMG_W / 1600.0            # = 0.5
    scale_y = IMG_CONTENT_H / 900.0    # = 0.5  (NOT IMG_H/900 which would be wrong)
    K[0] *= scale_x
    K[1] *= scale_y

    # Camera extrinsic R, T: sensor -> ego 
    R_cam = Quaternion(cs['rotation']).rotation_matrix  # (3, 3)
    t_cam = np.array(cs['translation'])  # (3,)
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = R_cam
    cam2ego[:3,  3] = t_cam
    ego2cam = np.linalg.inv(cam2ego)  # ego -> cam

    # Ego -> global 
    R_ego = Quaternion(ep['rotation']).rotation_matrix  # (3, 3)
    t_ego = np.array(ep['translation'])  # (3,)
    ego2global = np.eye(4)
    ego2global[:3, :3] = R_ego
    ego2global[:3,  3] = t_ego
    global2ego = np.linalg.inv(ego2global)  # global -> ego

    # Lidar sensor token comes from the reference sample
    lidar_sd  = nusc.get('sample_data', _get_lidar_token(nusc, sd['sample_token']))  # lidar sample_data
    lidar_cs  = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])  # lidar sensor
    lidar_ep  = nusc.get('ego_pose', lidar_sd['ego_pose_token'])  # ego pose at lidar timestamp

    R_lidar = Quaternion(lidar_cs['rotation']).rotation_matrix
    t_lidar = np.array(lidar_cs['translation'])
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = R_lidar
    lidar2ego[:3,  3] = t_lidar  # lidar -> ego

    R_lego = Quaternion(lidar_ep['rotation']).rotation_matrix
    t_lego = np.array(lidar_ep['translation'])
    lego2global = np.eye(4)
    lego2global[:3, :3] = R_lego
    lego2global[:3,  3] = t_lego  # lidar_ego -> global

    # lidar -> lidar_ego -> global -> cam_ego -> cam -> pixel
    lidar2img = K @ ego2cam @ global2ego @ lego2global @ lidar2ego
    return lidar2img.astype(np.float32)


def _get_lidar_token(nusc: NuScenes, sample_token: str) -> str:
    sample = nusc.get('sample', sample_token)
    return sample['data']['LIDAR_TOP']


def _can_bus_signal(nusc: NuScenes, sample_token: str,
                    prev_sample_token: str | None) -> np.ndarray:
    """
    Build an 18-dim CAN-bus-like signal from ego-pose delta.
    Layout mirrors BEVFormer's can_bus field:
      [dx, dy, dz, vx, vy, vz, ax, ay, az,
       roll, pitch, yaw, d_roll, d_pitch, d_yaw,
       speed, ?, rotation_angle_for_bev_rotate]
    """
    sig = np.zeros(18, dtype=np.float32)

    sample  = nusc.get('sample', sample_token)
    lidar_t = sample['data']['LIDAR_TOP']
    lsd     = nusc.get('sample_data', lidar_t)
    ep_cur  = nusc.get('ego_pose', lsd['ego_pose_token'])

    R_cur = Quaternion(ep_cur['rotation'])
    t_cur = np.array(ep_cur['translation'], dtype=np.float32)
    yaw_cur = R_cur.yaw_pitch_roll[0]

    if prev_sample_token is not None:
        prev_sample = nusc.get('sample', prev_sample_token)
        prev_lt     = prev_sample['data']['LIDAR_TOP']
        prev_lsd    = nusc.get('sample_data', prev_lt)
        ep_prev     = nusc.get('ego_pose', prev_lsd['ego_pose_token'])
        R_prev      = Quaternion(ep_prev['rotation'])
        t_prev      = np.array(ep_prev['translation'], dtype=np.float32)
        yaw_prev    = R_prev.yaw_pitch_roll[0]

        delta_t = t_cur - t_prev
        sig[0]  = float(delta_t[0])    # dx: ego translation x (metres, global frame)
        sig[1]  = float(delta_t[1])    # dy: ego translation y
        sig[2]  = float(delta_t[2])    # dz
        delta_yaw = yaw_cur - yaw_prev
        sig[14] = float(delta_yaw)     # d_yaw (index 14)
        sig[-2] = float(yaw_cur)       # absolute ego yaw (radians)  ← used for rotation
        # BEV rotation angle used in rotate_prev_bev: delta in degrees
        sig[-1] = float(np.degrees(delta_yaw))  # delta_yaw in degrees ← used for prev_bev rotation
    else:
        sig[-2] = float(yaw_cur)

    return sig


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class NuScenesMiniLoader:
    """
    Iterates over all samples in nuScenes mini v1.0 in chronological order
    (scene by scene, keyframe by keyframe).

    Usage:
        loader = NuScenesMiniLoader('/path/to/nuScenes_miniV1.0')
        for batch in loader:
            imgs      = batch['imgs']       # (1, 6, 3, H, W) torch.float32
            img_metas = batch['img_metas']  # list[dict]
            prev_bev  = batch['prev_bev']   # None or tensor from last frame
            ...
    """

    def __init__(self, dataroot: str, version: str = 'v1.0-mini'):
        self.nusc     = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = Path(dataroot)

    def _load_image(self, sample_data_token: str) -> np.ndarray:
        """Load and resize one camera image to (IMG_H, IMG_W, 3) uint8."""
        sd   = self.nusc.get('sample_data', sample_data_token)
        path = self.dataroot / sd['filename']
        img  = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)  # RGB, matches official BEVFormer
        img  = cv2.resize(img, (IMG_W, IMG_CONTENT_H), interpolation=cv2.INTER_LINEAR)
        # Pad to multiple of 32 — matches official PadMultiViewImage(size_divisor=32)
        pad_h = IMG_H - IMG_CONTENT_H   # = 30
        img   = np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode='constant')
        return img

    def _process_sample(
        self,
        sample_token: str,
        prev_token:   str | None,
    ) -> dict:
        sample = self.nusc.get('sample', sample_token)

        imgs_list       = []
        lidar2img_list  = []
        img_shape_list  = []

        for cam in CAM_NAMES:
            sd_token = sample['data'][cam]
            img_np   = self._load_image(sd_token)
            imgs_list.append(img_np)
            l2i = _get_lidar2img(self.nusc, sd_token)
            lidar2img_list.append(l2i)
            img_shape_list.append(img_np.shape[:2] + (3,))   # (H, W, 3)

        # (num_cams, 3, H, W) float32
        imgs_np = np.stack(imgs_list, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
        imgs_t  = torch.from_numpy(imgs_np)

        can_bus  = _can_bus_signal(self.nusc, sample_token, prev_token)

        img_metas = [{
            'lidar2img':  lidar2img_list,
            'img_shape':  img_shape_list,
            'can_bus':    can_bus,
            'sample_token': sample_token,
        }]

        return {
            'imgs':      imgs_t.unsqueeze(0),   # (1, num_cams, 3, H, W)
            'img_metas': img_metas,
        }

    def iter_scene(self, scene_idx: int = 0) -> Iterator[dict]:
        """Yield processed samples for a single scene in order."""
        scene    = self.nusc.scene[scene_idx]
        token    = scene['first_sample_token']
        prev_tok = None

        while token:
            sample  = self.nusc.get('sample', token)
            data    = self._process_sample(token, prev_tok)
            yield data
            prev_tok = token
            token    = sample['next'] if sample['next'] else None

    def __iter__(self) -> Iterator[dict]:
        """Iterate all scenes, all samples."""
        for idx in range(len(self.nusc.scene)):
            yield from self.iter_scene(scene_idx=idx)

    def __len__(self) -> int:
        return sum(s['nbr_samples'] for s in self.nusc.scene)
