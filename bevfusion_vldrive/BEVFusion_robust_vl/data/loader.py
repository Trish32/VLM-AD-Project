"""
nuScenes-mini data loader for the BEVFusion-PP port, using nuscenes-devkit
directly (no mmdet3d). Reproduces the bevf_pp test pipeline:

  Points: LoadPointsFromFile(5d) + LoadPointsFromMultiSweeps(10) ->
          [x, y, z, dt]  (intensity dropped; dt = current_ts - sweep_ts)
  Images: 6 cams -> MyResize(keep_ratio to (800,448)) -> MyNormalize(mean/std,
          to_rgb) -> MyPad(size_divisor=32).
  lidar2img: built in ORIGINAL (900x1600) pixel coordinates (the LSS frustum
             lives in original-resolution pixels; MyResize does NOT rescale it).

Each frame dict:
  points    : (N, 4) float32 torch tensor (lidar frame)
  img       : (6, 3, H, W) float32 torch tensor (normalized, padded)
  lidar2img : (6, 4, 4) float32 np array  (lidar -> original pixel)
  sample_token : str
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

CAM_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)   # RGB
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
IMG_SCALE = (800, 448)     # (max_w_bound, max_h_bound) for keep-ratio resize
SIZE_DIVISOR = 32
SWEEPS_NUM = 10


def _mat44(R, t):
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _sensor2global(nusc, sd_token):
    sd = nusc.get('sample_data', sd_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    sensor2ego = _mat44(Quaternion(cs['rotation']).rotation_matrix, np.array(cs['translation']))
    ego2global = _mat44(Quaternion(ep['rotation']).rotation_matrix, np.array(ep['translation']))
    return ego2global @ sensor2ego, cs, ep


def _imrescale_keepratio(img, scale):
    """mmcv.imrescale keep_ratio: scale so the image fits within scale bound."""
    h, w = img.shape[:2]
    max_long, max_short = max(scale), min(scale)
    factor = min(max_long / max(h, w), max_short / min(h, w))
    new_w, new_h = int(w * factor + 0.5), int(h * factor + 0.5)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _pad_to_divisor(img, divisor):
    h, w = img.shape[:2]
    ph = int(math.ceil(h / divisor)) * divisor
    pw = int(math.ceil(w / divisor)) * divisor
    out = np.zeros((ph, pw, img.shape[2]), dtype=img.dtype)
    out[:h, :w] = img
    return out


class NuScenesPPLoader:
    def __init__(self, dataroot, version='v1.0-mini'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = Path(dataroot)

    # -- splits ----------------------------------------------------------
    def scene_tokens(self, split='mini_val'):
        splits = create_splits_scenes()
        names = set(splits[split])
        return [s['token'] for s in self.nusc.scene if s['name'] in names]

    def sample_tokens(self, split='mini_val'):
        toks = []
        for st in self.scene_tokens(split):
            scene = self.nusc.get('scene', st)
            s = scene['first_sample_token']
            while s:
                toks.append(s)
                s = self.nusc.get('sample', s)['next']
        return toks

    # -- points ----------------------------------------------------------
    def _load_points(self, sample):
        lidar_token = sample['data']['LIDAR_TOP']
        lsd = self.nusc.get('sample_data', lidar_token)
        ref_global, ref_cs, ref_ep = _sensor2global(self.nusc, lidar_token)
        global2ref_lidar = np.linalg.inv(ref_global)
        ref_ts = sample['timestamp'] / 1e6

        def read_bin(path):
            pts = np.fromfile(str(self.dataroot / path), dtype=np.float32)
            return pts.reshape(-1, 5)

        # current keyframe
        cur = read_bin(lsd['filename'])
        cur[:, 4] = 0.0  # dt = 0
        all_pts = [cur[:, [0, 1, 2, 4]]]

        # gather up to SWEEPS_NUM previous non-keyframe sweeps
        sweep_sd = lsd
        n = 0
        while n < SWEEPS_NUM:
            if sweep_sd['prev'] == '':
                break
            sweep_sd = self.nusc.get('sample_data', sweep_sd['prev'])
            pts = read_bin(sweep_sd['filename'])
            sweep_global, _, _ = _sensor2global(self.nusc, sweep_sd['token'])
            T = global2ref_lidar @ sweep_global  # sweep lidar -> ref lidar
            xyz = pts[:, :3] @ T[:3, :3].T + T[:3, 3]
            dt = ref_ts - sweep_sd['timestamp'] / 1e6
            out = np.concatenate([xyz, np.full((xyz.shape[0], 1), dt, dtype=np.float32)], 1)
            all_pts.append(out.astype(np.float32))
            n += 1

        pts = np.concatenate(all_pts, 0).astype(np.float32)
        return torch.from_numpy(pts)

    # -- images + projections -------------------------------------------
    def _build_lidar2img(self, cam_token, lidar_token):
        cam_sd = self.nusc.get('sample_data', cam_token)
        cam2global, cam_cs, _ = _sensor2global(self.nusc, cam_token)
        lid2global, _, _ = _sensor2global(self.nusc, lidar_token)
        K = np.eye(4, dtype=np.float64)
        K[:3, :3] = np.array(cam_cs['camera_intrinsic'])  # ORIGINAL intrinsic
        lidar2cam = np.linalg.inv(cam2global) @ lid2global
        return (K @ lidar2cam).astype(np.float32)

    def _load_image(self, cam_token):
        sd = self.nusc.get('sample_data', cam_token)
        img = cv2.imread(str(self.dataroot / sd['filename']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = _imrescale_keepratio(img, IMG_SCALE)
        img = (img - IMG_MEAN) / IMG_STD
        img = _pad_to_divisor(img, SIZE_DIVISOR)
        return img  # (H, W, 3)

    def _load_images(self, sample):
        lidar_token = sample['data']['LIDAR_TOP']
        imgs, l2i = [], []
        for cam in CAM_NAMES:
            cam_token = sample['data'][cam]
            imgs.append(self._load_image(cam_token))
            l2i.append(self._build_lidar2img(cam_token, lidar_token))
        imgs = np.stack(imgs).transpose(0, 3, 1, 2)  # (6, 3, H, W)
        return torch.from_numpy(imgs.astype(np.float32)), np.stack(l2i)

    # -- public ----------------------------------------------------------
    def get_frame(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        points = self._load_points(sample)
        img, lidar2img = self._load_images(sample)
        return {
            'points': points,
            'img': img,
            'lidar2img': lidar2img,
            'sample_token': sample_token,
        }
