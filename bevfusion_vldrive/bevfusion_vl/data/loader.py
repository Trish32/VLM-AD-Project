"""
nuScenes-mini loader for MIT BEVFusion (det + seg), pure devkit (no mmdet3d).

Produces, per keyframe, the tensors the model needs:
  img               (N,3,256,704) ImageNet-normalized
  points            (M,5) [x,y,z,intensity,dt] in lidar frame (9 sweeps)
  camera2lidar      (N,4,4)   lidar<-cam
  lidar2camera      (N,4,4)
  lidar2image       (N,4,4)   intrinsic @ lidar2camera
  camera_intrinsics (N,4,4)
  camera2ego        (N,4,4)
  lidar2ego         (4,4)
  img_aug_matrix    (N,4,4)   ImageAug3D resize/crop (test: resize 0.48, crop)
  lidar_aug_matrix  (4,4)     identity (test)
All geometry chained through the global frame at each sensor's own timestamp.
"""
from __future__ import annotations

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
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SWEEPS_NUM = 9
FINAL_H, FINAL_W = 256, 704
RESIZE = 0.48
SRC_H, SRC_W = 900, 1600


def _mat44(R, t):
    # build a 4x4 homogeneous transform from rotation R (3x3) and translation t (3,)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _sensor2global(nusc, sd_token):
    """Compose sensor->ego->global for one sample_data record. Returns:
      sensor2global = e2g @ s2e   (the full sensor-to-world transform)
      s2e (sensor->ego, calibration), e2g (ego->world, pose at this timestamp), cs (calib dict).
    Routing through global is what makes cross-sensor / cross-time geometry exact."""
    sd = nusc.get('sample_data', sd_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])   # fixed sensor mount
    ep = nusc.get('ego_pose', sd['ego_pose_token'])                     # car pose at THIS timestamp
    s2e = _mat44(Quaternion(cs['rotation']).rotation_matrix, np.array(cs['translation']))  # sensor->ego
    e2g = _mat44(Quaternion(ep['rotation']).rotation_matrix, np.array(ep['translation']))  # ego->global
    return e2g @ s2e, s2e, e2g, cs


def _img_aug_matrix():
    """The 4x4 that encodes the SAME image preprocessing applied to the pixels (resize then
    crop), so geometry can undo it. MUST exactly match _load_image's resize+crop or the
    camera projection and the actual image disagree. resize 0.48 then center-w/bottom-h crop."""
    newW, newH = int(SRC_W * RESIZE), int(SRC_H * RESIZE)   # 1600,900 -> 768, 432
    crop_w = (newW - FINAL_W) // 2                           # 32  (center crop in width)
    crop_h = newH - FINAL_H                                  # 176 (bottom crop in height)
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = RESIZE                # scale x
    M[1, 1] = RESIZE                # scale y
    M[0, 3] = -crop_w              # then shift by the crop offset
    M[1, 3] = -crop_h
    return M, crop_w, crop_h, newW, newH


class NuScenesMITLoader:
    def __init__(self, dataroot, version='v1.0-mini'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = Path(dataroot)

    def sample_tokens(self, split='mini_val'):
        names = set(create_splits_scenes()[split])
        toks = []
        for s in self.nusc.scene:
            if s['name'] not in names:
                continue
            t = s['first_sample_token']
            while t:
                toks.append(t)
                t = self.nusc.get('sample', t)['next']
        return toks

    # -- points (9 sweeps, keep intensity, add dt) --
    def _load_points(self, sample):
        """Aggregate the keyframe lidar sweep + up to 9 previous sweeps into ONE cloud,
        all expressed in the keyframe's lidar frame. Each older sweep is motion-compensated
        (transformed via global) and tagged with Δt = how old it is. (M,5)=[x,y,z,intensity,Δt]."""
        lidar_token = sample['data']['LIDAR_TOP']
        lsd = self.nusc.get('sample_data', lidar_token)
        ref_global, ref_s2e, ref_e2g, _ = _sensor2global(self.nusc, lidar_token)   # keyframe lidar->global
        g2ref = np.linalg.inv(ref_global)                    # global->keyframe lidar
        ref_ts = sample['timestamp'] / 1e6                   # reference time (s)

        def read(path):
            return np.fromfile(str(self.dataroot / path), dtype=np.float32).reshape(-1, 5)

        cur = read(lsd['filename'])
        cur[:, 4] = 0.0                                      # keyframe sweep: Δt = 0
        pts = [cur]
        sd = lsd
        n = 0
        while n < SWEEPS_NUM and sd['prev']:
            sd = self.nusc.get('sample_data', sd['prev'])   # walk back to the previous sweep
            p = read(sd['filename'])
            sg, _, _, _ = _sensor2global(self.nusc, sd['token'])   # this sweep's lidar->global
            T = g2ref @ sg                                   # this sweep -> keyframe lidar (via global)
            p[:, :3] = p[:, :3] @ T[:3, :3].T + T[:3, 3]     # motion-compensate the points
            p[:, 4] = ref_ts - sd['timestamp'] / 1e6         # Δt of this sweep — the network can use point age to reason about moving objects
            pts.append(p)
            n += 1
        return torch.from_numpy(np.concatenate(pts, 0).astype(np.float32))   # (M, 5)

    def _load_image(self, cam_token, crop_w, crop_h, newW, newH):
        """Load + preprocess one camera image: BGR->RGB, resize 0.48, crop to 256x704,
        /255 then ImageNet normalize. MUST mirror _img_aug_matrix exactly (same resize+crop)."""
        sd = self.nusc.get('sample_data', cam_token)
        img = cv2.cvtColor(cv2.imread(str(self.dataroot / sd['filename'])), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)   # resize (matches aug scale)
        img = img[crop_h:crop_h + FINAL_H, crop_w:crop_w + FINAL_W]           # crop (matches aug shift)
        img = img.astype(np.float32) / 255.0                                 # ToTensor scaling
        img = (img - IMG_MEAN) / IMG_STD                                     # ImageNet normalize
        return img  # (256,704,3)

    def get_frame(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lid_global, _, lid_e2g, lid_cs = _sensor2global(self.nusc, lidar_token)   # lidar->global
        lidar2ego = _mat44(Quaternion(lid_cs['rotation']).rotation_matrix,
                           np.array(lid_cs['translation']))
        scene = self.nusc.get('scene', sample['scene_token'])
        location = self.nusc.get('log', scene['log_token'])['location']          # map name for seg GT

        aug, crop_w, crop_h, newW, newH = _img_aug_matrix()
        imgs, cam2lidar, lidar2cam, lidar2img, intrins, cam2ego, img_aug = \
            [], [], [], [], [], [], []
        for cam in CAM_NAMES:
            ct = sample['data'][cam]
            cam_global, cam_s2e, cam_e2g, cam_cs = _sensor2global(self.nusc, ct)   # cam->global
            # lidar -> camera = (global->camera) @ (lidar->global): routes through world frame
            l2c = np.linalg.inv(cam_global) @ lid_global
            K = np.eye(4, dtype=np.float64)
            K[:3, :3] = np.array(cam_cs['camera_intrinsic'])   # 3x3 intrinsics in a 4x4
            imgs.append(self._load_image(ct, crop_w, crop_h, newW, newH))
            lidar2cam.append(l2c)
            cam2lidar.append(np.linalg.inv(l2c))               # used by get_geometry (un-project)
            lidar2img.append(K @ l2c)                          # full lidar->pixel projection (used by _depth_image)
            intrins.append(K)
            cam2ego.append(cam_s2e)
            img_aug.append(aug.copy())                         # same aug for all 6 cams

        imgs = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32))

        def st(a):
            return torch.from_numpy(np.stack(a).astype(np.float32))

        return {
            'sample_token': sample_token,
            'img': imgs,
            'points': self._load_points(sample),
            'camera2lidar': st(cam2lidar),
            'lidar2camera': st(lidar2cam),
            'lidar2image': st(lidar2img),
            'camera_intrinsics': st(intrins),
            'camera2ego': st(cam2ego),
            'lidar2ego': torch.from_numpy(lidar2ego.astype(np.float32)),
            'img_aug_matrix': st(img_aug),
            'lidar_aug_matrix': torch.eye(4),
            'ego2global': torch.from_numpy(lid_e2g.astype(np.float32)),
            'location': location,
        }
