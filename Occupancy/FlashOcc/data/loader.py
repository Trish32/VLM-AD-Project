"""nuScenes-mini data loader for FlashOcc (BEVDetOCC), pure devkit.

Produces ``img_inputs`` tuples in the exact format the model expects:
    (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda)

The image preprocessing reproduces the *test* branch of the official
``PrepareImageInputs`` pipeline (no random augmentation):

    resize = input_W / src_W = 704 / 1600 = 0.44
    resize_dims = (704, 396);  crop = (0, 140, 704, 396) -> 704x256
    post_rot = diag(0.44, 0.44, 1);  post_tran = (0, -140, 0)
    bda = I(3)

Normalisation mirrors mmcv ``imnormalize(..., to_rgb=True)`` which reverses the
channel order(swaps channel 0 ↔ channel 2) of the (already-RGB) PIL image before subtracting the ImageNet
mean -- the FlashOcc checkpoint was trained with this exact (quirky) op so we
must replicate it bit-for-bit.
# BEVDet quirk: PIL gives RGB, but mmcv imnormalize(to_rgb=True) blindly
# reverses channels (its cvtColor assumes BGR input) -> the net was TRAINED
# on BGR. Reproduce the swap; do NOT "fix" it (verified: +5 mIoU vs no-swap).
img = img[..., ::-1]

"""
import os
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

INPUT_SIZE = (256, 704)     # (H, W)
SRC_SIZE = (900, 1600)      # (H, W)


def mmlab_normalize(pil_img):
    """Replicate mmcv imnormalize with to_rgb=True on an RGB PIL image."""
    img = np.array(pil_img, dtype=np.float32)        # np.array(PIL) is RGB, HWC
    img = img[..., ::-1]                             # reverses the last axis (channels) — the pure-NumPy equivalent of that cvtColor swap BGR<->RGB
    img = (img - IMG_MEAN) / IMG_STD
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)


def _test_img_transform():
    """Return (resize_dims, crop, post_rot 3x3, post_tran 3) for test mode."""
    fH, fW = INPUT_SIZE
    H, W = SRC_SIZE
    resize = float(fW) / float(W)                    # 0.44
    resize_dims = (int(W * resize), int(H * resize))  # (704, 396)
    newW, newH = resize_dims
    crop_h = int((1 - 0.0) * newH) - fH              # 140
    crop_w = int(max(0, newW - fW) / 2)              # 0
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    post_rot = torch.eye(3)
    post_rot[:2, :2] *= resize
    post_tran = torch.zeros(3)
    post_tran[0] = -crop[0]
    post_tran[1] = -crop[1]
    return resize_dims, crop, post_rot, post_tran


def _mat4(quat_wxyz, translation):
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = Quaternion(quat_wxyz).rotation_matrix
    m[:3, 3] = translation
    return torch.from_numpy(m)


class NuScenesOccLoader:
    def __init__(self, dataroot, version='v1.0-mini', verbose=False):
        self.nusc = NuScenes(version=version, dataroot=dataroot,
                             verbose=verbose)
        self.samples = list(self.nusc.sample)
        self.resize_dims, self.crop, self.post_rot, self.post_tran = \
            _test_img_transform()

    def __len__(self):
        return len(self.samples)

    def scene_indices(self, scene_name):
        """Return sample indices belonging to a scene (e.g. 'scene-0103')."""
        out = []
        for i, s in enumerate(self.samples):
            sc = self.nusc.get('scene', s['scene_token'])
            if sc['name'] == scene_name:
                out.append(i)
        return out

    def get_img_inputs(self, index):
        sample = self.samples[index]
        imgs, sensor2egos, ego2globals, intrins = [], [], [], []
        post_rots, post_trans = [], []

        for cam in CAMS:
            sd = self.nusc.get('sample_data', sample['data'][cam])
            cs = self.nusc.get('calibrated_sensor',
                               sd['calibrated_sensor_token'])
            ego = self.nusc.get('ego_pose', sd['ego_pose_token'])

            sensor2ego = _mat4(cs['rotation'], cs['translation'])
            ego2global = _mat4(ego['rotation'], ego['translation'])
            intrin = torch.tensor(np.array(cs['camera_intrinsic'],
                                           dtype=np.float32))

            path = os.path.join(self.nusc.dataroot, sd['filename'])
            img = Image.open(path)
            img = img.resize(self.resize_dims).crop(self.crop)
            imgs.append(mmlab_normalize(img))

            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            intrins.append(intrin)
            post_rots.append(self.post_rot.clone())
            post_trans.append(self.post_tran.clone())

        imgs = torch.stack(imgs)                 # (6, 3, 256, 704)
        sensor2egos = torch.stack(sensor2egos)   # (6, 4, 4)
        ego2globals = torch.stack(ego2globals)   # (6, 4, 4)
        intrins = torch.stack(intrins)           # (6, 3, 3)
        post_rots = torch.stack(post_rots)       # (6, 3, 3)
        post_trans = torch.stack(post_trans)     # (6, 3)
        bda = torch.eye(3)                       # test: identity

        return (imgs, sensor2egos, ego2globals, intrins, post_rots,
                post_trans, bda)

    def get_batched(self, index):
        """Same as get_img_inputs but with a leading batch dim of 1."""
        inputs = self.get_img_inputs(index)
        return tuple(t.unsqueeze(0) for t in inputs)

    def sample_token(self, index):
        return self.samples[index]['token']

    # ---- LiDAR ego origins for RayIoU -----------------------------------
    def _lidar_poses(self, sample):
        """Return (lidar2ego, ego2global) 4x4 numpy mats for a sample's LiDAR."""
        sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ego = self.nusc.get('ego_pose', sd['ego_pose_token'])
        lidar2ego = _mat4(cs['rotation'], cs['translation']).numpy()
        ego2global = _mat4(ego['rotation'], ego['translation']).numpy()
        return lidar2ego, ego2global

    def get_lidar_origins(self, index, max_sel=8):
        """Ego-frame ray origins for RayIoU (replicates EgoPoseDataset).

        Gathers every keyframe in the sample's scene, transforms each frame's
        LiDAR position into the *reference* sample's ego frame, keeps those
        within +-39 m, and subsamples to at most ``max_sel`` (=8) origins. This
        is what makes RayIoU probe occupancy from several viewpoints.

        Returns: (T, 3) float32 origins in the reference ego frame.
        """
        ref = self.samples[index]
        scene_tok = ref['scene_token']
        scene = [s for s in self.samples if s['scene_token'] == scene_tok]
        scene.sort(key=lambda s: s['timestamp'])

        ref_l2e, ref_e2g = self._lidar_poses(ref)
        ref_global_from_lidar = ref_e2g @ ref_l2e
        ref_lidar_from_global = np.linalg.inv(ref_global_from_lidar)
        ref_ego_from_lidar = ref_l2e

        origins = []
        for s in scene:
            if s['token'] == ref['token']:
                origin_tf = np.zeros(3, dtype=np.float32)
            else:
                l2e, e2g = self._lidar_poses(s)
                ref_from_curr = ref_lidar_from_global @ (e2g @ l2e)
                origin_tf = ref_from_curr[:3, 3].astype(np.float32)
            pad = np.ones(4)
            pad[:3] = origin_tf
            origin_tf = (ref_ego_from_lidar[:3] @ pad).astype(np.float32)
            if abs(origin_tf[0]) < 39 and abs(origin_tf[1]) < 39:
                origins.append(origin_tf)

        if len(origins) > max_sel:
            sel = np.round(np.linspace(0, len(origins) - 1, max_sel)).astype(int)
            origins = [origins[i] for i in sel]
        return np.stack(origins)
