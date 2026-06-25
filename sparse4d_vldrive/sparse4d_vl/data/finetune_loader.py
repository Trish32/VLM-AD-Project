"""
Fine-tuning data loader for Sparse4D on nuScenes mini.

Extends NuScenesSparse4DLoader to additionally load ground-truth 3D boxes
and class labels for each keyframe.

GT box format (matches anchor convention in detection3d.py):
  [x, y, z,  log_w, log_l, log_h,  sin_yaw, cos_yaw,  vx, vy, vz]  (11 dims)

Sizes are converted to log-space so regression targets are directly
comparable to the model's anchor predictions.

Class index order (matches CLASS_NAMES in detection3d.py):
  0: car               5: barrier
  1: truck             6: motorcycle
  2: construction_vehicle  7: bicycle
  3: bus               8: pedestrian
  4: trailer           9: traffic_cone
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from pyquaternion import Quaternion

from .loader import NuScenesSparse4DLoader, CAM_NAMES


# ---------------------------------------------------------------------------
# nuScenes category → class index
# Must match CLASS_NAMES order in model/detection3d.py
# ---------------------------------------------------------------------------

_CATEGORY_MAP: dict[str, int] = {
    'vehicle.car':                      0,
    'vehicle.truck':                    1,
    'vehicle.construction':             2,
    'vehicle.bus.bendy':                3,
    'vehicle.bus.rigid':                3,
    'vehicle.trailer':                  4,
    'movable_object.barrier':           5,
    'vehicle.motorcycle':               6,
    'vehicle.bicycle':                  7,
    'human.pedestrian.adult':           8,
    'human.pedestrian.child':           8,
    'human.pedestrian.construction_worker': 8,
    'human.pedestrian.police_officer':  8,
    'movable_object.trafficcone':       9,
}

# Maximum number of GT boxes per frame (pad/truncate to this)
MAX_GT = 100

# Default future horizon for motion forecasting (keyframes; 12 = 6 s at 2 Hz)
FUTURE_STEPS = 12


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class NuScenesFinetuneLoader(NuScenesSparse4DLoader):
    """
    Drop-in replacement for NuScenesSparse4DLoader that appends GT tensors.

    Each yielded dict now contains:
      imgs         : (1, N_cam, 3, H, W)  float32  [0, 255]
      img_metas    : dict  (same as base loader)
      gt_boxes     : (M, 11) float32   [x,y,z, log_w,log_l,log_h, sin,cos, vx,vy,vz]
      gt_labels    : (M,)    int64      class indices 0-9
      scene_idx    : int                which nuScenes scene this frame belongs to
      is_first_frame : bool             True for the first keyframe of a scene

    Frames with zero valid GT boxes are included (gt_boxes has shape (0, 11)).

    If future_steps > 0, each frame also gets motion-forecasting GT:
      gt_futures      : (M, T, 2) float32  future (x,y) DISPLACEMENTS from the
                        current centre, in the CURRENT lidar frame
      gt_future_mask  : (M, T)    bool     valid future steps (object still present)
    """

    def __init__(self, dataroot: str, version: str = 'v1.0-mini',
                 future_steps: int = 0, plan: bool = False,
                 with_map: bool = False, map_radius: float = 50.0,
                 num_map: int = 50, num_map_pts: int = 20):
        super().__init__(dataroot, version)
        self.future_steps = future_steps
        self.plan = plan   # also emit ego-future trajectory + driving command
        self.with_map = with_map
        self.map_radius, self.num_map, self.num_map_pts = map_radius, num_map, num_map_pts
        self._maps: dict = {}          # location → NuScenesMap (cached)
        self._map_cache: dict = {}     # sample_token → (pts, mask) (cached per epoch)

    def _get_map(self, location: str):
        if location not in self._maps:
            from nuscenes.map_expansion.map_api import NuScenesMap
            self._maps[location] = NuScenesMap(dataroot=str(self.dataroot),
                                               map_name=location)
        return self._maps[location]

    @staticmethod
    def _resample(coords: np.ndarray, n: int) -> np.ndarray:
        """Resample a polyline (m,2) to n points evenly along arc length."""
        if coords.shape[0] < 2:
            return np.repeat(coords[:1], n, axis=0) if coords.shape[0] else np.zeros((n, 2), np.float32)
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
        t = np.linspace(0, d[-1], n)
        return np.stack([np.interp(t, d, coords[:, 0]), np.interp(t, d, coords[:, 1])], -1)

    def _load_map(self, sample_token: str):
        """HD-map polylines near the ego, in the current LIDAR frame.
        Returns map_pts (num_map, num_pts, 2) float32, map_mask (num_map,) bool."""
        tok = sample_token
        if tok in self._map_cache:
            return self._map_cache[tok]
        P, M = self.num_map_pts, self.num_map
        sample = self.nusc.get('sample', tok)
        location = self.nusc.get('log', self.nusc.get('scene', sample['scene_token'])['log_token'])['location']
        nmap = self._get_map(location)

        lid_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ep = self.nusc.get('ego_pose', lid_sd['ego_pose_token'])
        cs = self.nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])
        R_e2g = Quaternion(ep['rotation']).rotation_matrix; t_e2g = np.array(ep['translation'])
        R_l2e = Quaternion(cs['rotation']).rotation_matrix; t_l2e = np.array(cs['translation'])
        ex, ey = float(t_e2g[0]), float(t_e2g[1]); r = self.map_radius
        patch = (ex - r, ey - r, ex + r, ey + r)

        polylines = []
        recs = nmap.get_records_in_patch(patch, ['road_divider', 'lane_divider'], mode='intersect')
        for layer in ('road_divider', 'lane_divider'):
            for token in recs.get(layer, []):
                rec = nmap.get(layer, token)
                line = nmap.extract_line(rec['line_token'])
                if line.is_empty:
                    continue
                g = np.array(line.coords)[:, :2]                # (m,2) global
                lid = np.array([self._global_to_lidar(np.array([x, y, 0.0]),
                                                      R_e2g, t_e2g, R_l2e, t_l2e)[:2]
                                for x, y in g], dtype=np.float32)
                # keep only the portion within the ego radius (extract_line
                # returns the whole divider, which can run far past the patch)
                near = lid[np.linalg.norm(lid, axis=1) <= r]
                if near.shape[0] < 2:
                    continue
                polylines.append(self._resample(near, P).astype(np.float32))
                if len(polylines) >= M:
                    break
            if len(polylines) >= M:
                break

        pts = np.zeros((M, P, 2), np.float32); mask = np.zeros((M,), bool)
        for i, pl in enumerate(polylines[:M]):
            pts[i] = pl; mask[i] = True
        out = (pts, mask)
        self._map_cache[tok] = out
        return out

    def _load_ego_future(self, sample_token: str):
        """
        Ego future trajectory + driving command for planning (SparseDrive).

        Returns
        -------
        ego_future : (T, 2) float32  future ego (x,y) in the CURRENT ego frame
        ego_mask   : (T,)   bool      valid future steps
        command    : int              0=right, 1=straight, 2=left (lateral @ end)
        """
        import numpy as np
        T = self.future_steps
        sample = self.nusc.get('sample', sample_token)
        lid_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ep = self.nusc.get('ego_pose', lid_sd['ego_pose_token'])
        R_e2g = Quaternion(ep['rotation']).rotation_matrix
        t_e2g = np.array(ep['translation'], dtype=np.float64)

        ego_fut = np.zeros((T, 2), dtype=np.float32)
        ego_mask = np.zeros((T,), dtype=bool)
        nxt = sample['next']
        for t in range(T):
            if not nxt:
                break
            fs = self.nusc.get('sample', nxt)
            f_sd = self.nusc.get('sample_data', fs['data']['LIDAR_TOP'])
            f_ep = self.nusc.get('ego_pose', f_sd['ego_pose_token'])
            disp_g = np.array(f_ep['translation'], dtype=np.float64) - t_e2g
            disp_e = R_e2g.T @ disp_g                      # global → current ego
            ego_fut[t] = [disp_e[0], disp_e[1]]
            ego_mask[t] = True
            nxt = fs['next']

        # Driving command from lateral offset at the last valid step (ego y = left+)
        command = 1                                         # straight
        if ego_mask.any():
            last = np.where(ego_mask)[0][-1]
            y = ego_fut[last, 1]
            if y > 2.0:
                command = 2                                 # left
            elif y < -2.0:
                command = 0                                 # right
        return (torch.from_numpy(ego_fut), torch.from_numpy(ego_mask),
                torch.tensor(command, dtype=torch.long))

    def _global_to_lidar(self, pos_g, R_e2g, t_e2g, R_l2e, t_l2e):
        """global xyz → current lidar frame xyz."""
        import numpy as np
        pos_e = R_e2g.T @ (pos_g - t_e2g)
        return R_l2e.T @ (pos_e - t_l2e)

    def _load_gt(self, sample_token: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parse nuScenes annotations for one keyframe.

        GT boxes are in the LIDAR_TOP sensor frame to match model predictions.
        nuScenes annotations are stored in global frame; convert via:
          global → ego → lidar_sensor

        Returns
        -------
        gt_boxes  : (M, 11) float32  log-space anchor format  (LIDAR_TOP frame)
        gt_labels : (M,)    int64
        """
        sample = self.nusc.get('sample', sample_token)

        # Build global → lidar transform for this sample
        lidar_token = sample['data']['LIDAR_TOP']
        lid_sd = self.nusc.get('sample_data', lidar_token)
        lid_ep = self.nusc.get('ego_pose', lid_sd['ego_pose_token'])
        lid_cs = self.nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])

        # ego → global rotation and translation
        R_e2g = Quaternion(lid_ep['rotation']).rotation_matrix   # (3,3)
        t_e2g = np.array(lid_ep['translation'], dtype=np.float64)

        # lidar → ego rotation and translation
        R_l2e = Quaternion(lid_cs['rotation']).rotation_matrix   # (3,3)
        t_l2e = np.array(lid_cs['translation'], dtype=np.float64)

        # lidar → global rotation (for yaw offset)
        R_l2g = R_e2g @ R_l2e
        lidar_yaw_in_global = float(np.arctan2(R_l2g[1, 0], R_l2g[0, 0]))  # lidar's own heading in global 

        boxes_list:  list[list[float]] = []
        labels_list: list[int]         = []
        futures_list: list[np.ndarray] = []   # (T, 2) per object
        fmask_list:   list[np.ndarray] = []   # (T,)  per object
        T = self.future_steps

        for ann_token in sample['anns']:
            ann      = self.nusc.get('sample_annotation', ann_token)
            cat_name = ann['category_name']

            label = _CATEGORY_MAP.get(cat_name, -1)
            if label == -1:
                continue   # skip unlabelled / irrelevant categories (e.g. animal, debris)

            # --- Position: global → ego → lidar ---
            pos_g = np.array(ann['translation'], dtype=np.float64)
            pos_e = R_e2g.T @ (pos_g - t_e2g)   # global → ego (inverse of ego→global)
            pos_l = R_l2e.T @ (pos_e - t_l2e)   # ego → lidar (inverse of lidar→ego)
            x, y, z = float(pos_l[0]), float(pos_l[1]), float(pos_l[2])

            # --- Future trajectory: follow the instance's 'next' chain ---
            # Future global centres → CURRENT lidar frame → displacement from (x,y).
            if T > 0:
                fut = np.zeros((T, 2), dtype=np.float32)
                fmask = np.zeros((T,), dtype=bool)
                nxt = ann['next']
                for t in range(T):
                    if not nxt:
                        break
                    fa = self.nusc.get('sample_annotation', nxt)
                    fpos_l = self._global_to_lidar(
                        np.array(fa['translation'], dtype=np.float64),
                        R_e2g, t_e2g, R_l2e, t_l2e)
                    fut[t]   = [fpos_l[0] - x, fpos_l[1] - y]   # displacement from now
                    fmask[t] = True
                    nxt = fa['next']
                futures_list.append(fut)
                fmask_list.append(fmask)

            # --- Size: convert to log-space (anchor format) + w/l swap ---
            # nuScenes ann['size'] = [width, length, height]; model anchor
            # slots 3/4/5 are [length, width, height] (slot 3 = extent along
            # heading, matching the reference dims[..., [1,0,2]] swap).
            w, l_dim, h = ann['size']  # nuScenes order = [width, length, height]
            w     = max(w, 0.01)  # Clamps guard against log(0) for degenerate annotations
            l_dim = max(l_dim, 0.01)
            h     = max(h, 0.01)
            log_w = np.log(l_dim)   # slot 3 ← length
            log_l = np.log(w)       # slot 4 ← width
            log_h = np.log(h)

            # --- Yaw: global frame → lidar frame ---
            yaw_global = Quaternion(ann['rotation']).yaw_pitch_roll[0]
            yaw_lidar  = yaw_global - lidar_yaw_in_global  # yaw is the global heading minus the lidar's own heading in global 
            sin_yaw = float(np.sin(yaw_lidar))
            cos_yaw = float(np.cos(yaw_lidar))

            # --- Velocity: global → lidar frame (rotate by R_l2g.T) ---
            velo_g = self.nusc.box_velocity(ann_token)   # (3,) global m/s
            velo_g = np.nan_to_num(velo_g, nan=0.0)  # nuScenes velocities can be NaN (objects seen in only one frame)
            velo_l = R_l2g.T @ velo_g                   # Velocity is rotated by R_l2g.T (global→lidar) but not translated — it's a direction vector
            vx, vy, vz = float(velo_l[0]), float(velo_l[1]), float(velo_l[2])

            boxes_list.append([x, y, z, log_w, log_l, log_h,
                                sin_yaw, cos_yaw, vx, vy, vz])
            labels_list.append(label)

        if boxes_list:
            gt_boxes  = torch.tensor(boxes_list,  dtype=torch.float32)  # (M, 11)
            gt_labels = torch.tensor(labels_list, dtype=torch.long)     # (M,)
        else:  # Frames with zero GT yield (0,11) — kept, not dropped, because they still supply negative examples for classification.
            gt_boxes  = torch.zeros(0, 11, dtype=torch.float32)
            gt_labels = torch.zeros(0,     dtype=torch.long)

        if T > 0:
            if futures_list:
                gt_futures     = torch.from_numpy(np.stack(futures_list))      # (M, T, 2)
                gt_future_mask = torch.from_numpy(np.stack(fmask_list))        # (M, T)
            else:
                gt_futures     = torch.zeros(0, T, 2, dtype=torch.float32)
                gt_future_mask = torch.zeros(0, T, dtype=torch.bool)
            return gt_boxes, gt_labels, gt_futures, gt_future_mask

        return gt_boxes, gt_labels

    # ------------------------------------------------------------------
    # Override _process_sample to inject GT
    # ------------------------------------------------------------------

    def _process_sample_with_gt(
        self, sample_token: str, scene_idx: int, is_first: bool
    ) -> dict:
        frame = self._process_sample(sample_token)
        if self.future_steps > 0:
            gt_boxes, gt_labels, gt_futures, gt_future_mask = self._load_gt(sample_token)
            frame['gt_futures']     = gt_futures      # (M, T, 2)
            frame['gt_future_mask'] = gt_future_mask  # (M, T)
            if self.plan:
                ego_fut, ego_mask, command = self._load_ego_future(sample_token)
                frame['ego_future']      = ego_fut    # (T, 2) current ego frame
                frame['ego_future_mask'] = ego_mask   # (T,)
                frame['command']         = command    # 0/1/2
            if self.with_map:
                map_pts, map_mask = self._load_map(sample_token)
                frame['img_metas']['map_pts']  = map_pts    # (num_map, num_pts, 2) lidar frame
                frame['img_metas']['map_mask'] = map_mask   # (num_map,)
        else:
            gt_boxes, gt_labels = self._load_gt(sample_token)
        frame['gt_boxes']       = gt_boxes     # (M, 11)
        frame['gt_labels']      = gt_labels    # (M,)
        frame['scene_idx']      = scene_idx
        frame['is_first_frame'] = is_first
        return frame

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_scene(self, scene_idx: int = 0) -> Iterator[dict]:   # type: ignore[override]
        scene = self.nusc.scene[scene_idx]
        token = scene['first_sample_token']
        first = True
        while token:
            sample = self.nusc.get('sample', token)
            yield self._process_sample_with_gt(token, scene_idx, first)
            first = False
            token = sample['next'] if sample['next'] else None

    def __iter__(self) -> Iterator[dict]:   # type: ignore[override]
        for idx in range(len(self.nusc.scene)):
            yield from self.iter_scene(scene_idx=idx)
