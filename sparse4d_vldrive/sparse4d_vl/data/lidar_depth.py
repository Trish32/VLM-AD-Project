"""
LiDAR → sparse depth ground-truth for dense depth supervision (Sparse4D-v2).

Pure NumPy / PyTorch — no mmdet3d, no point-cloud library.  A nuScenes
LIDAR_TOP sweep is just a binary blob of float32 (x, y, z, intensity, ring),
read directly with np.fromfile.

The depth GT is produced by projecting the LiDAR points into each camera using
the SAME 4x4 `projection_mat` the image loader already builds (lidar -> pixel):

    [u*d, v*d, d]^T = projection_mat[:3, :] @ [x, y, z, 1]^T

because the intrinsic K's third row is [0, 0, 1, 0], the third projected
component IS the camera-frame depth d.  So pixel = (u, v) = proj_xy / d and
depth = d, all from one matmul — no separate intrinsic handling needed.

Points are scattered into a (H_feat, W_feat) grid at the depth-head's output
stride, keeping the NEAREST (minimum-depth) point per cell so foreground wins
over background along a ray.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_lidar_points(nusc, sample_token: str) -> np.ndarray:
    """
    Load the LIDAR_TOP point cloud for a keyframe, in the lidar sensor frame.

    Returns
    -------
    pts : (N, 3) float32   [x, y, z] in the LIDAR_TOP frame
    """
    sample      = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lid_sd      = nusc.get('sample_data', lidar_token)
    path        = Path(nusc.dataroot) / lid_sd['filename']

    # nuScenes LiDAR: float32, 5 columns (x, y, z, intensity, ring_index)
    scan = np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)
    return scan[:, :3].astype(np.float32)


def build_depth_targets(
    points:         np.ndarray,    # (N, 3) lidar-frame
    projection_mat: np.ndarray,    # (N_cam, 4, 4)  lidar -> pixel
    img_wh:         np.ndarray,    # (N_cam, 2)  [W, H] of the full input image
    feat_hw:        tuple[int, int],   # (H_feat, W_feat) of the depth-head output
    min_depth:      float = 1e-3,
    max_depth:      float = 60.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project LiDAR points into every camera and scatter into a sparse depth grid.

    Returns
    -------
    depth : (N_cam, H_feat, W_feat) float32   metric depth, 0 where no point
    mask  : (N_cam, H_feat, W_feat) bool      True where a point was scattered
    """
    N_cam = projection_mat.shape[0]
    H_f, W_f = feat_hw

    depth = np.zeros((N_cam, H_f, W_f), dtype=np.float32)
    valid = np.zeros((N_cam, H_f, W_f), dtype=bool)

    # Homogeneous lidar points: (N, 4)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), np.float32)], axis=1)

    for c in range(N_cam):
        P = projection_mat[c][:3, :]                 # (3, 4)
        proj = pts_h @ P.T                           # (N, 3) = [u*d, v*d, d]
        d = proj[:, 2]

        in_front = d > min_depth
        u = proj[in_front, 0] / d[in_front]          # full-image pixel coords
        v = proj[in_front, 1] / d[in_front]
        dd = d[in_front]

        W_img, H_img = float(img_wh[c, 0]), float(img_wh[c, 1])
        sx = W_f / W_img                             # full-image -> feature stride
        sy = H_f / H_img
        fx = np.floor(u * sx).astype(np.int64)
        fy = np.floor(v * sy).astype(np.int64)

        keep = (
            (fx >= 0) & (fx < W_f) & (fy >= 0) & (fy < H_f) &
            (dd < max_depth)
        )
        fx, fy, dd = fx[keep], fy[keep], dd[keep]

        # Keep the closest point per cell (foreground wins). Sort far->near so
        # nearer writes land last and overwrite.
        order = np.argsort(-dd)
        fx, fy, dd = fx[order], fy[order], dd[order]
        depth[c, fy, fx] = dd
        valid[c, fy, fx] = True

    return torch.from_numpy(depth), torch.from_numpy(valid)


def depth_targets_for_sample(
    nusc,
    sample_token:   str,
    projection_mat: np.ndarray,    # (N_cam, 4, 4)
    img_wh:         np.ndarray,    # (N_cam, 2)
    feat_hw:        tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience: load points and build targets in one call."""
    pts = load_lidar_points(nusc, sample_token)
    return build_depth_targets(pts, projection_mat, img_wh, feat_hw)
