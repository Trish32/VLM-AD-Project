"""
Camera->BEV view transforms for MIT BEVFusion (pure PyTorch, MPS).
- LSSTransform (seg): depthnet predicts depth dist + context; lift-splat.
- DepthLSSTransform (det): also uses a sparse lidar depth image (scalar) via
  dtransform, concatenated before depthnet.
bev_pool (CUDA) replaced by index_add sum-pooling (nz=1). Geometry uses
camera2lidar / intrinsics / img_aug / lidar_aug matrices supplied by the loader.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .bev_pool import BEVPool


def gen_dx_bx(xbound, ybound, zbound):
    """Turn [min, max, step] bounds for x/y/z into the BEV grid descriptors.
    dx (3,) = voxel size per axis (the step);
    bx (3,) = metric center of the FIRST cell (min + step/2);
    nx (3,) = number of cells per axis. Together these map a metric point -> grid index
    via idx = round((p - (bx - dx/2)) / dx). Used by get_geometry / bev_pool."""
    dx = torch.tensor([r[2] for r in [xbound, ybound, zbound]])                   # (3,) cell size
    bx = torch.tensor([r[0] + r[2] / 2.0 for r in [xbound, ybound, zbound]])      # (3,) first-cell center
    nx = torch.tensor([round((r[1] - r[0]) / r[2]) for r in [xbound, ybound, zbound]],
                      dtype=torch.long)                                           # (3,) cell counts
    return dx, bx, nx


class BaseTransform(nn.Module):
    """Shared lift-splat machinery. Subclasses differ only in how they predict the
    per-pixel depth distribution + context (get_cam_feats). This base owns the fixed
    frustum, the calibration -> 3D projection (get_geometry), and the splat (bev_pool)."""

    def __init__(self, in_channels, out_channels, image_size, feature_size,
                 xbound, ybound, zbound, dbound):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size          # (iH, iW) e.g. (256, 704)
        self.feature_size = feature_size      # (fH, fW) e.g. (32, 88) = image/8
        self.xbound, self.ybound, self.zbound, self.dbound = xbound, ybound, zbound, dbound
        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)   # each (3,)
        self.dx = nn.Parameter(dx, requires_grad=False)  # (3,) BEV cell size
        self.bx = nn.Parameter(bx, requires_grad=False)  # (3,) first-cell center
        self.nx = nn.Parameter(nx, requires_grad=False)  # (3,) BEV cell counts
        self.C = out_channels                            # context channels per point
        self.frustum = nn.Parameter(self.create_frustum(), requires_grad=False)  # (D,fH,fW,3)
        self.D = self.frustum.shape[0]                   # number of depth bins (e.g. 118)
        # Precomputed BEV pooling (caches geometry-derived indices, interval reduction)
        self.pool = BEVPool(dx.clone(), bx.clone(), nx.clone(), precompute=True)

    def create_frustum(self):
        """Build the fixed (D, fH, fW, 3) grid of candidate image-points: for every
        feature pixel (u,v), one entry per depth bin d. Coords are (u, v, d) in PIXEL
        space (u,v on the ORIGINAL image scale, d in metres). This is the 'what-if'
        cloud -- every place each pixel could be -- before depth is predicted."""
        iH, iW = self.image_size
        fH, fW = self.feature_size
        # depth bins broadcast across the feature grid -> (D, fH, fW)
        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D = ds.shape[0]
        # pixel u/v coords spread over the full image extent (feature grid is image/8)
        xs = torch.linspace(0, iW - 1, fW).view(1, 1, fW).expand(D, fH, fW)   # (D,fH,fW) u
        ys = torch.linspace(0, iH - 1, fH).view(1, fH, 1).expand(D, fH, fW)   # (D,fH,fW) v
        return torch.stack((xs, ys, ds), -1)             # (D, fH, fW, 3) = (u, v, d)

    def get_geometry(self, camera2lidar_rots, camera2lidar_trans, intrins,
                     post_rots, post_trans, extra_rots, extra_trans):
        """PROJECT: map every frustum point (u,v,d) into the shared lidar/ego 3D frame.
        Exactly inverts the loader's preprocessing chain, in order:
          1. undo image augmentation (post_rots/trans)
          2. pixel->camera ray: (u*d, v*d, d) then K^-1  (the perspective un-projection)
          3. camera->lidar (extrinsics)
          4. apply lidar augmentation (identity at test)
        Returns (B, N, D, fH, fW, 3) metric xyz."""
        B, N, _ = camera2lidar_trans.shape
        # (1) undo image-aug translation then rotation  -> back to raw pixel coords
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)                       # (B,N,D,fH,fW,3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))  # (...,3,1)
        # (2) perspective un-projection: scale (u,v) by depth so K^-1 recovers the ray
        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)    # (u*d,v*d,d)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))                      # (B,N,3,3) = R @ K^-1
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)           # (B,N,D,fH,fW,3) camera->lidar rot
        # (3) add camera position in lidar frame
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)
        # (4) lidar augmentation (rotation then translation; identity at eval)
        points = extra_rots.view(B, 1, 1, 1, 1, 3, 3).repeat(1, N, 1, 1, 1, 1, 1).matmul(
            points.unsqueeze(-1)).squeeze(-1)
        points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)
        return points                                    # (B, N, D, fH, fW, 3) metric xyz in lidar frame

    def bev_pool(self, geom_feats, x):
        # SPLAT: drop each 3D point's feature into its BEV cell and sum (see bev_pool.py).
        # Precomputed BEV pooling (interval reduction + cached geometry indices).
        return self.pool(geom_feats, x)

    def geom_from_mats(self, camera2lidar, camera_intrinsics, img_aug_matrix,
                       lidar_aug_matrix):
        """Slice the 4x4 transform matrices from the loader into the rot/trans blocks
        get_geometry expects, then call it. (...,4,4) -> geometry (B,N,D,fH,fW,3)."""
        c2l_r = camera2lidar[..., :3, :3]          # (B,N,3,3) camera->lidar rotation
        c2l_t = camera2lidar[..., :3, 3]           # (B,N,3)   camera->lidar translation
        intrins = camera_intrinsics[..., :3, :3]   # (B,N,3,3) K
        post_rots = img_aug_matrix[..., :3, :3]    # (B,N,3,3) image-aug rotation/scale
        post_trans = img_aug_matrix[..., :3, 3]    # (B,N,3)   image-aug translation (crop)
        extra_rots = lidar_aug_matrix[..., :3, :3] # (B,3,3)   lidar-aug rotation (eye at test)
        extra_trans = lidar_aug_matrix[..., :3, 3] # (B,3)     lidar-aug translation (0 at test)
        return self.get_geometry(c2l_r, c2l_t, intrins, post_rots, post_trans,
                                 extra_rots, extra_trans)


def _downsample_block(c):
    # 3 convs (middle one stride-2) that halve the BEV resolution after splatting.
    return nn.Sequential(
        nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
        nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
        nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
    )


class LSSTransform(BaseTransform):
    """Segmentation view-transform: depth distribution predicted from IMAGE features
    alone (no lidar depth prior). depthnet is a single 1x1 conv -> D + C channels."""

    def __init__(self, in_channels, out_channels, image_size, feature_size,
                 xbound, ybound, zbound, dbound, downsample=2):
        super().__init__(in_channels, out_channels, image_size, feature_size,
                         xbound, ybound, zbound, dbound)
        # one conv emits D depth logits AND C context channels per pixel
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        self.downsample = _downsample_block(out_channels) if downsample > 1 else nn.Identity()

    def get_cam_feats(self, x):
        """LIFT: per pixel, softmax over D depths, outer-product with C context.
        x: (B, N, C_in, fH, fW)  ->  (B, N, D, fH, fW, C) features placed along depth."""
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)
        x = self.depthnet(x)                              # (B*N, D+C, fH, fW)
        depth = x[:, :self.D].softmax(dim=1)             # (B*N, D, fH, fW) prob over depth bins
        # outer product: depth (B*N,1,D,fH,fW) * context (B*N,C,1,fH,fW) -> (B*N,C,D,fH,fW)
        x = depth.unsqueeze(1) * x[:, self.D:self.D + self.C].unsqueeze(2)
        x = x.view(B, N, self.C, self.D, fH, fW).permute(0, 1, 3, 4, 5, 2)   # (B,N,D,fH,fW,C)
        return x

    def forward(self, img_feats, points, camera2lidar, camera_intrinsics,
                img_aug_matrix, lidar_aug_matrix, lidar2image):
        geom = self.geom_from_mats(camera2lidar, camera_intrinsics,
                                   img_aug_matrix, lidar_aug_matrix)   # (B,N,D,fH,fW,3) where each point lands
        x = self.get_cam_feats(img_feats)                              # (B,N,D,fH,fW,C) what to place there
        x = self.bev_pool(geom, x)                                    # (B,C,Hbev,Wbev) splat+sum
        return self.downsample(x)                                     # (B,C,Hbev/2,Wbev/2)


class DepthLSSTransform(BaseTransform):
    """Detection view-transform: same lift-splat, but the depth prediction is CONDITIONED
    on a sparse lidar depth image. Real lidar returns are projected into each camera to
    give a per-pixel depth hint, encoded by dtransform and concatenated before depthnet.
    This makes monocular depth far less ambiguous -> sharper camera BEV for detection."""

    def __init__(self, in_channels, out_channels, image_size, feature_size,
                 xbound, ybound, zbound, dbound, downsample=2):
        super().__init__(in_channels, out_channels, image_size, feature_size,
                         xbound, ybound, zbound, dbound)
        # encodes the 1-channel sparse lidar-depth image -> 64ch (also downsamples /8 to match feats)
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1), nn.BatchNorm2d(8), nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        # depthnet now takes image feats + depth encoding -> D depth logits + C context
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        self.downsample = _downsample_block(out_channels) if downsample > 1 else nn.Identity()

    def get_cam_feats(self, x, d):
        """LIFT (depth-conditioned). x: image feats (B,N,C_in,fH,fW); d: lidar depth image
        (B,N,1,iH,iW). -> (B,N,D,fH,fW,C)."""
        B, N, C, fH, fW = x.shape
        d = d.view(B * N, *d.shape[2:])                  # (B*N, 1, iH, iW)
        x = x.view(B * N, C, fH, fW)                     # (B*N, C_in, fH, fW)
        d = self.dtransform(d)                            # (B*N, 64, fH, fW) encoded depth prior
        x = torch.cat([d, x], dim=1)                     # (B*N, C_in+64, fH, fW)
        x = self.depthnet(x)                              # (B*N, D+C, fH, fW)
        depth = x[:, :self.D].softmax(dim=1)             # (B*N, D, fH, fW)
        # depth (B*N,1,D,fH,fW) * context (B*N,C,1,fH,fW) -> (B*N,C,D,fH,fW)
        x = depth.unsqueeze(1) * x[:, self.D:self.D + self.C].unsqueeze(2)
        x = x.view(B, N, self.C, self.D, fH, fW).permute(0, 1, 3, 4, 5, 2)   # (B,N,D,fH,fW,C)
        return x

    def _depth_image(self, points, lidar2image, img_aug_matrix, lidar_aug_matrix):
        """Build the sparse lidar-depth image: project real lidar points into each camera (lidar→image)
        and write their distance at the landed pixel. -> (B, N, 1, iH, iW), mostly zeros.
        This is the FORWARD projection (lidar->image), the inverse of get_geometry.
         It scatters real lidar distances onto the image plane to give detection a depth prior. """
        B = len(points)
        N = lidar2image.shape[1]
        iH, iW = self.image_size
        depth = torch.zeros(B, N, 1, iH, iW, device=points[0].device)
        for b in range(B):
            cur = points[b][:, :3]                       # (M, 3) xyz lidar points
            la = lidar_aug_matrix[b]
            # undo lidar augmentation so points are in the raw lidar frame the calib expects
            cur = cur - la[:3, 3]
            cur = torch.inverse(la[:3, :3]).matmul(cur.transpose(1, 0))   # (3, M)
            l2i = lidar2image[b]                          # (N, 4, 4) lidar->each image
            cur = l2i[:, :3, :3].matmul(cur)             # (N, 3, M) rotate
            cur += l2i[:, :3, 3].reshape(-1, 3, 1)       # translate -> (N,3,M) camera-frame homog
            dist = cur[:, 2, :]                           # (N, M) depth = z before normalize
            cur[:, 2, :] = torch.clamp(cur[:, 2, :], 1e-5, 1e5)
            cur[:, :2, :] /= cur[:, 2:3, :]              # perspective divide -> pixel (u,v)
            ia = img_aug_matrix[b]                        # apply the SAME image aug as the feats
            cur = ia[:, :3, :3].matmul(cur) + ia[:, :3, 3].reshape(-1, 3, 1)
            cur = cur[:, :2, :].transpose(1, 2)[..., [1, 0]]   # (N, M, 2) as (row, col) = (v, u)
            on = ((cur[..., 0] >= 0) & (cur[..., 0] < iH) &     # (N, M) inside the image?
                  (cur[..., 1] >= 0) & (cur[..., 1] < iW))
            for c in range(N):
                mc = cur[c, on[c]].long()                # (m_c, 2) valid pixel coords for cam c
                depth[b, c, 0, mc[:, 0], mc[:, 1]] = dist[c, on[c]]   # scatter distance
        return depth                                     # (B, N, 1, iH, iW)

    def forward(self, img_feats, points, camera2lidar, camera_intrinsics,
                img_aug_matrix, lidar_aug_matrix, lidar2image):
        d = self._depth_image(points, lidar2image, img_aug_matrix, lidar_aug_matrix)  # (B,N,1,iH,iW)
        geom = self.geom_from_mats(camera2lidar, camera_intrinsics,
                                   img_aug_matrix, lidar_aug_matrix)   # (B,N,D,fH,fW,3)
        x = self.get_cam_feats(img_feats, d)                          # (B,N,D,fH,fW,C)
        x = self.bev_pool(geom, x)                                    # (B,C,Hbev,Wbev)
        return self.downsample(x)                                     # (B,C,Hbev/2,Wbev/2)
