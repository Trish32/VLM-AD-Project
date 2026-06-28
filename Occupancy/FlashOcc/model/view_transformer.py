"""LSS (Lift-Splat-Shoot) view transformer.

Pure-PyTorch / MPS reimplementation of
``projects/mmdet3d_plugin/models/necks/view_transformer.py`` (the plain
``LSSViewTransformer`` used by flashocc-r50, i.e. no depth supervision).

The only CUDA dependency in the original is ``bev_pool_v2``.  Semantically that
op computes, for every (camera, depth-bin, pixel) frustum point that falls
inside the BEV grid::

    bev[voxel] += depth[point] * context_feat[pixel]

i.e. a depth-weighted scatter-add of the context feature into BEV voxels.  We
reproduce it exactly with ``Tensor.index_add_`` which is supported on MPS — no
cumsum trick, no custom kernel.  The frustum geometry is computed on CPU in
float (matrix inverses on MPS are flaky) and only the scatter runs on-device.
"""
import torch
import torch.nn as nn


class LSSViewTransformer(nn.Module):
    def __init__(self, grid_config, input_size, in_channels=256,
                 out_channels=64, downsample=16, collapse_z=True):
        super().__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.collapse_z = collapse_z
        self.out_channels = out_channels

        self._create_grid_infos(**grid_config)
        self.frustum = self._create_frustum(grid_config['depth'], input_size,
                                            downsample)        # (D,fH,fW,3)
        self.D = self.frustum.shape[0]
        # depth_net: 1x1 conv producing D depth logits + out_channels context
        self.depth_net = nn.Conv2d(in_channels, self.D + out_channels, 1)

    def _create_grid_infos(self, x, y, z, **kwargs):
        self.grid_lower_bound = torch.tensor([c[0] for c in (x, y, z)])
        self.grid_interval = torch.tensor([c[2] for c in (x, y, z)])
        self.grid_size = torch.tensor([(c[1] - c[0]) / c[2] for c in (x, y, z)])

    def _create_frustum(self, depth_cfg, input_size, downsample):
        """Build the per-camera frustum template of candidate 3D points.

        For every feature-map pixel (fH x fW) we enumerate D discrete depths
        along the camera ray. Each frustum entry is (u, v, d): the pixel's
        coordinate in the *original input image* (note x/y span 0..W_in-1, not
        the feature size) plus a metric depth d in metres. This template is
        camera-agnostic; the actual camera geometry is applied in get_ego_coor.

        For flashocc-r50: depth_cfg=[1,45,0.5] -> D=88; input 256x704,
        downsample 16 -> fH=16, fW=44. Result (88, 16, 44, 3).
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        # D depth planes [1.0, 1.5, ..., 44.5] broadcast over the feature grid.
        d = torch.arange(*depth_cfg, dtype=torch.float).view(-1, 1, 1)
        D = d.shape[0]
        d = d.expand(D, H_feat, W_feat)
        # Pixel u,v sampled at feature-map locations but in input-image pixels.
        x = torch.linspace(0, W_in - 1, W_feat).view(1, 1, W_feat).expand(
            D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat).view(1, H_feat, 1).expand(
            D, H_feat, W_feat)
        return torch.stack((x, y, d), -1)      # (D, fH, fW, 3) :: (u, v, d)

    def get_ego_coor(self, sensor2ego, cam2imgs, post_rots, post_trans, bda):
        """Map every frustum point into the key-ego (BEV) frame.

        This is the geometric core of lift-splat. Starting from the (u,v,d)
        frustum template we reverse the imaging pipeline step by step:

          1. Undo image-view augmentation (resize/crop/flip/rotate), which was
             stored as an affine (post_rot, post_tran), to recover the pixel
             coordinate in the raw camera image.
          2. Back-project pixel -> camera ray. A pinhole maps a 3D camera point
             (X,Y,Z) to pixel (u,v) via u=fx*X/Z+cx etc., i.e. homogeneous
             [u*d, v*d, d] = K @ [X, Y, Z] with Z=d. So we form [u*d, v*d, d]
             and multiply by K^-1 to get the metric camera-space point.
          3. Camera -> key-ego via the extrinsic rotation+translation
             (sensor2ego already folded into the key-ego frame upstream).
          4. Apply BEV data augmentation `bda` (identity at test time).

        Returns (B, N, D, fH, fW, 3): each frustum point's (x,y,z) in metres in
        the key-ego frame. Done in float64 on CPU (MPS lacks float64 inverse).
        """
        B, N, _, _ = sensor2ego.shape
        frustum = self.frustum.to(sensor2ego)
        # (1) undo image-view augmentation: x_img = post_rot^-1 @ (frustum - post_tran)
        points = frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))
        # (2) pixel (u,v,d) -> homogeneous (u*d, v*d, d) so that K^-1 @ . gives
        #     the metric camera-space ray point.
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        # (3) camera -> ego: combine = R_{cam->ego} @ K^-1, then add translation
        combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points = points + sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)
        # (4) BEV-space augmentation (rotation/scale/flip); identity at test time
        points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(
            points.unsqueeze(-1)).squeeze(-1)
        return points

    def _pooling_ranks(self, coor):
        """Compute flat indices linking each frustum point to its BEV voxel.

        Replicates the official voxel_pooling_prepare_v2 but drops the cumsum
        interval bookkeeping (that only speeds up the custom CUDA kernel; our
        index_add does not need it).

        For each of the B*N*D*fH*fW frustum points we produce three flat
        indices, then keep only points that land inside the BEV grid:
          - ranks_depth: index into the flattened depth volume (B*N*D*fH*fW).
          - ranks_feat:  index into the flattened context features
            (B*N*fH*fW) -- all D depths of a pixel share ONE context vector.
          - ranks_bev:   index into the flattened BEV grid (B*Dz*Dy*Dx) the
            point falls in. Points sharing a voxel get the same rank and will
            be summed together by the scatter.
        """
        B, N, D, H, W, _ = coor.shape
        num = B * N * D * H * W
        device = coor.device
        nx, ny, nz = [int(self.grid_size[i]) for i in range(3)]

        # Identity index over the depth volume (one per frustum point).
        ranks_depth = torch.arange(num, dtype=torch.long, device=device)
        # Context index: built per pixel (no D axis) then repeated over depth,
        # so depth bin k of pixel p maps to the same context feature p.
        ranks_feat = torch.arange(num // D, dtype=torch.long, device=device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W).expand(
            B, N, D, H, W).reshape(-1)

        # Metres -> integer voxel coordinate: (point - grid_origin) / cell_size.
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor)).long().view(num, 3)
        batch_idx = torch.arange(B, device=device).reshape(B, 1).expand(
            B, num // B).reshape(num)

        # Drop points outside the [0, grid_size) box on every axis.
        kept = (
            (coor[:, 0] >= 0) & (coor[:, 0] < nx) &
            (coor[:, 1] >= 0) & (coor[:, 1] < ny) &
            (coor[:, 2] >= 0) & (coor[:, 2] < nz))
        coor, ranks_depth, ranks_feat, batch_idx = \
            coor[kept], ranks_depth[kept], ranks_feat[kept], batch_idx[kept]

        # Row-major flatten of (batch, z, y, x) -> single BEV rank.
        ranks_bev = (batch_idx * (nz * ny * nx) +
                     coor[:, 2] * (ny * nx) +
                     coor[:, 1] * nx + coor[:, 0])
        return ranks_bev, ranks_depth, ranks_feat

    def voxel_pooling(self, coor, depth, feat):
        """The "splat": depth-weighted scatter-add of context into BEV voxels.

        This is our pure-PyTorch replacement for the CUDA ``bev_pool_v2``. The
        op it implements is, for every in-grid frustum point:

            bev[voxel] += depth_prob[point] * context_feat[pixel] via index_add_

        Each pixel contributes its context vector to every depth bin, weighted
        by that bin's predicted probability -- "soft" depth. Points landing in
        the same voxel accumulate (index_add_ sums duplicate indices), which is
        exactly the pooling. Verified to match the official kernel's test case.

        Args:
            coor:  (B, N, D, fH, fW, 3) frustum coords in BEV space (on CPU)
            depth: (B, N, D, fH, fW)    softmaxed depth distribution
            feat:  (B, N, C, fH, fW)    context features
        Returns:
            (B, C*Dz, Dy, Dx)  -- Dz collapsed to 1 for flashocc-r50
        """
        ranks_bev, ranks_depth, ranks_feat = self._pooling_ranks(coor)
        # ranks are computed on CPU (geometry); move them to the feature device.
        ranks_bev = ranks_bev.to(feat.device)
        ranks_depth = ranks_depth.to(feat.device)
        ranks_feat = ranks_feat.to(feat.device)

        B, N, C, H, W = feat.shape
        nx, ny, nz = [int(self.grid_size[i]) for i in range(3)]

        # Flatten so the rank tensors can gather rows directly.
        feat = feat.permute(0, 1, 3, 4, 2).reshape(-1, C)   # (B*N*fH*fW, C)
        depth = depth.reshape(-1)                           # (B*N*D*fH*fW,)

        # Per-point value = scalar depth prob * its pixel's context vector.
        vals = depth[ranks_depth].unsqueeze(1) * feat[ranks_feat]   # (P, C)
        # Scatter-add into the flat BEV grid; collisions (same voxel) sum.
        out = torch.zeros(B * nz * ny * nx, C, dtype=vals.dtype,
                          device=vals.device)
        out.index_add_(0, ranks_bev, vals)
        out = out.view(B, nz, ny, nx, C).permute(0, 4, 1, 2, 3)  # (B,C,Dz,Dy,Dx)
        if self.collapse_z:
            # Dz==1 here, so this just drops the singleton height axis into C.
            out = torch.cat(out.unbind(dim=2), 1)    # (B, C*Dz, Dy, Dx)
        return out

    def forward(self, x, sensor2ego, cam2imgs, post_rots, post_trans, bda):
        """Lift (predict depth+context) then splat (scatter into BEV).

        Args:
            x: (B, N, C_in, fH, fW) image features from the neck
            sensor2ego, cam2imgs, post_rots, post_trans: (B, N, 3/4, ...)
            bda: (B, 3, 3)
        Returns:
            bev_feat: (B, C*Dz, Dy, Dx),  depth: (B*N, D, fH, fW)
        """
        B, N, C, H, W = x.shape
        feat = x.view(B * N, C, H, W)
        # LIFT: one 1x1 conv(depth_net) predicts, per pixel, D depth logits AND a C-dim
        # context vector, concatenated along channels.
        feat = self.depth_net(feat)
        depth_digit = feat[:, :self.D]                      # depth logits
        tran_feat = feat[:, self.D:self.D + self.out_channels]   # context
        # Softmax over depth -> a soft probability distribution along each ray.
        depth = depth_digit.softmax(dim=1)

        # Geometry (frustum -> BEV coords) on CPU in float64 for the inverses;
        # the scatter itself runs on x's device (MPS/CPU).
        coor = self.get_ego_coor(
            sensor2ego.cpu().double(), cam2imgs.cpu().double(),
            post_rots.cpu().double(), post_trans.cpu().double(),
            bda.cpu().double()).float()
        # SPLAT: scatter depth-weighted context into the BEV grid.
        bev_feat = self.voxel_pooling(
            coor, depth.view(B, N, self.D, H, W),
            tran_feat.view(B, N, self.out_channels, H, W))
        # depth is returned for optional depth supervision (unused in this cfg).
        return bev_feat, depth.view(B, N, self.D, H, W).flatten(0, 1)
