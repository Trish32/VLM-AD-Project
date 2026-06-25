"""PointPillarsScatter — scatter (M, C) voxel features onto a dense BEV canvas."""
from __future__ import annotations

import torch
import torch.nn as nn


class PointPillarsScatter(nn.Module):
    def __init__(self, in_channels=64, output_shape=(400, 400)):
        super().__init__()
        self.in_channels = in_channels
        self.ny, self.nx = output_shape  # (H=y, W=x)

    def forward(self, voxel_features, coors, batch_size):
        """
        voxel_features: (M, C)
        coors: (M, 4) [batch, z, y, x]
        returns: (B, C, ny, nx)
        """
        batch_canvas = []
        for b in range(batch_size):
            canvas = voxel_features.new_zeros((self.in_channels, self.nx * self.ny))
            mask = coors[:, 0] == b
            this_coors = coors[mask]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]  # y * nx + x
            indices = indices.long()
            voxels = voxel_features[mask].t()
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)
        batch_canvas = torch.stack(batch_canvas, 0)
        return batch_canvas.view(batch_size, self.in_channels, self.ny, self.nx)
