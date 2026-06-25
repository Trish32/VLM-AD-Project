"""
HardVFE — pure-PyTorch port of mmdet3d HardVFE (PointPillars-style voxel
feature encoder used by bevf_pp).

Config: in_channels=4, feat_channels=[64, 64], with_cluster_center=True,
with_voxel_center=True, with_distance=False  =>  decorated in_channels=10.
Two VFELayers: layer0 in=10 out=64 (cat_max -> 128), layer1 in=128 out=64
(max_out, no cat) -> final per-voxel feature (M, 64).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_paddings_indicator(actual_num, max_num):
    """Boolean mask (M, max_num): True for valid points within each voxel."""
    actual_num = actual_num.unsqueeze(1)
    max_num_arange = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(1, -1)
    return actual_num.int() > max_num_arange


class VFELayer(nn.Module):
    """linear -> BN1d -> ReLU -> (max over points) with optional concat."""

    def __init__(self, in_channels, out_channels, max_out=True, cat_max=True):
        super().__init__()
        self.cat_max = cat_max
        self.max_out = max_out
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs):
        # inputs: (M, T, C)
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        # BN over channel dim: (M, T, C) -> (M, C, T) -> norm -> back
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        pointwise = F.relu(x)
        if self.max_out:
            aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        else:
            return pointwise
        if not self.cat_max:
            return aggregated.squeeze(1)
        repeated = aggregated.repeat(1, voxel_count, 1)
        return torch.cat([pointwise, repeated], dim=2)


class HardVFE(nn.Module):
    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, 64),
                 voxel_size=(0.25, 0.25, 8.0),
                 point_cloud_range=(-50, -50, -5, 50, 50, 3),
                 with_cluster_center=True,
                 with_voxel_center=True,
                 with_distance=False):
        super().__init__()
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self._with_distance = with_distance
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 3
        self.in_channels = in_channels

        self.vx, self.vy, self.vz = voxel_size
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

        feat_channels = [self.in_channels] + list(feat_channels)
        layers = []
        for i in range(len(feat_channels) - 1):
            in_f = feat_channels[i]
            out_f = feat_channels[i + 1]
            if i > 0:
                in_f *= 2
            last = (i == len(feat_channels) - 2)
            layers.append(VFELayer(in_f, out_f,
                                   max_out=True,
                                   cat_max=not last))
        self.vfe_layers = nn.ModuleList(layers)

    def forward(self, features, num_points, coors):
        """
        features: (M, T, C_raw)   raw point features per voxel
        num_points: (M,)          valid points per voxel
        coors: (M, 4)             [batch, z, y, x]
        returns: (M, 64)
        """
        features_ls = [features]
        if self._with_cluster_center:
            points_mean = (features[:, :, :3].sum(dim=1, keepdim=True) /
                           num_points.type_as(features).view(-1, 1, 1))
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        if self._with_voxel_center:
            f_center = features.new_zeros((features.size(0), features.size(1), 3))
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx + self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy + self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (
                coors[:, 1].type_as(features).unsqueeze(1) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            features_ls.append(torch.norm(features[:, :, :3], 2, 2, keepdim=True))

        voxel_feats = torch.cat(features_ls, dim=-1)
        # zero out padded points
        mask = get_paddings_indicator(num_points, features.shape[1])
        voxel_feats *= mask.unsqueeze(-1).type_as(voxel_feats)

        for vfe in self.vfe_layers:
            voxel_feats = vfe(voxel_feats)
        return voxel_feats
