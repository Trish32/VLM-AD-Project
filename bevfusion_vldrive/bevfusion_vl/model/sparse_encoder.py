"""
SparseEncoder (VoxelNet LiDAR backbone) — pure-PyTorch port for MIT BEVFusion.
Module names mirror the checkpoint (conv_input, encoder_layers.encoder_layerN,
conv_out) so the 126 lidar tensors load directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spconv import SparseTensor, SubMConv3d, SparseConv3d, to_dense


class SparseBatchNorm(nn.BatchNorm1d):
    """BatchNorm1d over a SparseTensor's features (N, C). Since features are stored as a
    flat (N, C) list of occupied voxels, ordinary 1D BatchNorm over the channel dim is
    exactly the right normalization -- no need for a spatial 3D BN(adv for sparse representation)."""

    def forward(self, x: SparseTensor) -> SparseTensor:
        feats = super().forward(x.features)          # (N, C) -> (N, C) normalized
        return SparseTensor(feats, x.indices, x.spatial_shape, x.batch_size)


class SparseReLU(nn.Module):
    # ReLU on the (N, C) feature list; coords/shape pass through unchanged.
    def forward(self, x: SparseTensor) -> SparseTensor:
        return SparseTensor(F.relu(x.features), x.indices, x.spatial_shape, x.batch_size)


def _bn(c):
    # match the checkpoint's BN hyperparams (eps 1e-3, momentum 0.01)
    return SparseBatchNorm(c, eps=1e-3, momentum=0.01)


class SparseBasicBlock(nn.Module):
    """Residual block, sparse version: two submanifold 3x3x3 convs + a skip connection.
    Both convs are SUBMANIFOLD, so the occupied-voxel set is identical in and out -- which
    is what lets `out.features + identity` line up element-for-element (same coords)."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = SubMConv3d(channels, channels, 3)
        self.bn1 = _bn(channels)
        self.conv2 = SubMConv3d(channels, channels, 3)
        self.bn2 = _bn(channels)

    def forward(self, x: SparseTensor) -> SparseTensor:
        identity = x.features                        # Features live as a flat (N, C) list skip path
        out = self.conv1(x)                          # SubMConv -> (N, C), same coords
        out = self.bn1(out)
        out = SparseTensor(F.relu(out.features), out.indices, out.spatial_shape, out.batch_size)
        out = self.conv2(out)                        # SubMConv -> (N, C), same coords
        out = self.bn2(out)
        # add skip (valid only because submanifold keeps voxel order identical), then ReLU
        out = SparseTensor(F.relu(out.features + identity), out.indices,
                           out.spatial_shape, out.batch_size)
        return out


class SparseEncoder(nn.Module):
    """VoxelNet LiDAR backbone. 
    The stage builder encodes one rule: the last block of stages 1–3 is a strided downsample, 
    everything else is a residual block, and the final stage doesn't downsample at all.
    conv_input -> 4 encoder stages (submanifold residual
    blocks, with stages 1-3 ending in a strided downsample) -> conv_out (collapses z) ->
    fold z into channels -> dense 2D BEV. 
    Module names mirror the checkpoint so the 126 lidar tensors load 0/0.

    encoder_channels[i] lists the block output-channels of stage i; the LAST entry of
    stages 0-2 is the strided (downsampling) conv, the rest are residual blocks. Stage 3
    has no downsample. 
    encoder_paddings[i] gives the z-padding of each stride conv (chosen
    so the spatial dims halve cleanly while z shrinks in a controlled way)."""

    def __init__(self, in_channels=5, sparse_shape=(1440, 1440, 41),
                 base_channels=16, output_channels=128,
                 encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                 encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0))):
        super().__init__()
        self.sparse_shape = list(sparse_shape)       # [X, Y, Z] e.g. [1440, 1440, 41]

        # stem: submanifold conv lifting the 5-dim voxel feature to base_channels 16
        self.conv_input = nn.Sequential(
            SubMConv3d(in_channels, base_channels, 3), _bn(base_channels), SparseReLU())

        self.encoder_layers = nn.ModuleDict()
        in_c = base_channels
        n_stage = len(encoder_channels)
        # encoder_channels[i] lists the block output-channels of stage i
        for i, blocks in enumerate(encoder_channels):
            layer = []
            for j, out_c in enumerate(blocks):
                is_last = (j == len(blocks) - 1)
                if is_last and i != n_stage - 1:
                    # last block of stages 0-2 = strided conv that downsamples (x,y,z all /2)
                    pad = encoder_paddings[i][j]
                    pad = (pad, pad, pad) if isinstance(pad, int) else tuple(pad)
                    layer.append(nn.Sequential(
                        SparseConv3d(in_c, out_c, (3, 3, 3), (2, 2, 2), pad),
                        _bn(out_c), SparseReLU()))
                    in_c = out_c
                else:
                    # interior blocks = submanifold residual blocks (no resolution change)
                    layer.append(SparseBasicBlock(in_c))
            # encoder_layer1: 2 residual blocks @16, then strided SparseConv3d 16→32, conv_input(1440 × 1440 × 41) → (720 × 720 × 21)
            # encoder_layer2: 2 residual blocks @32, then strided SparseConv3d 32→64, (720 × 720 × 21) → (360 × 360 × 11)
            # encoder_layer3: 2 residual blocks @64, then strided SparseConv3d 64→128 (z-pad 0), (360 × 360 × 11) → (180 × 180 × 5)
            # encoder_layer4: 2 residual blocks @128, no downsample, (180 × 180 × 5) → (180 × 180 × 5)
            self.encoder_layers[f'encoder_layer{i + 1}'] = nn.Sequential(*layer)

        # collapse height: kernel/stride act ONLY on z k=(1,1,3)/s=(1,1,2) -> crush the height dimension z: 5 -> 2 before folding it into channels
        self.conv_out = nn.Sequential(
            SparseConv3d(in_c, output_channels, (1, 1, 3), (1, 1, 2), (0, 0, 0)),
            _bn(output_channels), SparseReLU())  # (180 × 180 × 5) → (180 × 180 × 2)

    def forward(self, voxel_features, coors, batch_size):
        """voxel_features (N,5) mean-pooled per voxel; coors (N,4) int [b, x, y, z].
        Returns dense BEV (B, output_channels*Zout, X', Y') = (B, 256, 180, 180) for det."""
        x = SparseTensor(voxel_features, coors.long(), self.sparse_shape, batch_size)
        x = self.conv_input(x)                    # stem -> 16ch, full res
        for layer in self.encoder_layers.values():
            x = layer(x)                          # 4 stages: downsample x,y,z; deepen channels
        x = self.conv_out(x)                      # crush z to 2 -> 128ch sparse tensor
        # to_dense → permute → view folds those 2 z-slices into the channel axis: 128 ch × 2 = 256. 
        dense = to_dense(x)                       # (B, C, S0, S1, S2) = (B,128,180,180,2)
        N, C, S0, S1, S2 = dense.shape
        # fold the z-axis (S2) into channels(128 ch × 2 z → channels): (B,128,180,180,2) -> (B, 128*2=256, 180, 180)
        dense = dense.permute(0, 1, 4, 2, 3).contiguous().view(N, C * S2, S0, S1)
        return dense                              # (B, 256, 180, 180) lidar BEV — a 2D feature map ready to fuse with the camera BEV
