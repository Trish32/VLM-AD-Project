"""
Dense depth head for Sparse4D-v2 dense-depth supervision — pure PyTorch.

A small per-camera conv stack on one FPN level predicts a single-channel metric
depth map.  Supervised by sparse LiDAR depth (data/lidar_depth.py).  This is a
TRAINING-ONLY auxiliary branch: it shapes the image backbone's geometric
features and is discarded at inference (the detection forward never calls it).

Output depth is forced positive with softplus so the L1 target (metres) is
always well-defined.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    """
    Parameters
    ----------
    in_channels : FPN feature channels (256)
    hidden      : conv hidden width
    level       : which FPN level to predict on (0 = stride 4, highest res)
    """

    def __init__(self, in_channels: int = 256, hidden: int = 256, level: int = 0):
        super().__init__()
        self.level = level
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),      nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        """
        feature_maps : list of (B*N_cam, C, H_l, W_l)
        Returns depth : (B*N_cam, H_sel, W_sel)  positive metres
        """
        feat = feature_maps[self.level]
        depth = self.net(feat).squeeze(1)            # (B*N_cam, H, W)
        return F.softplus(depth)


def depth_l1_loss(
    pred:   torch.Tensor,   # (B*N_cam, H, W)  predicted metric depth
    target: torch.Tensor,   # (B*N_cam, H, W)  GT metric depth (0 = no point)
    mask:   torch.Tensor,   # (B*N_cam, H, W)  bool, True where GT valid
) -> torch.Tensor:
    """Masked L1 between predicted and LiDAR depth on valid pixels only."""
    if mask.sum() == 0:
        return pred.sum() * 0.0          # keep graph alive, no valid pixels
    # Align spatial size if the head predicts at a different resolution
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.unsqueeze(1), size=target.shape[-2:],
                             mode='bilinear', align_corners=False).squeeze(1)
    diff = (pred - target).abs()[mask]
    return diff.mean()
