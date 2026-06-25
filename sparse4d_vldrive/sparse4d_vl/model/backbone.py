"""
ResNet-50 backbone + 4-level FPN neck — pure PyTorch, MPS-compatible.

Matches Sparse4D-v2 config:
  img_backbone : ResNet-50  out_indices=(0,1,2,3) → C2,C3,C4,C5
  img_neck     : FPN  in_channels=[256,512,1024,2048]  out_channels=256  num_outs=4

For a 256×704 input the four FPN output spatial sizes are:
  level 0 (stride  4): 64×176   (C2 lateral)
  level 1 (stride  8): 32×88    (C3 lateral)
  level 2 (stride 16): 16×44    (C4 lateral)
  level 3 (stride 32):  8×22    (C5 lateral)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# ResNet-50 backbone (4 output stages)
# ---------------------------------------------------------------------------

class ResNet50(nn.Module):
    """
    ResNet-50 with four C-stage outputs.
    frozen_stages=1 → conv1, bn1, layer1 frozen (matched to Sparse4D-v2 config).
    norm_eval=True  → all BN in eval mode during train.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        r = tvm.resnet50(weights=weights)

        self.conv1   = r.conv1
        self.bn1     = r.bn1
        self.relu    = r.relu
        self.maxpool = r.maxpool
        self.layer1  = r.layer1   # C2:  256 ch, stride  4
        self.layer2  = r.layer2   # C3:  512 ch, stride  8
        self.layer3  = r.layer3   # C4: 1024 ch, stride 16
        self.layer4  = r.layer4   # C5: 2048 ch, stride 32

        # Freeze stem + layer1 (frozen_stages=1)
        frozen = (
            list(self.conv1.parameters()) +
            list(self.bn1.parameters()) +
            list(self.layer1.parameters())
        )
        for p in frozen:
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        x   : (B, 3, H, W)
        out : [C2, C3, C4, C5]  shapes [(B,256,H/4,W/4), …, (B,2048,H/32,W/32)]
        """
        x  = self.relu(self.bn1(self.conv1(x)))
        x  = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


# ---------------------------------------------------------------------------
# 4-level FPN neck
# ---------------------------------------------------------------------------

class _ConvBN(nn.Module):
    """1×1 or 3×3 conv without norm/act (matches mmdet FPN with norm_cfg=None)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FPN4Level(nn.Module):
    """
    Standard FPN with top-down pathway.
    in_channels  = [256, 512, 1024, 2048]  (C2…C5)
    out_channels = 256
    num_outs     = 4
    """

    def __init__(self, in_channels: list[int] = (256, 512, 1024, 2048),
                 out_ch: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            _ConvBN(c, out_ch, k=1) for c in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            _ConvBN(out_ch, out_ch, k=3, padding=1) for _ in in_channels
        ])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        feats : [C2, C3, C4, C5]
        returns: [P2, P3, P4, P5]  each (B, 256, H_l, W_l)
        """
        # Lateral projections
        laterals = [l(f) for l, f in zip(self.lateral_convs, feats)]

        # Top-down pathway (start from C5)
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )

        # Output convs
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outs   # [P2, P3, P4, P5]


# ---------------------------------------------------------------------------
# Combined backbone + neck
# ---------------------------------------------------------------------------

class Sparse4DBackbone(nn.Module):
    """
    ResNet-50 + 4-level FPN.  Input: (B*N_cam, 3, H, W).
    Output: list of 4 feature maps [(B*N_cam, 256, H_l, W_l)].
    Also returns spatial_shapes: (4, 2) long tensor [(H_l, W_l)].
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNet50(pretrained=pretrained)
        self.neck     = FPN4Level()

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        x             : (B*N_cam, 3, H, W)
        feature_maps  : list of 4 tensors [(B*N_cam, 256, H_l, W_l)]
        spatial_shapes: (4, 2) long  — [(H_l, W_l)] for each level
        """
        c_feats      = self.backbone(x)          # [C2, C3, C4, C5]
        feature_maps = self.neck(c_feats)        # [P2, P3, P4, P5]

        spatial_shapes = torch.tensor(
            [[f.shape[-2], f.shape[-1]] for f in feature_maps],
            dtype=torch.long, device=x.device,
        )                                        # (4, 2)

        return feature_maps, spatial_shapes
