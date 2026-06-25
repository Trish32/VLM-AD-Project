"""
ResNet-50 backbone + single-level FPN neck.

Matches the BEVFormer-tiny_fp16 official checkpoint:
  img_backbone: ResNet-50, frozen_stages=1, out_indices=(3,) → C5 (2048 ch)
  img_neck:     FPN, in_channels=[2048], out_channels=256, num_outs=1

Key-name correspondence (official checkpoint → our model):
  img_backbone.*  →  backbone.*
  img_neck.*      →  neck.*

The torchvision ResNet-50 attribute names (conv1, bn1, layer1 … layer4) are
kept verbatim so the backbone keys match without any further renaming.
The FPN ConvModule sub-keys (.conv.weight / .conv.bias) match mmdet with
norm_cfg=None, act_cfg=None.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


# ── FPN building block ─────────────────────────────────────────────────────────

class _ConvModule(nn.Module):
    """
    Minimal ConvModule matching mmdet's ConvModule(norm_cfg=None, act_cfg=None).
    Key structure: .conv.weight  /  .conv.bias
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── Backbone ───────────────────────────────────────────────────────────────────

class ResNet50Backbone(nn.Module):
    """
    ResNet-50 C5 backbone (stride 32, 2048-channel output).

    frozen_stages=1  →  conv1, bn1, layer1 parameters frozen.
    norm_eval=True   →  all BatchNorm layers kept in eval mode.

    Attribute names mirror torchvision / mmdet so that the official
    BEVFormer-tiny_fp16 checkpoint keys load directly after the prefix
    substitution  img_backbone.*  →  backbone.*
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        r = tvm.resnet50(weights=weights)

        # Keep the same attribute names as torchvision (= mmdet naming)
        self.conv1   = r.conv1      # 7×7, stride 2
        self.bn1     = r.bn1
        self.relu    = r.relu
        self.maxpool = r.maxpool    # 3×3, stride 2
        self.layer1  = r.layer1    # 256 ch,  stride /4
        self.layer2  = r.layer2    # 512 ch,  stride /8
        self.layer3  = r.layer3    # 1024 ch, stride /16
        self.layer4  = r.layer4    # 2048 ch, stride /32  (C5)

        # Freeze stem + layer1 (frozen_stages=1)
        for p in (list(self.conv1.parameters()) +
                  list(self.bn1.parameters()) +
                  list(self.layer1.parameters())):
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        """Keep all BatchNorm layers in eval mode (norm_eval=True)."""
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x  : (B*num_cams, 3, H, W)
        out: (B*num_cams, 2048, H//32, W//32)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# ── FPN neck ───────────────────────────────────────────────────────────────────

class FPNNeck(nn.Module):
    """
    Single-level FPN neck matching BEVFormer-tiny's img_neck:
      in_channels=[2048], out_channels=256, num_outs=1

    Key structure under 'neck.' prefix:
      neck.lateral_convs.0.conv.weight   (256 × 2048 × 1 × 1)
      neck.lateral_convs.0.conv.bias     (256,)
      neck.fpn_convs.0.conv.weight       (256 × 256  × 3 × 3)
      neck.fpn_convs.0.conv.bias         (256,)

    This matches mmdet FPN with norm_cfg=None, act_cfg=None so the official
    checkpoint loads via  img_neck.*  →  neck.*
    """
    def __init__(self, in_ch: int = 2048, out_ch: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            _ConvModule(in_ch,  out_ch, 1),           # 1×1 projection
        ])
        self.fpn_convs = nn.ModuleList([
            _ConvModule(out_ch, out_ch, 3, padding=1), # 3×3 refine
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x  : (B*num_cams, 2048, Hf, Wf)
        out: (B*num_cams,  256, Hf, Wf)
        """
        x = self.lateral_convs[0](x)
        x = self.fpn_convs[0](x)
        return x
