"""
GeneralizedLSSFPN neck (MIT BEVFusion camera neck). Top-down: each level i is
cat(level_i, upsample(level_{i+1})) -> lateral 1x1 (conv+BN+ReLU) -> fpn 3x3
(conv+BN+ReLU). in [192,384,768] out 256, returns 2 levels (/8, /16).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """ConvModule(conv -> BN2d -> ReLU). conv has bias=False (norm follows)."""

    def __init__(self, in_c, out_c, k, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.activate = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class GeneralizedLSSFPN(nn.Module):
    """Fuses the Swin feature pyramid into a single-resolution camera feature map.

    KEY DIFFERENCE from a standard FPN: here each level is CONCATENATED with the
    upsampled coarser level *before* the lateral 1x1 conv (a standard FPN ADDS them
    *after* the lateral conv). So the lateral conv's input channels = in[i] + (channels
    of the upsampled neighbor), which is why `extra` is added to in_channels[i] below.
    in (192,384,768) -> out 256; returns the 2 finer levels (/8, /16)."""

    def __init__(self, in_channels=(192, 384, 768), out_channels=256, num_outs=3,
                 start_level=0):
        super().__init__()
        self.in_channels = list(in_channels)
        self.num_ins = len(in_channels)         # 3 pyramid levels in
        self.start_level = start_level
        self.backbone_end_level = self.num_ins - 1   # produce num_ins-1 = 2 outputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(start_level, self.backbone_end_level):
            # `extra` = channels of the upsampled neighbor concatenated onto level i:
            #   topmost pair uses the raw coarse level (in_channels[i+1]); else it's the
            #   already-fused out_channels.
            extra = in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels
            self.lateral_convs.append(ConvBNReLU(in_channels[i] + extra, out_channels, 1))  # 1x1 fuse->256
            self.fpn_convs.append(ConvBNReLU(out_channels, out_channels, 3, padding=1))     # 3x3 smooth

    def forward(self, inputs):
        # inputs: list of 3 maps [(B,192,32,88), (B,384,16,44), (B,768,8,22)]
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]
        used = len(laterals) - 1                # = 2 output levels
        # top-down: coarsest -> finest, fusing each level with the upsampled coarser one
        for i in range(used - 1, -1, -1):
            x = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:],
                              mode='bilinear', align_corners=True)   # upsample coarser to level i size
            laterals[i] = torch.cat([laterals[i], x], dim=1)         # CONCAT (not add) before lateral
            laterals[i] = self.lateral_convs[i](laterals[i])         # 1x1 -> out_channels
            laterals[i] = self.fpn_convs[i](laterals[i])             # 3x3 refine
        return [laterals[i] for i in range(used)]   # [(B,256,32,88), (B,256,16,44)]; vtransform uses [0]
