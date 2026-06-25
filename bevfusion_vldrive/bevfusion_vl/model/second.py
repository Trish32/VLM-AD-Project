"""
SECOND backbone + SECONDFPN neck (MIT BEVFusion decoder) + ConvFuser.
in=256 -> [128,256] (strides [1,2]); FPN up [1,2] with use_conv_for_no_stride
(stride-1 = Conv2d 1x1, stride-2 = ConvTranspose2d) -> concat 512.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bn(c):
    return nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)


class SECOND(nn.Module):
    """BEV backbone (a plain 2D conv net over the fused BEV map). Two stages: stage 0
    keeps resolution (stride 1), stage 1 downsamples (stride 2). Each stage = one
    (possibly strided) 3x3 conv that changes channels, then `layer_num` 3x3 convs that
    refine at fixed channels. Returns BOTH stage outputs (different resolutions) for the FPN."""

    def __init__(self, in_channels=256, out_channels=(128, 256),
                 layer_nums=(5, 5), layer_strides=(1, 2)):
        super().__init__()
        in_filters = [in_channels, *out_channels[:-1]]   # [256, 128] -- input ch of each stage
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            # leading conv: changes channels and applies the stage stride (1 or 2)
            block = [nn.Conv2d(in_filters[i], out_channels[i], 3,
                               stride=layer_strides[i], padding=1, bias=False),
                     _bn(out_channels[i]), nn.ReLU(inplace=True)]
            for _ in range(layer_num):       # 5 refinement convs at fixed channels
                block += [nn.Conv2d(out_channels[i], out_channels[i], 3, padding=1, bias=False),
                          _bn(out_channels[i]), nn.ReLU(inplace=True)]
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        # x: (B, 256, 180, 180) fused BEV
        outs = []
        for blk in self.blocks:
            x = blk(x)                       # NOTE: chained -- stage 1 consumes stage 0's output
            outs.append(x)
        return outs                          # [(B,128,180,180), (B,256,90,90)]


class SECONDFPN(nn.Module):
    """Bring SECOND's multi-resolution stages to a COMMON size and concat them. Each input
    level gets its own "deblock": stride-1 levels use a 1x1 conv (use_conv_for_no_stride),
    stride-2 levels use a ConvTranspose to upsample. Outputs concat -> the shared BEV feature."""

    def __init__(self, in_channels=(128, 256), out_channels=(256, 256),
                 upsample_strides=(1, 2)):
        super().__init__()
        deblocks = []
        for i, out_c in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                # upsample the coarse stage back to full BEV resolution
                up = nn.ConvTranspose2d(in_channels[i], out_c, kernel_size=stride,
                                        stride=stride, bias=False)
            else:  # use_conv_for_no_stride=True -> a 1x1 conv instead of identity/deconv
                up = nn.Conv2d(in_channels[i], out_c, kernel_size=1, stride=1, bias=False)
            deblocks.append(nn.Sequential(up, _bn(out_c), nn.ReLU(inplace=True)))
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, feats):
        # feats: [(B,128,180,180), (B,256,90,90)] -> each deblock -> (B,256,180,180)
        ups = [deblock(feats[i]) for i, deblock in enumerate(self.deblocks)]
        return torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]   # (B, 512, 180, 180)


class ConvFuser(nn.Sequential):
    """THE FUSION POINT: concat camera BEV (80ch) + lidar BEV (256ch) on the channel axis,
    then a single 3x3 conv mixes them down to 256ch. This is the whole "fusion" -- both
    streams already share the same BEV grid (by design), so fusing is just concat + conv.
    Subclasses nn.Sequential so checkpoint keys are 0.* (conv) / 1.* (bn), matching official.
    Input order [camera, lidar] matters -- it must match the conv's learned input channels."""

    def __init__(self, in_channels=(80, 256), out_channels=256):
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),  # 336 -> 256
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        # feats = [cam_bev (B,80,180,180), lidar_bev (B,256,180,180)]
        return super().forward(torch.cat(feats, dim=1))   # concat -> (B,336,180,180) -> conv -> (B,256,180,180)
