"""
SECOND backbone + SECONDFPN neck — pure-PyTorch ports matching mmdet3d.

SECOND: each block = [stride-conv, BN, ReLU] + layer_num*[conv, BN, ReLU].
  Sequential indices match the checkpoint (conv at 0,3,6,..., BN at 1,4,7,...).
SECONDFPN: per-level ConvTranspose2d(stride) -> BN -> ReLU, then concat.
  For stride==1 mmdet3d still uses a deconv with kernel=1 (matches ckpt shapes).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _bn(c):
    return nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)


class SECOND(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=(64, 128, 256),
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2)):
        super().__init__()
        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                nn.Conv2d(in_filters[i], out_channels[i], 3,
                          stride=layer_strides[i], padding=1, bias=False),
                _bn(out_channels[i]),
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_num):
                block.append(nn.Conv2d(out_channels[i], out_channels[i], 3,
                                       padding=1, bias=False))
                block.append(_bn(out_channels[i]))
                block.append(nn.ReLU(inplace=True))
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        return tuple(outs)


class SECONDFPN(nn.Module):
    def __init__(self,
                 in_channels=(64, 128, 256),
                 out_channels=(128, 128, 128),
                 upsample_strides=(1, 2, 4)):
        super().__init__()
        deblocks = []
        for i, out_c in enumerate(out_channels):
            stride = upsample_strides[i]
            up = nn.ConvTranspose2d(in_channels[i], out_c,
                                    kernel_size=stride, stride=stride, bias=False)
            deblocks.append(nn.Sequential(up, _bn(out_c), nn.ReLU(inplace=True)))
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, x):
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        out = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
        return [out]
