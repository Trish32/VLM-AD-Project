"""
FPN (mmdet) base + FPNC neck — pure-PyTorch port for bevf_pp.

The checkpoint contains only conv weights+biases (no norm, no activation
params) for img_neck, consistent with FPNC built with norm_cfg=None, act_cfg=None.
FPN: 4 lateral 1x1 convs + 4 fpn 3x3 convs, top-down nearest upsample, plus one
extra max-pool output (num_outs=5). FPNC: adaptively resize all 5 outputs to
target_size=(final_dim//downsample)=(112,200), concat, reduc_conv -> outC.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNC(nn.Module):
    def __init__(self, in_channels=(96, 192, 384, 768), out_channels=256,
                 num_outs=5, outC=256, final_dim=(900, 1600), downsample=8,
                 use_adp=True):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.use_adp = use_adp
        self.target_size = (final_dim[0] // downsample, final_dim[1] // downsample)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for ic in in_channels:
            self.lateral_convs.append(_ConvModule(ic, out_channels, 1, padding=0))
            self.fpn_convs.append(_ConvModule(out_channels, out_channels, 3, padding=1))

        if use_adp:
            adp_list = []
            for i in range(num_outs):
                if i == 0:
                    resize = nn.AdaptiveAvgPool2d(self.target_size)
                else:
                    resize = nn.Upsample(size=self.target_size, mode='bilinear', align_corners=True)
                adp_list.append(nn.Sequential(
                    resize, _ConvModule(out_channels, out_channels, 1, padding=0)))
            self.adp = nn.ModuleList(adp_list)

        self.reduc_conv = _ConvModule(out_channels * num_outs, outC, 3, padding=1)

    def _fpn_forward(self, inputs):
        laterals = [lc(inputs[i]) for i, lc in enumerate(self.lateral_convs)]
        used = len(laterals)
        for i in range(used - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used)]
        # extra max-pool output (add_extra_convs=False)
        while len(outs) < self.num_outs:
            outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return outs

    def forward(self, x):
        outs = self._fpn_forward(x)
        if len(outs) > 1:
            resize_outs = []
            if self.use_adp:
                for i in range(len(outs)):
                    resize_outs.append(self.adp[i](outs[i]))
            else:
                for o in outs:
                    if o.shape[2:] != self.target_size:
                        o = F.interpolate(o, self.target_size, mode='bilinear', align_corners=True)
                    resize_outs.append(o)
            out = torch.cat(resize_outs, dim=1)
            out = self.reduc_conv(out)
        else:
            out = outs[0]
        return [out]


class _ConvModule(nn.Module):
    """Conv2d only (no norm/act), bias=True — matches img_neck checkpoint."""

    def __init__(self, in_c, out_c, k, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)
