"""
CBSwinTransformer — composite (dual) Swin backbone, pure-PyTorch port of
mmdet3d cbnet.py. Two Swin modules; the first runs normally, its multi-scale
features are fed (via cb_linears + spatial interpolation) into the second.
Only the second module's outputs are returned.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin import SwinTransformer


class CBSwinTransformer(nn.Module):
    def __init__(self, embed_dim=96, cb_del_stages=1, **kwargs):
        super().__init__()
        self.cb_del_stages = cb_del_stages
        self.cb_modules = nn.ModuleList()
        for cb_idx in range(2):
            cb_module = SwinTransformer(embed_dim=embed_dim, **kwargs)
            if cb_idx > 0:
                cb_module.del_layers(cb_del_stages)
            self.cb_modules.append(cb_module)

        self.num_layers = self.cb_modules[0].num_layers
        cb_inplanes = [embed_dim * 2 ** i for i in range(self.num_layers)]

        self.cb_linears = nn.ModuleList()
        for i in range(self.num_layers):
            linears = nn.ModuleList()
            if i >= self.cb_del_stages - 1:
                jrange = 4 - i
                for j in range(jrange):
                    if cb_inplanes[i + j] != cb_inplanes[i]:
                        layer = nn.Conv2d(cb_inplanes[i + j], cb_inplanes[i], 1)
                    else:
                        layer = nn.Identity()
                    linears.append(layer)
            self.cb_linears.append(linears)

    def spatial_interpolate(self, x, H, W):
        B, C = x.shape[:2]
        if H != x.shape[2] or W != x.shape[3]:
            x = F.interpolate(x, size=(H, W), mode='nearest')
        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        return x

    def _get_cb_feats(self, feats, tmps):
        cb_feats = []
        Wh, Ww = tmps[0][-2:]
        for i in range(self.num_layers):
            feed = 0
            if i >= self.cb_del_stages - 1:
                jrange = 4 - i
                for j in range(jrange):
                    tmp = self.cb_linears[i][j](feats[j + i])
                    tmp = self.spatial_interpolate(tmp, Wh, Ww)
                    feed = feed + tmp
            cb_feats.append(feed)
            Wh, Ww = tmps[i + 1][-2:]
        return cb_feats

    def forward(self, x):
        outs = []
        cb_feats = None
        for i, module in enumerate(self.cb_modules):
            if i == 0:
                feats, tmps = module(x)
            else:
                feats, tmps = module(x, cb_feats, tmps)
            outs.append(feats)
            if i < len(self.cb_modules) - 1:
                cb_feats = self._get_cb_feats(outs[-1], tmps)
        if len(outs) > 1:
            outs = outs[-1]
        return tuple(outs)
