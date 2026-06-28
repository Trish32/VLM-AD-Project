"""Image neck (CustomFPN) and BEV neck (FPN_LSS).

Pure-PyTorch reimplementation mirroring
``projects/mmdet3d_plugin/models/necks/fpn.py`` (CustomFPN) and
``necks/lss_fpn.py`` (FPN_LSS).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomFPN(nn.Module):
    """Feature Pyramid Network, FlashOcc variant.

    For flashocc-r50: in_channels=[1024, 2048], out_channels=256, num_outs=1,
    start_level=0, out_ids=[0], with NO norm/activation on the conv modules
    (the checkpoint only has conv weight+bias for each lateral / fpn conv).

    Forward returns a single feature map (the only ``out_id``):
    lateral(C4) gets the up-sampled lateral(C5) added to it, then a 3x3 fpn
    conv is applied.
    """

    def __init__(self, in_channels=(1024, 2048), out_channels=256,
                 out_ids=(0,)):
        super().__init__()
        self.out_ids = list(out_ids)
        self.lateral_convs = nn.ModuleList([
            _ConvModule(c, out_channels, 1) for c in in_channels
        ])
        # one fpn conv per out_id (3x3)
        self.fpn_convs = nn.ModuleList([
            _ConvModule(out_channels, out_channels, 3, padding=1)
            for _ in self.out_ids
        ])

    def forward(self, inputs):
        """Args: inputs = [C4 (B,1024,16,44), C5 (B,2048,8,22)].

        Returns the single fused map (B, 256, 16, 44).
        """
        # 1x1 lateral convs map each scale to the common 256-channel width.
        laterals = [lat(x) for lat, x in zip(self.lateral_convs, inputs)]
        # Top-down pathway: walk from the coarsest level down, upsampling each
        # to the next-finer resolution and adding it in. Nearest interpolation
        # matches the official CustomFPN (no learned upsampling here).
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')  # C5 is upsampled (nearest) and added onto C4
        # A 3x3 conv smooths the aliasing introduced by the additive upsample.
        outs = [self.fpn_convs[j](laterals[i])
                for j, i in enumerate(self.out_ids)]
        return outs[0]      # only out_id 0 is used (num_outs == 1)


class _ConvModule(nn.Module):
    """mmcv-style ConvModule wrapper exposing ``.conv`` so the checkpoint keys
    ``<name>.conv.weight`` / ``.conv.bias`` map directly."""

    def __init__(self, in_c, out_c, k, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)


class FPN_LSS(nn.Module):
    """BEV neck. Upsamples the coarsest BEV feature, concatenates with the
    finest, fuses with two 3x3 conv-bn-relu, then a final 2x upsample block.

    For flashocc-r50: in_channels = 512 + 128 = 640, out_channels = 256,
    scale_factor=4, input_feature_index=(0, 2), extra_upsample=2.
    Input  feats = [(B,128,100,100), (B,256,50,50), (B,512,25,25)]
    Output       = (B, 256, 200, 200)
    """

    def __init__(self, in_channels=640, out_channels=256, scale_factor=4,
                 input_feature_index=(0, 2), extra_upsample=2):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)
        channels_factor = 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor,
                      out_channels * channels_factor, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * channels_factor),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=extra_upsample, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(out_channels * channels_factor, out_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, padding=0),
        )

    def forward(self, feats):
        """Fuse the finest and coarsest BEV levels, upsample to full grid.

        Only two of the three pyramid levels are used (indices 0 and 2): the
        coarse level carries large-context BEV semantics, the fine level keeps
        spatial detail. Concatenate-then-conv (rather than FPN-style add) lets
        the conv learn how to weight the two. A final 2x block lifts the
        100x100 fused map to the 200x200 occupancy resolution.
        """
        x2 = feats[self.input_feature_index[0]]     # (B,128,100,100) fine
        x1 = feats[self.input_feature_index[1]]     # (B,512,25,25)   coarse
        x1 = self.up(x1)                            # bilinear x4 -> (B,512,100,100)
        x = torch.cat([x2, x1], dim=1)              # (B,640,100,100)
        x = self.conv(x)                            # fuse -> (B,512,100,100)
        x = self.up2(x)                             # x2 + project -> (B,256,200,200)
        return x
