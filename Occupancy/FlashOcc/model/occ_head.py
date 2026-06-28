"""BEVOCCHead2D occupancy head.

Pure-PyTorch reimplementation of
``projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py::BEVOCCHead2D``.

A 2D BEV feature (B, C, Dy, Dx) is turned into a per-voxel class distribution
(B, Dx, Dy, Dz, n_cls) by a 3x3 conv followed by an MLP "predicter" that
expands the channel dim into Dz*n_cls logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvModule(nn.Module):
    """ConvModule exposing ``.conv`` (no norm/act) -> checkpoint key match."""

    def __init__(self, in_c, out_c, k, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)


class BEVOCCHead2D(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, Dz=16, num_classes=18,
                 use_predicter=True):
        super().__init__()
        self.Dz = Dz
        self.num_classes = num_classes
        self.use_predicter = use_predicter
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = _ConvModule(in_dim, out_channels, 3, padding=1)
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Softplus(),
                nn.Linear(out_dim * 2, num_classes * Dz),
            )

    def forward(self, img_feats):
        """Turn the flat BEV feature into a 3D voxel class volume.

        FlashOcc's key efficiency trick: it never builds a 3D feature volume.
        Instead a 2D BEV map (one vector per x,y column) is fed through an MLP
        that outputs Dz*n_cls numbers per column, which are then *reshaped*
        into the height (Dz) and class axes. So the height structure is
        predicted by the channel-mixing MLP, not by 3D convolutions.

        Args:
            img_feats: (B, C, Dy, Dx)
        Returns:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
        """
        # final_conv smooths the BEV feature; permute puts x,y last->first so
        # the trailing axis is the per-column channel vector for the MLP.
        # (B,C,Dy,Dx) -> (B,Dx,Dy,C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # MLP per column: C -> 2C -> Dz*n_cls, then split height vs class.
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)
        return occ_pred

    @staticmethod
    def get_occ(occ_pred):
        """Argmax over classes -> (B, Dx, Dy, Dz) uint8 label volume.

        softmax is monotonic so it doesn't change the argmax; it's kept only to
        mirror the official head (and in case scores are needed later).
        """
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        return occ_res
