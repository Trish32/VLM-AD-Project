"""
Shared Sparse4D base — backbone, image normalisation, and meta-tensor helpers
used by Sparse4Dv1/v2 (sparse4d_v2.py) and Sparse4Dv3 (sparse4d_v3.py).

Device: MPS → CUDA → CPU (auto-selected at construction).
FP32 only; no autocast anywhere.
Image normalisation: ImageNet mean/std (RGB).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import Sparse4DBackbone


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUSCENES_MEAN = torch.tensor([123.675, 116.280, 103.530])  # RGB
NUSCENES_STD  = torch.tensor([ 58.395,  57.120,  57.375])


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _Sparse4DBase(nn.Module):
    """Common backbone + feature extraction shared by all Sparse4D variants."""

    NUM_CAMS   = 6
    EMBED_DIMS = 256
    NUM_GROUPS = 8
    NUM_LEVELS = 4
    NUM_PTS    = 13   # 7 fixed + 6 learnable, matching reference checkpoint
    NUM_CLASSES = 10

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self.device = _get_device()
        self.backbone = Sparse4DBackbone(pretrained=pretrained_backbone)

    # ------------------------------------------------------------------
    # Image normalisation
    # ------------------------------------------------------------------

    def _normalize(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, N_cam, 3, H, W) float [0, 255] → normalised."""
        mean = NUSCENES_MEAN.to(imgs.device)[None, None, :, None, None]
        std  = NUSCENES_STD .to(imgs.device)[None, None, :, None, None]
        return (imgs - mean) / std

    # ------------------------------------------------------------------
    # Extract per-camera FPN features
    # ------------------------------------------------------------------

    def _extract_features(
        self, imgs: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        imgs : (B, N_cam, 3, H, W)
        Returns
        -------
        feature_maps   : list of 4 tensors (B*N_cam, 256, H_l, W_l)
        spatial_shapes : (4, 2) long
        """
        B, N_cam, C, H, W = imgs.shape
        imgs_flat = imgs.reshape(B * N_cam, C, H, W)
        feature_maps, spatial_shapes = self.backbone(imgs_flat)
        return feature_maps, spatial_shapes

    # ------------------------------------------------------------------
    # Prepare meta tensors (projection_mat, image_wh)
    # ------------------------------------------------------------------

    @staticmethod
    def _meta_to_tensors(img_metas: dict, device: torch.device):
        """
        Converts numpy arrays in img_metas to float32 tensors on device.
        Returns
        -------
        projection_mat : (1, N_cam, 4, 4)
        image_wh       : (1, N_cam, 2)
        ego2global     : (1, 4, 4)
        """
        proj  = torch.from_numpy(img_metas['projection_mat']).float().to(device)
        proj  = proj.unsqueeze(0)                              # (1, N_cam, 4, 4)

        e2g   = torch.from_numpy(img_metas['ego2global']).float().to(device)
        e2g   = e2g.unsqueeze(0)                               # (1, 4, 4)

        wh_np = img_metas['img_wh']                            # (N_cam, 2)
        wh    = torch.from_numpy(wh_np).float().to(device).unsqueeze(0)  # (1, N_cam, 2)

        return proj, wh, e2g
