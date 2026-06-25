"""
BEVFormerEncoder — pure PyTorch, MPS-compatible.

Stacks N BEVFormerLayer blocks, each containing:
  TSA -> LayerNorm -> SCA -> LayerNorm -> FFN -> LayerNorm

Also owns the geometric helpers:
  get_reference_points : builds the 3-D / 2-D BEV anchor grid
  point_sampling       : projects 3-D anchors to each camera plane (FP32, no TF32)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tsa import TemporalSelfAttention
from .sca import SpatialCrossAttention


# ---------------------------------------------------------------------------
# One BEV transformer layer  (TSA + SCA + FFN)
# ---------------------------------------------------------------------------

class BEVFormerLayer(nn.Module):

    def __init__(self, embed_dim=256, num_cams=6, ffn_dim=512,
                 num_heads=8, num_points_sca=8, num_points_tsa=4,
                 num_z=4, bev_h=50, bev_w=50, dropout=0.1):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.tsa  = TemporalSelfAttention(embed_dim=embed_dim, num_heads=num_heads,
                                          num_levels=1, num_points=num_points_tsa,
                                          num_bev_queue=2, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.sca  = SpatialCrossAttention(embed_dim=embed_dim, num_cams=num_cams,
                                          num_heads=num_heads, num_levels=1,
                                          num_points=num_points_sca, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, bev_query, feat_flat, ref_2d, ref_3d_cam, bev_mask,
                img_spatial_shapes, level_start_index, prev_bev=None, bev_pos=None):
        """
        bev_query        : (B, L, C)
        feat_flat        : (num_cams, H*W, B, C)
        ref_2d           : (B*2, L, 1, 2)
        ref_3d_cam       : (num_cams, B, L, num_Z, 2)
        bev_mask         : (num_cams, B, L, num_Z)
        img_spatial_shapes: (1, 2)  — image feature map (H, W)
        level_start_index: (1,)
        prev_bev         : (B, L, C) or None
        bev_pos          : (1, L, C) or None — learned BEV positional encoding
        """
        device = bev_query.device

        # TSA operates on BEV grid, so spatial_shapes = [(bev_h, bev_w)]
        bev_shapes = torch.tensor([[self.bev_h, self.bev_w]],
                                   dtype=torch.long, device=device)
        bev_level_idx = torch.zeros(1, dtype=torch.long, device=device)

        # bev_pos is passed to TSA and SCA so they can add it to Q before attn
        q = self.norm1(self.tsa(bev_query, prev_bev, ref_2d, bev_shapes, bev_level_idx,
                                query_pos=bev_pos))

        # SCA operates on image feature maps
        q = self.norm2(self.sca(q, feat_flat, ref_3d_cam, bev_mask,
                                img_spatial_shapes, level_start_index,
                                query_pos=bev_pos))

        q = self.norm3(q + self.ffn(q))
        return q


# ---------------------------------------------------------------------------
# BEVFormer Encoder
# ---------------------------------------------------------------------------

class BEVFormerEncoder(nn.Module):

    def __init__(
        self,
        embed_dim:       int   = 256,
        bev_h:           int   = 50,
        bev_w:           int   = 50,
        num_cams:        int   = 6,
        num_layers:      int   = 3,
        num_points_in_pillar: int = 4,
        pc_range:        list  = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        ffn_dim:         int   = 512,
        num_heads:       int   = 8,
        num_points_sca:  int   = 8,
        num_points_tsa:  int   = 4,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.bev_h      = bev_h
        self.bev_w      = bev_w
        self.pc_range   = pc_range
        self.num_z      = num_points_in_pillar

        self.layers = nn.ModuleList([
            BEVFormerLayer(embed_dim=embed_dim, num_cams=num_cams,
                           ffn_dim=ffn_dim, num_heads=num_heads,
                           num_points_sca=num_points_sca,
                           num_points_tsa=num_points_tsa,
                           num_z=num_points_in_pillar,
                           bev_h=bev_h, bev_w=bev_w,
                           dropout=dropout)
            for _ in range(num_layers)
        ])

    # ------------------------------------------------------------------
    # Geometric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_reference_points(H, W, Z=8, num_z=4, dim='3d',
                              bs=1, device='cpu', dtype=torch.float32):
        """Build normalized reference point grid on the BEV plane. [0,1] inside pc_range """
        if dim == '3d':
            # zs = linspace(0.5, 3.5, 4).view(4,1,1).expand(4,50,50) / 4 -> 4 height levels at z∈{0.125, 0.375, 0.625, 0.875} (normalised [0,1])
            zs = torch.linspace(0.5, Z - 0.5, num_z, dtype=dtype, device=device
                                ).view(-1, 1, 1).expand(num_z, H, W) / Z  
            # xs = linspace(0.5, 49.5, 50).view(1,1,50).expand(4,50,50) / 50 → x grid across BEV width, each in [0.01, 0.99]
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device
                                ).view(1, 1, W).expand(num_z, H, W) / W
            # ys = linspace(0.5, 49.5, 50).view(1,50,1).expand(4,50,50) / 50 → y grid across BEV height, each in [0.01, 0.99]
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device
                                ).view(1, H, 1).expand(num_z, H, W) / H
            ref = torch.stack((xs, ys, zs), -1)              # (num_z, H, W, 3)
            ref = ref.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # (num_z, H*W, 3)
            ref = ref[None].repeat(bs, 1, 1, 1)               # (bs, num_z, H*W, 3)
            return ref

        # '2d' — reference points for TSA
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            indexing='ij',
        )
        ref_y = ref_y.reshape(-1)[None] / H                  # (1, H*W)
        ref_x = ref_x.reshape(-1)[None] / W
        ref   = torch.stack((ref_x, ref_y), -1)              # (1, H*W, 2)
        ref   = ref.repeat(bs, 1, 1).unsqueeze(2)             # (bs, H*W, 1, 2)
        return ref

    def point_sampling(self, ref_3d, img_metas, device):
        """
        Project 3-D BEV anchors into each camera image plane.

        ref_3d : (bs, num_z, H*W, 3)  — normalised [0,1] inside pc_range
        returns:
          ref_cam : (num_cams, bs, H*W, num_z, 2)  — image-normalised [0,1]
          bev_mask: (num_cams, bs, H*W, num_z)      — True where visible
        """
        # Step A: Collect lidar->img matrices  (bs, num_cams, 4, 4)
        lidar2img = np.stack(
            [np.stack(meta['lidar2img'], axis=0) for meta in img_metas], axis=0
        )  
        lidar2img = torch.tensor(lidar2img, dtype=torch.float32, device=device)  # (bs, num_cams, 4, 4)

        pc = self.pc_range
        ref = ref_3d.clone().float()
        ref[..., 0] = ref[..., 0] * (pc[3] - pc[0]) + pc[0]  # x: [0,1] → [-51.2, +51.2] m
        ref[..., 1] = ref[..., 1] * (pc[4] - pc[1]) + pc[1]  # y: [0,1] → [-51.2, +51.2] m
        ref[..., 2] = ref[..., 2] * (pc[5] - pc[2]) + pc[2]  # z: [0,1] → [-5.0, +3.0] m

        # Step B — homogeneous coords: (bs, num_z, H*W, 4)
        ones = torch.ones(*ref.shape[:-1], 1, dtype=ref.dtype, device=device)  # (bs, num_z, H*W, 1)
        ref_h = torch.cat([ref, ones], dim=-1)               # (bs, nz, HW, 3)+(bs, nz, HW, 1)->(bs, nz, HW, 4)

        bs, nz, HW, _ = ref_h.shape
        num_cams = lidar2img.shape[1]

        # Step C — batch matmul against all 6 lidar2img matrices: (bs, 1, 1, num_cams, 4, 4) x (bs, nz, HW, 1, 4, 1) -> (bs, nz, HW, nc, 4)
        ref_exp  = ref_h.unsqueeze(3).unsqueeze(-1)           # (1, 4, 2500, 1, 4, 1)
        l2i_exp  = lidar2img.unsqueeze(1).unsqueeze(2)        # (1, 1, 1, 6, 4, 4)

        # Step D — perspective divide + image normalisation to get ref_cam in [0,1] relative to image size
        proj = torch.matmul(l2i_exp, ref_exp).squeeze(-1)     # (1, 4, 2500, 6, 4)

        eps = 1e-5
        valid_z = proj[..., 2:3] > eps                        # depth > 0

        u = proj[..., 0:1] / proj[..., 2:3].clamp(min=eps)  # pixel x divided by depth
        v = proj[..., 1:2] / proj[..., 2:3].clamp(min=eps)  # pixel y divided by depth

        # Normalise by image size (stored in img_metas)
        img_h = img_metas[0]['img_shape'][0][0]  # height of the input image (not the feature map) — same for all cams in batch
        img_w = img_metas[0]['img_shape'][0][1]  # width of the input image (not the feature map) — same for all cams in batch
        u = u / img_w  # normalised pixel x in [0,1]
        v = v / img_h

        # Step E — FOV mask
        # Mask: within image bounds and positive depth
        in_bounds = (valid_z & (u > 0) & (u < 1) & (v > 0) & (v < 1))  # (bs, nz, HW, nc, 1)

        # Assemble uv: (bs, nz, HW, nc, 2) — call nan_to_num while contiguous;
        # MPS nan_to_num produces wrong results on non-contiguous views.
        ref_cam = torch.cat([u, v], dim=-1).nan_to_num(0.0)

        # Reorder dims -> (num_cams, bs, HW, nz, 2)
        ref_cam  = ref_cam.permute(3, 0, 2, 1, 4)            # (nc, bs, HW, nz, 2)
        bev_mask = in_bounds.squeeze(-1).permute(3, 0, 2, 1)  # (nc, bs, HW, nz)

        return ref_cam, bev_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        bev_query:   torch.Tensor,   # (L, bs, C)  L = bev_h * bev_w
        feat_flat:   torch.Tensor,   # (num_cams, H*W, bs, C)
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        img_metas:   list,
        prev_bev:    torch.Tensor | None = None,  # (bs, L, C) or (L, bs, C) or None
        shift:       torch.Tensor | None = None,  # (bs, 2) xy shift from ego-motion
        bev_pos:     torch.Tensor | None = None,  # (1, L, C) BEV positional encoding
    ) -> torch.Tensor:               # (bs, L, C)

        L, bs, C = bev_query.shape
        device   = bev_query.device
        dtype    = bev_query.dtype

        # Convert to batch-first
        bev_query = bev_query.permute(1, 0, 2)   # (bs, L, C)

        if prev_bev is not None and prev_bev.shape[0] == L:
            prev_bev = prev_bev.permute(1, 0, 2)  # (bs, L, C)

        # 3-D reference points for SCA  — shape (bs, num_z, L, 3)
        Z_size = self.pc_range[5] - self.pc_range[2]  # 3.0 - (-5.0) = 8.0 metres tall
        ref_3d = self.get_reference_points(
            self.bev_h, self.bev_w, Z=Z_size, num_z=self.num_z,
            dim='3d', bs=bs, device=device, dtype=dtype
        )                                          # (bs, num_z, L, 3)

        # 2-D reference points for TSA
        ref_2d = self.get_reference_points(
            self.bev_h, self.bev_w, dim='2d',
            bs=bs, device=device, dtype=dtype
        )                                          # (bs, L, 1, 2)

        # ego-motion shift （Translation）applied to prev BEV ref points ← deformable attention samples here from the old BEV
        if shift is not None:
            shift_ref_2d = ref_2d.clone()  # (bs, 2500, 1, 2) — current BEV grid coords
            shift_ref_2d += shift[:, None, None, :]  # offset the reference points by the ego-motion shift (in normalised BEV units)
        else:
            shift_ref_2d = ref_2d

        # Stack ref_2d for two bev_queue entries: (bs*2, L, 1, 2)
        hybrid_ref_2d = torch.cat([shift_ref_2d, ref_2d], dim=0)

        # Project 3-D anchors to camera planes — must be FP32, no TF32
        with torch.no_grad():
            ref_cam, bev_mask = self.point_sampling(
                ref_3d, img_metas, device
            )

        # Prepare prev_bev for TSA: (bs*2, L, C)
        if prev_bev is not None:
            prev_flat = torch.cat([prev_bev, bev_query], dim=0)
        else:
            prev_flat = torch.cat([bev_query, bev_query], dim=0)

        for layer in self.layers:
            bev_query = layer(
                bev_query,
                feat_flat,
                hybrid_ref_2d,
                ref_cam,
                bev_mask,
                img_spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_flat[:bs],
                bev_pos=bev_pos,
            )

        return bev_query   # (bs, L, C)
