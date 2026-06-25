"""
Spatial Cross Attention (SCA) — pure PyTorch, MPS-compatible.

Geometric lift-and-project:
  1. BEV reference points are pre-projected to each camera's image plane
     (done in the encoder's point_sampling step).
  2. For each BEV query that is visible in a camera (bev_mask), a small
     deformable offset is learned and added to the reference points.
  3. Image features at those locations are extracted with F.grid_sample.
  4. Weighted sum across (num_points) gives the sampled feature per query
     per camera; cameras are then averaged with valid-count normalisation.

Reference shapes (batch-first):
  query              : (B, bev_h*bev_w, C)
  value              : (num_cams, H*W, B, C)  — flattened image features
  reference_points_cam: (num_cams, B, bev_h*bev_w, num_Z, 2)  in [0,1]
  bev_mask           : (num_cams, B, bev_h*bev_w, num_Z)       bool
"""

import math
import torch
import torch.nn as nn
from .deform_attn import ms_deform_attn_core


class SpatialCrossAttention(nn.Module):

    def __init__(
        self,
        embed_dim:   int   = 256,
        num_cams:    int   = 6,
        num_heads:   int   = 8,
        num_levels:  int   = 1,
        num_points:  int   = 8,
        dropout:     float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_cams   = num_cams
        self.num_heads  = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim   = embed_dim // num_heads

        self.value_proj       = nn.Linear(embed_dim, embed_dim)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.output_proj      = nn.Linear(embed_dim, embed_dim)
        self.dropout          = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2
        ).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid[:, :, i, :] *= i + 1
        self.sampling_offsets.bias = nn.Parameter(grid.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias,   0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query:               torch.Tensor,  # (B, L, C)
        value:               torch.Tensor,  # (num_cams, H*W, B, C)
        reference_points_cam: torch.Tensor, # (num_cams, B, L, num_Z, 2)
        bev_mask:            torch.Tensor,  # (num_cams, B, L, num_Z)  bool
        spatial_shapes:      torch.Tensor,  # (num_levels, 2)
        level_start_index:   torch.Tensor,  # (num_levels,)
        query_pos:           torch.Tensor | None = None,  # (1, L, C) BEV positional encoding
    ) -> torch.Tensor:                      # (B, L, C)

        B, L, _ = query.shape
        num_cams = value.shape[0]
        num_Z    = reference_points_cam.shape[3]

        identity = query  # residual uses original query without pos

        # Add BEV positional encoding to query before computing offsets/weights
        # (matches official SCA: query = query + query_pos inside the module)
        if query_pos is not None:
            query = query + query_pos

        slots = torch.zeros(B, L, self.embed_dim, device=query.device, dtype=query.dtype)

        # Per-camera mask: query is "active" for camera i if any Z is valid
        # queries whose projected points are all outside every camera are skipped entirely via the rebatch
        cam_active = bev_mask.any(dim=-1)   # (num_cams, B, L)

        # Batch-0 indices (identical across batch for static scene geometry)
        cam_indices = []
        for i in range(num_cams):
            idx = cam_active[i, 0].nonzero(as_tuple=False).squeeze(-1)  # (k,)
            cam_indices.append(idx)

        max_len = max((len(idx) for idx in cam_indices), default=0)
        if max_len == 0:
            return self.dropout(self.output_proj(slots)) + identity

        # Rebatch: only forward the visible queries per camera
        q_rebatch   = query.new_zeros(B, num_cams, max_len, self.embed_dim)
        ref_rebatch = reference_points_cam.new_zeros(B, num_cams, max_len, num_Z, 2)

        for j in range(B):
            for i, idx in enumerate(cam_indices):
                if len(idx) == 0:
                    continue
                q_rebatch[j, i,   :len(idx)] = query[j, idx]
                ref_rebatch[j, i, :len(idx)] = reference_points_cam[i, j, idx]

        # Flatten camera into batch dim: (B*num_cams, max_len, C)
        q_flat   = q_rebatch.view(B * num_cams, max_len, self.embed_dim)
        ref_flat = ref_rebatch.view(B * num_cams, max_len, num_Z, 2)  

        # value: (num_cams, H*W, B, C) -> (B*num_cams, H*W, C)
        v_flat = value.permute(2, 0, 1, 3).reshape(B * num_cams, -1, self.embed_dim)
        v_proj = self.value_proj(v_flat)
        v_proj = v_proj.view(B * num_cams, v_proj.shape[1], self.num_heads, self.head_dim)

        # How SCA assembles the sampling locations:
        # 1. Offsets and weights from the BEV query features
        # q_flat: (6, k, 256)  — k = visible queries for this camera
        offsets = self.sampling_offsets(q_flat).view(
            B * num_cams, max_len, self.num_heads, self.num_levels, self.num_points, 2
        )  # offsets (B*nc, L, nh, nl, p, 2): (6, k, 8_heads, 1_level, 8_points, 2) 
        attn = self.attention_weights(q_flat).view(
            B * num_cams, max_len, self.num_heads, self.num_levels * self.num_points
        ).softmax(-1).view(
            B * num_cams, max_len, self.num_heads, self.num_levels, self.num_points
        )

        # 2. Normalise offsets by image feature spatial size  (W, H ordering) so offset=1 means "1 pixel"
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
        ).to(query.device)  # (num_levels, 2) in (W, H) order

        # ref_flat: (B*nc, max_len, num_Z, 2)
        # We distribute num_points across num_Z anchors:  num_points // num_Z per anchor.
        pts_per_z = self.num_points // num_Z  # e.g. 8 // 4 = 2

        ref_exp = ref_flat[:, :, None, None, None, :, :]       # (B*nc, L, 1, 1, 1, num_Z, 2)
        offsets_r = offsets / offset_normalizer[None, None, None, :, None, :]
        # offsets_r: (B*nc, L, num_heads, num_levels, num_points, 2)
        # 3. Reshape offsets so the num_points dim covers pts_per_z x num_Z
        offsets_r = offsets_r.view(
            B * num_cams, max_len, self.num_heads, self.num_levels, pts_per_z, num_Z, 2
        )                                                        # (B*nc, L, nh, nl, p, Z, 2)

        # 4. Assemble sampling locations by adding offsets to reference points
        sampling_locs = ref_exp + offsets_r  # (B*nc, L, 1, 1, 1, num_Z, 2) + (B*nc, L, nh, nl, pts_per_z, num_Z, 2)
        sampling_locs = sampling_locs.view(
            B * num_cams, max_len, self.num_heads, self.num_levels, self.num_points, 2
        )  # (B*nc, L, nh, nl, p, 2) final sampling coordinates in [0,1]

        # Run deformable attention -> (B*num_cams, max_len, C)
        out = ms_deform_attn_core(v_proj, spatial_shapes, sampling_locs, attn)

        # Scatter back into full-resolution slots
        out = out.view(B, num_cams, max_len, self.embed_dim)
        for j in range(B):
            for i, idx in enumerate(cam_indices):
                if len(idx) > 0:
                    slots[j, idx] += out[j, i, :len(idx)]

        # Normalise by number of contributing cameras per query
        count = cam_active.float().permute(1, 2, 0).sum(-1).clamp(min=1.0)  # (B, L)
        slots = slots / count.unsqueeze(-1)  # Further ensures a camera with no visible anchors doesn't dilute other cameras.

        return self.dropout(self.output_proj(slots)) + identity
