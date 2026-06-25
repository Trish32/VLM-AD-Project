"""
Temporal Self Attention (TSA) — pure PyTorch, MPS-compatible.

Faithfully reproduces the BEVFormer TSA logic:
  - Concatenates prev_bev and current bev_query as the key/value sequence.
  - Generates deformable offsets on the 2D BEV plane from the concat query.
  - Samples both bev_queue slots via ms_deform_attn_core (F.grid_sample). TSA does not use nn.MultiheadAttention.
  - Averages the two sampled outputs and applies the residual connection.

Reference shapes (batch-first):
  query     : (B, bev_h*bev_w, C)
  prev_bev  : (B, bev_h*bev_w, C)  or None -> treated as identity
  ref_2d    : (B*2, bev_h*bev_w, 1, 2)     normalised [0,1] BEV grid
"""

import math
import torch
import torch.nn as nn
from .deform_attn import ms_deform_attn_core


class TemporalSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        num_bev_queue: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim     = embed_dim
        self.num_heads     = num_heads
        self.num_levels    = num_levels
        self.num_points    = num_points
        self.num_bev_queue = num_bev_queue
        self.head_dim      = embed_dim // num_heads

        # Offset / weight networks receive [prev_bev || query] (2*C input)
        self.sampling_offsets = nn.Linear(
            embed_dim * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            embed_dim * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points,
        )
        self.value_proj  = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout     = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2
        ).repeat(1, self.num_levels * self.num_bev_queue, self.num_points, 1)
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
        query:          torch.Tensor,           # (B, L, C)
        prev_bev:       torch.Tensor | None,    # (B, L, C) or None
        ref_2d:         torch.Tensor,           # (B*2, L, 1, 2)
        spatial_shapes: torch.Tensor,           # (1, 2)  [(bev_h, bev_w)]
        level_start_index: torch.Tensor,        # (1,)    [0]
        query_pos:      torch.Tensor | None = None,  # (1, L, C) BEV positional encoding
    ) -> torch.Tensor:                          # (B, L, C)

        B, L, _ = query.shape
        identity = query  # residual uses original query without pos

        if prev_bev is None:
            prev_bev = query

        # V: [prev_bev, query] with NO positional encoding
        value = torch.cat([prev_bev, query], dim=0)  # (B*2, L, C) concat along batch dimension for efficient processing of both bev_queue entries together

        # Add BEV positional encoding to query before computing offsets/weights
        # (matches official TSA: query = query + query_pos inside the module)
        if query_pos is not None:
            query = query + query_pos

        # Q: Input for offset/weight generation: [prev_bev || query+pos] -> (B, L, 2C)
        q_inp = torch.cat([value[:B], query], dim=-1)

        # V: Project value for all bev_queue entries
        v = self.value_proj(value)                                        # (B*2, L, C)
        v = v.view(B * self.num_bev_queue, L, self.num_heads, self.head_dim)

        # Sampling offsets: (B, L, num_heads, num_bev_queue, num_levels, num_points, 2)
        offsets = self.sampling_offsets(q_inp).view(
            B, L, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2
        )

        # Attention weights: softmax over (num_levels * num_points) per bev_queue entry
        attn = self.attention_weights(q_inp).view(
            B, L, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points
        ).softmax(-1).view(
            B, L, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points
        )

        # Normalise offsets by spatial shape (W, H ordering)
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
        ).to(query.device)                                               # (1, 2)

        # ref_2d: (B*2, L, 1, 2) -> broadcast to (B, L, 1, 1, 1, 1, 2) for subtraction
        ref = ref_2d.view(B, self.num_bev_queue, L, 1, 1, 1, 2)        # (B,2,L,1,1,1,2)
        ref = ref.permute(0, 2, 3, 1, 4, 5, 6)                         # (B,L,1,2,1,1,2)

        # Normalised offsets
        offsets_n = offsets / offset_normalizer[None, None, None, None, :, None, :]

        # Sampling locations: Add offsets to reference points (B, L, num_heads, num_bev_queue, num_levels, num_points, 2)
        sampling_locs = ref + offsets_n

        # Sampling locations: Reshape to (B*2, L, num_heads, num_levels, num_points, 2)
        sampling_locs = sampling_locs.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            B * self.num_bev_queue, L, self.num_heads, self.num_levels, self.num_points, 2
        )
        attn = attn.permute(0, 3, 1, 2, 4, 5).reshape(
            B * self.num_bev_queue, L, self.num_heads, self.num_levels, self.num_points
        )

        # Deformable attention -> (B*2, L, C)
        # Samples v (the BEV feature map) at sampling_locs → (B*2, L, C)
        out = ms_deform_attn_core(v, spatial_shapes, sampling_locs, attn)

        # Fuse the two BEV-queue slots: mean over bev_queue -> (B, L, C)
        out = out.view(B, self.num_bev_queue, L, self.embed_dim).mean(1)  # (B, L, C) — equal-weight average

        return self.dropout(self.output_proj(out)) + identity
