"""
Sparse4DHead — iterative decoder for sparse 3-D object queries.

Supports both Sparse4D-v1 and v2 via the `operation_order` list.

Supported operation tokens
--------------------------
  'gnn'        : self-attention over N queries (query_pos = anchor_embed)
  'temp_gnn'   : cross-attention current queries ← cached temporal features
                 (key_pos = anchor_embed of the temporal slots); falls back to
                 self-attention with its own weights when no temporal data
  'norm'       : LayerNorm applied to instance_feature
  'deformable' : DeformableFeatureAggregation
  'ffn'        : AsymmetricFFN feed-forward network
  'refine'     : SparseBox3DRefinementModule → update anchor + emit cls_logits

Matches the reference Sparse4D semantics:
  - anchor_embed = anchor_encoder(anchor), recomputed at the START of each
    stage (equivalent to the reference recomputing it after each refine)
  - GNN attention adds anchor_embed as positional encoding on query/key
    (mmcv MultiheadAttention convention; value gets no positional encoding)
  - temp_gnn keys/values are the cached temporal features ONLY, with
    key_pos = anchor_embed[:, :N_temp]  (temporal instances occupy the
    first N_temp slots after InstanceBank.update())

Sparse4D-v2 stage layout (each stage = one Sparse4DHead with unique weights):
  Stage 0 (single-frame): ['deformable', 'ffn', 'norm', 'refine']
  Stages 1-5 (temporal) : ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import (
    DeformableFeatureAggregation,
    GraphAttention,
    AsymmetricFFN,
)
from .detection3d import SparseBox3DEncoder, SparseBox3DRefinementModule


class Sparse4DHead(nn.Module):
    """
    Parameters
    ----------
    embed_dims      : feature dimension (256)
    num_groups      : groups in DeformableFeatureAggregation (8)
    num_levels      : FPN levels (4)
    num_cams        : camera count (6)
    num_pts         : keypoints per anchor (13)
    num_classes     : detection classes (10)
    operation_order : list of operation tokens for one decoder stage
    num_stages      : how many times operation_order is repeated
    use_camera_embed: add per-camera embedding (v2)
    ffn_dims        : hidden dim of FFN (1024)
    dropout         : attention / FFN dropout
    """

    def __init__(
        self,
        embed_dims:       int   = 256,
        num_groups:       int   = 8,
        num_levels:       int   = 4,
        num_cams:         int   = 6,
        num_pts:          int   = 7,
        num_classes:      int   = 10,
        operation_order:  list[str] | None = None,
        num_stages:       int   = 6,
        use_temporal:     bool  = False,
        use_camera_embed: bool  = False,
        ffn_dims:         int   = 1024,
        dropout:          float = 0.1,
        residual_mode:    str   = "cat",
    ):
        super().__init__()
        self.embed_dims     = embed_dims
        self.operation_order = operation_order or [
            'gnn', 'norm', 'deformable', 'norm', 'ffn', 'norm', 'refine'
        ]
        self.num_stages     = num_stages

        # --- Anchor box positional encoder ---
        self.anchor_encoder = SparseBox3DEncoder(embed_dims)

        # --- GNN: standard self-attention ---
        self.gnn = GraphAttention(embed_dims, num_heads=8, dropout=dropout)

        # --- GNN for temporal cross-attention (keys/values = temporal queries) ---
        self.temp_gnn = GraphAttention(embed_dims, num_heads=8, dropout=dropout)

        # When residual_mode="cat" the DFA output is 2×embed_dims wide
        ffn_in_dims = embed_dims * 2 if residual_mode == "cat" else embed_dims

        # --- Deformable feature aggregation ---
        self.deform = DeformableFeatureAggregation(
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=num_cams,
            num_pts=num_pts,
            use_camera_embed=use_camera_embed,
            temporal_fusion=False,
            residual_mode=residual_mode,
        )

        # --- FFN (in_dims=512 when DFA uses residual_mode="cat") ---
        self.ffn = AsymmetricFFN(embed_dims=embed_dims, in_dims=ffn_in_dims,
                                  ffn_dims=ffn_dims, dropout=dropout)

        # --- LayerNorms (one per 'norm' token in operation_order) ---
        n_norms = self.operation_order.count('norm')
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(max(n_norms, 1))
        ])

        # --- Box refinement ---
        self.refine = SparseBox3DRefinementModule(embed_dims, num_classes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        instance_feature:      torch.Tensor,         # (B, N, D)
        anchor:                torch.Tensor,         # (B, N, 11)
        feature_maps:          list[torch.Tensor],   # [(B*N_cam, D, H_l, W_l)] × 4
        projection_mat:        torch.Tensor,         # (B, N_cam, 4, 4)
        image_wh:              torch.Tensor,         # (B, N_cam, 2)
        temp_instance_feature: torch.Tensor | None = None,  # (B, N_temp, D) cached
        time_interval:         float | torch.Tensor = 0.5,  # current frame Δt
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        instance_feature : (B, N, D)    decoder output
        anchor           : (B, N, 11)   refined anchor boxes
        cls_logits       : (B, N, C)    from the LAST 'refine' step
        """
        cls_logits = None

        for stage in range(self.num_stages):
            # Recompute anchor positional embedding from the current anchors at the start of every stage
            # (reference codebase recomputes after each refine — same thing)
            anchor_embed = self.anchor_encoder(anchor)            # (B, N, D)
            norm_idx = 0

            for op in self.operation_order:
                """
                Every op is a residual add (instance_feature = instance_feature + op(...)) 
                except norm, deformable (which carries its own cat-residual internally), and refine
                """
                # self-attention with query_pos=anchor_embed. 
                # The positional embedding is added to query and key but not value — that's the mmcv MultiheadAttention convention
                if op == 'gnn':
                    instance_feature = instance_feature + self.gnn(
                        instance_feature, query_pos=anchor_embed
                    )

                elif op == 'temp_gnn':  # cross-attention
                    if temp_instance_feature is not None:
                        # Temporal instances/600 cached instances occupy slots [0, N_temp) after
                        # InstanceBank.update(), so their positional embedding
                        # is the first N_temp rows of anchor_embed.
                        n_temp  = temp_instance_feature.shape[1]
                        key_pos = anchor_embed[:, :n_temp]
                        instance_feature = instance_feature + self.temp_gnn(
                            instance_feature,
                            key_value=temp_instance_feature,
                            query_pos=anchor_embed,
                            key_pos=key_pos,
                        )
                    else:
                        # First frame (no cache): self-attention with temp_gnn's own weights
                        instance_feature = instance_feature + self.temp_gnn(
                            instance_feature, query_pos=anchor_embed
                        ) # note it's a separate module from gnn, so it has its own checkpoint weights even in this degenerate mode.

                elif op == 'norm':
                    instance_feature = self.norms[norm_idx % len(self.norms)](
                        instance_feature
                    )
                    norm_idx += 1

                elif op == 'deformable':
                    # DFA with residual_mode="cat" includes the skip connection
                    # internally (cat of aggregated + input)
                    instance_feature = self.deform(
                        instance_feature = instance_feature,
                        anchor           = anchor,
                        anchor_embed     = anchor_embed,
                        feature_maps     = feature_maps,
                        projection_mat   = projection_mat,
                        image_wh         = image_wh,
                    )

                elif op == 'ffn':
                    instance_feature = self.ffn(instance_feature)

                elif op == 'refine':
                    anchor, cls_logits = self.refine(
                        instance_feature,
                        anchor,
                        anchor_embed=anchor_embed,
                        time_interval=time_interval,
                    )

        if cls_logits is None:
            anchor_embed = self.anchor_encoder(anchor)
            anchor, cls_logits = self.refine(
                instance_feature, anchor,
                anchor_embed=anchor_embed, time_interval=time_interval,
            )

        return instance_feature, anchor, cls_logits
