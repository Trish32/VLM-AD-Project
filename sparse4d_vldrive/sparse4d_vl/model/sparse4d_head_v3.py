"""
Sparse4DHeadV3 — one decoder stage with DECOUPLED attention (Sparse4D-v3).

Differences from the v2 head (sparse4d_head_v2.py):

Decoupled attention (decouple_attn=True in the reference):
  - attention runs at 2×embed_dims = 512:
      q = cat([instance_feature, anchor_embed], -1)
      k = cat([kv_feature,       kv_embed],     -1)
      v = fc_before(kv_feature)            # Linear 256→512, no bias
      out = fc_after(q + dropout(attn(q, k, v)))   # Linear 512→256, no bias
  - the residual lives at 512 dims INSIDE the attention block (mmcv
    MultiheadAttention adds identity=query); fc_after then maps to 256.
    The result REPLACES instance_feature (no outer residual).
  - 'gnn' self-attention: k = q, v = fc_before(instance_feature)
  - 'temp_gnn' with cache: k = cat(temp), v = fc_before(temp_feature)
  - 'temp_gnn' first frame: k = q, v = q  (the raw 512 cat — reference mmcv
    behaviour when key/value are None: value defaults to key, NOT fc_before)

Anchor encoder: SparseBox3DEncoderV3 (cat mode).
Refinement: SparseBox3DRefinementModuleV3 → (anchor, cls_logits, quality).

fc_before / fc_after are SHARED across all stages in the reference (one pair
on the head); each per-stage instance here is loaded with the same checkpoint
weights, which is numerically identical.

Stage op orders (same flat checkpoint layout as v2, layers 0-38):
  Stage 0:    ['deformable', 'ffn', 'norm', 'refine']
  Stages 1-5: ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import DeformableFeatureAggregation, GraphAttention, AsymmetricFFN
from .detection3d_v3 import SparseBox3DEncoderV3, SparseBox3DRefinementModuleV3


class Sparse4DHeadV3(nn.Module):

    def __init__(
        self,
        embed_dims:       int   = 256,
        num_groups:       int   = 8,
        num_levels:       int   = 4,
        num_cams:         int   = 6,
        num_pts:          int   = 13,
        num_classes:      int   = 10,
        operation_order:  list[str] | None = None,
        ffn_dims:         int   = 1024,
        dropout:          float = 0.1,
        residual_mode:    str   = "cat",
    ):
        super().__init__()
        self.embed_dims      = embed_dims
        self.operation_order = operation_order or [
            'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine'
        ]

        # --- v3 anchor encoder (cat mode, per-component dims) ---
        self.anchor_encoder = SparseBox3DEncoderV3()

        # --- Decoupled attention: 512-dim q/k/v ---
        D2 = embed_dims * 2
        self.gnn      = GraphAttention(D2, num_heads=num_groups, dropout=dropout)
        self.temp_gnn = GraphAttention(D2, num_heads=num_groups, dropout=dropout)
        self.fc_before = nn.Linear(embed_dims, D2, bias=False)
        self.fc_after  = nn.Linear(D2, embed_dims, bias=False)

        ffn_in_dims = embed_dims * 2 if residual_mode == "cat" else embed_dims

        self.deform = DeformableFeatureAggregation(
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=num_cams,
            num_pts=num_pts,
            use_camera_embed=True,
            temporal_fusion=False,
            residual_mode=residual_mode,
        )

        self.ffn = AsymmetricFFN(embed_dims=embed_dims, in_dims=ffn_in_dims,
                                  ffn_dims=ffn_dims, dropout=dropout)

        n_norms = self.operation_order.count('norm')
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(max(n_norms, 1))
        ])

        self.refine = SparseBox3DRefinementModuleV3(embed_dims, num_classes)

    # ------------------------------------------------------------------
    # Decoupled graph attention
    # ------------------------------------------------------------------

    def _graph_attn(
        self,
        attn:      GraphAttention,
        feat:      torch.Tensor,                  # (B, N, D)
        embed:     torch.Tensor,                  # (B, N, D)
        kv_feat:   torch.Tensor | None = None,    # (B, N_kv, D)
        kv_embed:  torch.Tensor | None = None,    # (B, N_kv, D)
        attn_mask: torch.Tensor | None = None,    # (N, N_kv) bool, True = block
    ) -> torch.Tensor:                            # (B, N, D)
        q = torch.cat([feat, embed], dim=-1)                  # (B, N, 2D)
        if kv_feat is not None:
            k = torch.cat([kv_feat, kv_embed], dim=-1)        # (B, N_kv, 2D)=(B,N_kv,512)  Linear 256→512, no bias
            v = self.fc_before(kv_feat)                       # (B, N_kv, 2D)
        else:
            # mmcv defaults: key = query, value = key (raw cat, no fc_before)
            k = q
            v = q
        out, _ = attn.attn(q, k, v, need_weights=False, attn_mask=attn_mask)
        # mmcv MultiheadAttention: identity (= q) + dropout(out), then fc_after
        return self.fc_after(q + attn.dropout(out))  # Linear 512→256, no bias

    # ------------------------------------------------------------------
    # Forward (one stage)
    # ------------------------------------------------------------------

    def forward(
        self,
        instance_feature:      torch.Tensor,         # (B, N, D)
        anchor:                torch.Tensor,         # (B, N, 11)
        feature_maps:          list[torch.Tensor],   # [(B*N_cam, D, H_l, W_l)] × 4
        projection_mat:        torch.Tensor,         # (B, N_cam, 4, 4)
        image_wh:              torch.Tensor,         # (B, N_cam, 2)
        temp_instance_feature: torch.Tensor | None = None,  # (B, N_temp, D)
        time_interval:         float | torch.Tensor = 0.5,
        attn_mask:             torch.Tensor | None = None,   # (N, N) self-attn mask (DN)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (instance_feature, anchor, cls_logits, quality).

        attn_mask (denoising): bool (N, N) self-attention mask isolating DN
        query groups.  Applied to 'gnn' always, and to 'temp_gnn' only when it
        falls back to self-attention (no temporal cache) — when temp_gnn does
        cross-attention to the cache the keys contain no DN queries, so no mask
        is needed.
        """
        # anchor_embed recomputed from the current anchors at stage start
        # (equivalent to the reference recomputing after each refine)
        anchor_embed = self.anchor_encoder(anchor)            # (B, N, D)
        norm_idx = 0
        cls_logits, quality = None, None

        for op in self.operation_order:

            if op == 'gnn':
                # Reference passes value=instance_feature explicitly:
                # k = q (cat), v = fc_before(instance_feature)
                instance_feature = self._graph_attn(
                    self.gnn, instance_feature, anchor_embed,
                    kv_feat=instance_feature, kv_embed=anchor_embed,
                    attn_mask=attn_mask,
                )

            elif op == 'temp_gnn':
                if temp_instance_feature is not None:
                    n_temp = temp_instance_feature.shape[1]
                    instance_feature = self._graph_attn(
                        self.temp_gnn, instance_feature, anchor_embed,
                        kv_feat=temp_instance_feature,
                        kv_embed=anchor_embed[:, :n_temp],
                    )
                else:
                    instance_feature = self._graph_attn(
                        self.temp_gnn, instance_feature, anchor_embed,
                        attn_mask=attn_mask,
                    )

            elif op == 'norm':
                instance_feature = self.norms[norm_idx % len(self.norms)](
                    instance_feature
                )
                norm_idx += 1

            elif op == 'deformable':
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
                anchor, cls_logits, quality = self.refine(
                    instance_feature,
                    anchor,
                    anchor_embed=anchor_embed,
                    time_interval=time_interval,
                )

        return instance_feature, anchor, cls_logits, quality
