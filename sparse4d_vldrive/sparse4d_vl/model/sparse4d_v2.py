"""
Sparse4D v1 and v2 — pure PyTorch, MPS-compatible end-to-end models.

Both models share the same backbone (ResNet-50 + 4-level FPN) and a
Sparse4DHead.  The key differences:

  Sparse4Dv1  (H1 — single frame):
    - 900 persistent queries; no temporal cache
    - Operation order: ['gnn','norm','deformable','norm','ffn','norm','refine'] × 6

  Sparse4Dv1H4  (H4 — 4-frame temporal):
    - Same as v1 but DeformableFeatureAggregation has LinearFusionModule

  Sparse4Dv2  (HInf — continuous temporal cache):
    - 900 queries = 600 temporal (from cache) + 300 fresh priors
    - Stage 0: single-frame deformable (no GNN, no temp)
    - Stages 1-5: temp_gnn + gnn + norm + temporal_deformable + ffn + norm + refine
    - use_camera_embed=True

Device: MPS → CUDA → CPU (auto-selected at construction).
FP32 only; no autocast anywhere.

Image normalisation: ImageNet mean/std (RGB, same as BEVFormer).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .instance_bank    import InstanceBank
from .sparse4d_head_v2 import Sparse4DHead
from .detection3d      import SparseBox3DDecoder
from .sparse4d_base    import _Sparse4DBase
from .depth_head       import DepthHead


# ---------------------------------------------------------------------------
# Sparse4D v1
# ---------------------------------------------------------------------------

class Sparse4Dv1(_Sparse4DBase):
    """
    Sparse4D-v1 single-frame (H1).
    900 persistent queries, no temporal cache.

    Operation order per decoder stage:
      ['gnn', 'norm', 'deformable', 'norm', 'ffn', 'norm', 'refine'] × 6
    """

    V1_OP_ORDER = ['gnn', 'norm', 'deformable', 'norm', 'ffn', 'norm', 'refine']

    def __init__(self, pretrained_backbone: bool = False,
                 num_anchor: int = 900, anchor_path: str | None = None):
        super().__init__(pretrained_backbone)
        self.decoder = SparseBox3DDecoder(num_output=300, score_threshold=0.1)
        D = self.EMBED_DIMS

        self.instance_bank = InstanceBank(
            num_anchor=num_anchor,
            embed_dims=D,
            num_temp_instances=0,   # no temporal cache for v1 H1
        )

        self.head = Sparse4DHead(
            embed_dims=D,
            num_groups=self.NUM_GROUPS,
            num_levels=self.NUM_LEVELS,
            num_cams=self.NUM_CAMS,
            num_pts=self.NUM_PTS,
            num_classes=self.NUM_CLASSES,
            operation_order=self.V1_OP_ORDER,
            num_stages=6,
            use_temporal=False,
            use_camera_embed=False,
            ffn_dims=D * 4,
        )

        self.to(device=self.device, dtype=torch.float32)

    def forward(self, imgs: torch.Tensor, img_metas: dict) -> dict:
        """
        imgs      : (1, N_cam, 3, H, W)  float [0, 255]
        img_metas : dict from NuScenesSparse4DLoader
        Returns   : dict with bboxes, scores, labels, instance_feature
        """
        B = imgs.shape[0]
        imgs = imgs.float().to(self.device)
        imgs = self._normalize(imgs)

        feature_maps, _ = self._extract_features(imgs)
        proj, wh, ego2g = self._meta_to_tensors(img_metas, self.device)

        # Get instances (single-frame: no temporal state)
        instance_feature, anchor, *_ = self.instance_bank.get(B)
        instance_feature = instance_feature.to(self.device)
        anchor           = anchor.to(self.device)

        instance_feature, anchor, cls_logits = self.head(
            instance_feature=instance_feature,
            anchor=anchor,
            feature_maps=feature_maps,
            projection_mat=proj,
            image_wh=wh,
        )

        results = self.decoder(anchor, cls_logits)

        return {
            'detections':       results,
            'instance_feature': instance_feature,
            'anchor':           anchor,
            'cls_logits':       cls_logits,
        }


# ---------------------------------------------------------------------------
# Sparse4D v2
# ---------------------------------------------------------------------------

class Sparse4Dv2(_Sparse4DBase):
    """
    Sparse4D-v2 with continuous temporal instance cache (HInf).

    N = 900 queries = 600 temporal (from last frame's top-confident predictions)
                    + 300 fresh learnable priors.

    Operation order:
      Stage 0 (single-frame):
        ['deformable', 'ffn', 'norm', 'refine']
      Stages 1-5 (temporal):
        ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']
    """

    # Single-frame stage runs once, temporal stages run 5 times
    V2_STAGE0_OP  = ['deformable', 'ffn', 'norm', 'refine']
    V2_TEMP_OP    = ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']

    def __init__(
        self,
        pretrained_backbone: bool = False,
        num_anchor:          int  = 900,
        num_temp_instances:  int  = 600,
        anchor_path:         str | None = None,
        confidence_decay:    float = 0.6,
        with_depth:          bool = False,
        depth_level:         int  = 0,
    ):
        super().__init__(pretrained_backbone)
        self.decoder = SparseBox3DDecoder(num_output=300, score_threshold=0.1)
        D = self.EMBED_DIMS

        # Optional dense-depth supervision branch (training only; not in the
        # checkpoint, so it stays randomly initialised and is ignored at eval).
        self.depth_head = DepthHead(in_channels=D, level=depth_level) if with_depth else None

        self.instance_bank = InstanceBank(
            num_anchor=num_anchor,
            embed_dims=D,
            num_temp_instances=num_temp_instances,
            confidence_decay=confidence_decay,
            anchor_path=anchor_path,
        )

        # Stage 0 — single-frame decoder: one Sparse4DHead with stage-0's op order
        self.head_single = Sparse4DHead(
            embed_dims=D,
            num_groups=self.NUM_GROUPS,
            num_levels=self.NUM_LEVELS,
            num_cams=self.NUM_CAMS,
            num_pts=self.NUM_PTS,
            num_classes=self.NUM_CLASSES,
            operation_order=self.V2_STAGE0_OP,
            num_stages=1,
            use_temporal=False,
            use_camera_embed=True,
            ffn_dims=D * 4,
        )

        # Stages 1-5 — one unique Sparse4DHead per stage (no shared weights in stages)
        # use_temporal=False: DFA has no linear_fusion; temporal cross-attn via temp_gnn only
        self.head_temporal_stages = nn.ModuleList([
            Sparse4DHead(
                embed_dims=D,
                num_groups=self.NUM_GROUPS,
                num_levels=self.NUM_LEVELS,
                num_cams=self.NUM_CAMS,
                num_pts=self.NUM_PTS,
                num_classes=self.NUM_CLASSES,
                operation_order=self.V2_TEMP_OP,
                num_stages=1,
                use_temporal=False,
                use_camera_embed=True,
                ffn_dims=D * 4,
            )
            for _ in range(5)
        ])

        self.to(device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Scene boundary
    # ------------------------------------------------------------------

    def reset_state(self):
        """Call at the start of each new nuScenes scene."""
        self.instance_bank.reset_state()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        imgs:      torch.Tensor,  # (1, N_cam, 3, H, W)  float [0, 255]
        img_metas: dict,
    ) -> dict:
        """
        Returns dict with:
          detections       : list[dict]  — boxes_3d, scores_3d, labels_3d
          instance_feature : (1, N, D)
          anchor           : (1, N, 11)
          cls_logits       : (1, N, C)
        """
        B = imgs.shape[0]
        imgs = imgs.float().to(self.device)
        imgs = self._normalize(imgs)

        # ---- Backbone + FPN ----
        feature_maps, _ = self._extract_features(imgs)

        # ---- Meta tensors ----
        proj, wh, ego2g = self._meta_to_tensors(img_metas, self.device)

        # lidar2ego (lidar sensor → ego vehicle), needed for correct coordinate conversion
        l2e_np = img_metas.get('lidar2ego', None)
        if l2e_np is not None:
            lidar2ego = torch.from_numpy(l2e_np).float().to(self.device)
        else:
            lidar2ego = None

        # ---- Instance bank: fresh priors + cached temporal (separate) ----
        timestamp = img_metas.get('timestamp', None)

        instance_feature, anchor, temp_feat, temp_anchor, time_interval = \
            self.instance_bank.get(
                batch_size=B,
                ego2global=ego2g,
                lidar2ego=lidar2ego,
                timestamp=timestamp,
            )
        instance_feature = instance_feature.to(self.device)
        anchor           = anchor.to(self.device)

        # ---- Stage 0: single-frame decoder on the 900 fresh priors ----
        instance_feature, anchor, cls_logits = self.head_single(
            instance_feature=instance_feature,
            anchor=anchor,
            feature_maps=feature_maps,
            projection_mat=proj,
            image_wh=wh,
            time_interval=time_interval,
        )

        # Per-stage predictions for auxiliary supervision (reference trains
        # every refine output, not just the last one)
        stage_preds = [(anchor, cls_logits)]

        # ---- Merge temporal instances: [cached 600, top-300 fresh] ----
        instance_feature, anchor = self.instance_bank.update(
            instance_feature, anchor, cls_logits
        )

        # ---- Stages 1-5: per-stage temporal decoder (unique weights per stage) ----
        # temp_gnn keys/values are the ORIGINAL cached features all stages
        if temp_feat is not None:
            temp_feat = temp_feat.to(self.device)

        for temporal_head in self.head_temporal_stages:
            instance_feature, anchor, cls_logits = temporal_head(
                instance_feature=instance_feature,
                anchor=anchor,
                feature_maps=feature_maps,
                projection_mat=proj,
                image_wh=wh,
                temp_instance_feature=temp_feat,
                time_interval=time_interval,
            )
            stage_preds.append((anchor, cls_logits))

        # ---- Cache for next frame ----
        self.instance_bank.cache(
            instance_feature=instance_feature,
            anchor=anchor,
            cls_logits=cls_logits,
            ego2global=ego2g,
            lidar2ego=lidar2ego,
            timestamp=timestamp,
        )

        results = self.decoder(anchor, cls_logits)

        out = {
            'detections':       results,        # list[dict]
            'instance_feature': instance_feature,  # (1, N, D)
            'anchor':           anchor,         # (1, N, 11)
            'cls_logits':       cls_logits,     # (1, N, C)
            'stage_preds':      stage_preds,    # [(anchor, cls_logits)] × 6
        }
        # Dense depth prediction (training only)
        if self.depth_head is not None:
            out['depth_pred'] = self.depth_head(feature_maps)  # (B*N_cam, H, W)
        return out

