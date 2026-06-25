"""
Sparse4D-v3 — pure PyTorch, MPS-compatible end-to-end model.

Same overall recipe as Sparse4Dv2 (sparse4d_v2.py):
  - ResNet-50 + 4-level FPN backbone (shared)
  - 900 queries = 600 temporal cache + 300 fresh priors
  - Stage 0 single-frame on fresh priors, InstanceBank.update() merge,
    stages 1-5 temporal (each stage has UNIQUE weights from the checkpoint)

v3 additions:
  - decoupled attention (512-dim q/k/v with shared fc_before / fc_after)
  - v3 anchor encoder (per-component cat: pos 128 + size 32 + yaw 32 + vel 64)
  - quality estimation: refine emits (centerness, yawness); decoder re-ranks
    final scores by sigmoid(cls) * sigmoid(centerness)

Checkpoint: sparse4dv3_r50.pth (39 flat head.layers, same indices as v2).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .instance_bank    import InstanceBank
from .sparse4d_head_v3 import Sparse4DHeadV3
from .detection3d_v3   import SparseBox3DDecoderV3
from .sparse4d_base    import _Sparse4DBase
from .denoising        import generate_dn_groups, build_dn_attn_mask
from .depth_head       import DepthHead
from .motion_head      import TrajectoryHead
from .motion_planning  import AgentMotionHead, EgoPlanner, MapEncoder


class Sparse4Dv3(_Sparse4DBase):
    """
    Operation order (flat checkpoint layers 0-38, identical layout to v2):
      Stage 0 (single-frame): ['deformable', 'ffn', 'norm', 'refine']
      Stages 1-5 (temporal) : ['temp_gnn', 'gnn', 'norm', 'deformable',
                               'ffn', 'norm', 'refine']
    """

    V3_STAGE0_OP = ['deformable', 'ffn', 'norm', 'refine']
    V3_TEMP_OP   = ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']

    def __init__(
        self,
        pretrained_backbone: bool = False,
        num_anchor:          int  = 900,
        num_temp_instances:  int  = 600,
        anchor_path:         str | None = None,
        confidence_decay:    float = 0.6,
        with_depth:          bool = False,
        depth_level:         int  = 0,
        with_motion:         bool = False,
        motion_modes:        int  = 6,
        motion_steps:        int  = 12,
        with_planning:       bool = False,
        ego_steps:           int  = 6,
        with_map:            bool = False,
    ):
        super().__init__(pretrained_backbone)
        D = self.EMBED_DIMS

        self.decoder = SparseBox3DDecoderV3(num_output=300, score_threshold=0.05)

        # Optional dense-depth branch (training only; not in the checkpoint).
        self.depth_head = DepthHead(in_channels=D, level=depth_level) if with_depth else None

        # Optional QCNet-style motion head (Laplace multimodal).
        self.motion_head = (
            TrajectoryHead(embed_dims=D, num_modes=motion_modes, future_steps=motion_steps)
            if with_motion else None
        )

        # SparseDrive motion planner: anchored agent motion + ego planner.
        if with_planning:
            self.agent_motion = AgentMotionHead(
                embed_dims=D, num_modes=motion_modes, future_steps=motion_steps,
                with_map=with_map)
            self.ego_planner = EgoPlanner(
                embed_dims=D, ego_modes=3, ego_steps=ego_steps, with_map=with_map)
            self.map_encoder = MapEncoder(embed_dims=D) if with_map else None
        else:
            self.agent_motion = None
            self.ego_planner = None
            self.map_encoder = None

        self.instance_bank = InstanceBank(
            num_anchor=num_anchor,
            embed_dims=D,
            num_temp_instances=num_temp_instances,
            confidence_decay=confidence_decay,
            anchor_path=anchor_path,
            v3_yaw_projection_bug=True,   # checkpoint trained with the ref bug
        )

        # Stage 0 — single-frame decoder
        self.head_single = Sparse4DHeadV3(
            embed_dims=D,
            num_groups=self.NUM_GROUPS,
            num_levels=self.NUM_LEVELS,
            num_cams=self.NUM_CAMS,
            num_pts=self.NUM_PTS,
            num_classes=self.NUM_CLASSES,
            operation_order=self.V3_STAGE0_OP,
            ffn_dims=D * 4,
        )

        # Stages 1-5 — one unique head per stage
        self.head_temporal_stages = nn.ModuleList([
            Sparse4DHeadV3(
                embed_dims=D,
                num_groups=self.NUM_GROUPS,
                num_levels=self.NUM_LEVELS,
                num_cams=self.NUM_CAMS,
                num_pts=self.NUM_PTS,
                num_classes=self.NUM_CLASSES,
                operation_order=self.V3_TEMP_OP,
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

    def _encode_map(self, img_metas):
        """Build map tokens from HD-map polylines in img_metas (if present)."""
        if self.map_encoder is None or 'map_pts' not in img_metas:
            return None, None
        pts = torch.as_tensor(img_metas['map_pts'], dtype=torch.float32,
                              device=self.device).unsqueeze(0)        # (1,M,P,2)
        mask = torch.as_tensor(img_metas['map_mask'], dtype=torch.bool,
                               device=self.device).unsqueeze(0)       # (1,M)
        return self.map_encoder(pts), mask

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
          quality          : (1, N, 2)   [centerness, yawness]
          stage_preds      : [(anchor, cls_logits)] × 6  (for fine-tuning)
        """
        B = imgs.shape[0]
        imgs = imgs.float().to(self.device)
        imgs = self._normalize(imgs)

        # ---- Backbone + FPN ----
        feature_maps, _ = self._extract_features(imgs)

        # ---- Meta tensors ----
        proj, wh, ego2g = self._meta_to_tensors(img_metas, self.device)

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
        instance_feature, anchor, cls_logits, quality = self.head_single(
            instance_feature=instance_feature,
            anchor=anchor,
            feature_maps=feature_maps,
            projection_mat=proj,
            image_wh=wh,
            time_interval=time_interval,
        )

        stage_preds = [(anchor, cls_logits, quality)]

        # ---- Merge temporal instances: [cached 600, top-300 fresh] ----
        instance_feature, anchor = self.instance_bank.update(
            instance_feature, anchor, cls_logits
        )

        # ---- Stages 1-5: per-stage temporal decoder ----
        if temp_feat is not None:
            temp_feat = temp_feat.to(self.device)

        for temporal_head in self.head_temporal_stages:
            instance_feature, anchor, cls_logits, quality = temporal_head(
                instance_feature=instance_feature,
                anchor=anchor,
                feature_maps=feature_maps,
                projection_mat=proj,
                image_wh=wh,
                temp_instance_feature=temp_feat,
                time_interval=time_interval,
            )
            stage_preds.append((anchor, cls_logits, quality))

        # ---- Cache for next frame ----
        self.instance_bank.cache(
            instance_feature=instance_feature,
            anchor=anchor,
            cls_logits=cls_logits,
            ego2global=ego2g,
            lidar2ego=lidar2ego,
            timestamp=timestamp,
        )

        # ---- Tracking: assign/propagate track IDs (inference only) ----
        instance_id = None
        if not self.training:
            instance_id = self.instance_bank.get_instance_id(
                cls_logits, threshold=self.decoder.score_threshold)

        # ---- Map tokens (shared by motion + planning) ----
        map_feat, map_mask = self._encode_map(img_metas)

        # ---- Motion forecasting: multi-modal future trajectories ----
        trajectories = traj_mode_logits = None
        if self.motion_head is not None:
            trajectories, traj_mode_logits = self.motion_head(instance_feature)
        elif self.agent_motion is not None:                 # SparseDrive motion
            trajectories, traj_mode_logits = self.agent_motion(
                instance_feature, anchor, map_feature=map_feat, map_mask=map_mask)

        results = self.decoder(anchor, cls_logits, quality, instance_id=instance_id,
                               trajectories=trajectories, mode_logits=traj_mode_logits)

        out = {
            'detections':       results,
            'instance_feature': instance_feature,
            'anchor':           anchor,
            'cls_logits':       cls_logits,
            'quality':          quality,
            'stage_preds':      stage_preds,
        }
        if trajectories is not None:
            out['trajectories'] = trajectories            # (B, N, K, T, 2[+2])
            out['traj_mode_logits'] = traj_mode_logits    # (B, N, K)
        # ---- Ego planning (SparseDrive) ----
        if self.ego_planner is not None:
            ego_traj, ego_logits = self.ego_planner(
                instance_feature, map_feature=map_feat, map_mask=map_mask)
            out['ego_traj'] = ego_traj                    # (B, 3, Te, 2)
            out['ego_logits'] = ego_logits                # (B, 3)
            # attach the command-best ego plan to the result dict
            out['detections'][0]['ego_traj'] = ego_traj[0]
            out['detections'][0]['ego_logits'] = ego_logits[0]
        return out

    # ------------------------------------------------------------------
    # Training forward with Temporal Instance Denoising (DN)
    # ------------------------------------------------------------------

    def forward_train(
        self,
        imgs:       torch.Tensor,   # (1, N_cam, 3, H, W)
        img_metas:  dict,
        gt_boxes:   torch.Tensor,   # (M, 11)  lidar-frame anchor format
        gt_labels:  torch.Tensor,   # (M,)
        dn_groups:  int = 5,
    ) -> dict:
        """
        Mirror of forward() but injects DN query groups (see model/denoising.py).

        DN queries are appended AFTER the 900 regular queries and isolated by an
        attention mask, so the regular path — including InstanceBank.update()/
        cache() and the decoder — is identical to inference.  DN groups are
        rebuilt each frame (single-frame denoising) and never enter the temporal
        cache; the regular queries alone are cached for the next frame.

        Returns (in addition to the normal keys):
          dn_stage_preds : [(dn_anchor (P,11), dn_logits (P,C))] × 6
          dn_labels      : (P,)   target class per DN query
          dn_gt_boxes    : (P,11) clean box per DN query
        """
        B = imgs.shape[0]
        imgs = self._normalize(imgs.float().to(self.device))
        feature_maps, _ = self._extract_features(imgs)
        proj, wh, ego2g = self._meta_to_tensors(img_metas, self.device)

        l2e_np = img_metas.get('lidar2ego', None)
        lidar2ego = (torch.from_numpy(l2e_np).float().to(self.device)
                     if l2e_np is not None else None)
        timestamp = img_metas.get('timestamp', None)

        instance_feature, anchor, temp_feat, temp_anchor, time_interval = \
            self.instance_bank.get(batch_size=B, ego2global=ego2g,
                                   lidar2ego=lidar2ego, timestamp=timestamp)
        instance_feature = instance_feature.to(self.device)
        anchor           = anchor.to(self.device)
        num_reg          = anchor.shape[1]
        if temp_feat is not None:
            temp_feat = temp_feat.to(self.device)

        # ---- Build DN groups ----
        gt_boxes  = gt_boxes.to(self.device)
        gt_labels = gt_labels.to(self.device)
        dn_anchor, dn_labels, dn_gt_boxes, M = generate_dn_groups(
            gt_boxes, gt_labels, num_groups=dn_groups, num_classes=self.NUM_CLASSES)
        P = dn_anchor.shape[0]
        dn_active = P > 0

        if dn_active:
            dn_anchor = dn_anchor.unsqueeze(0)                          # (1, P, 11)
            dn_feat   = torch.zeros(B, P, self.EMBED_DIMS, device=self.device)
            attn_mask = build_dn_attn_mask(num_reg, dn_groups, M, self.device)
        else:
            attn_mask = None

        def _cat(reg, dn):
            return torch.cat([reg, dn], dim=1) if dn_active else reg

        # ---- Stage 0 (no attention; DN runs independently) ----
        feat0   = _cat(instance_feature, dn_feat) if dn_active else instance_feature
        anc0    = _cat(anchor, dn_anchor)         if dn_active else anchor
        feat0, anc0, cls0, q0 = self.head_single(
            instance_feature=feat0, anchor=anc0, feature_maps=feature_maps,
            projection_mat=proj, image_wh=wh, time_interval=time_interval)

        instance_feature, dn_feat   = (feat0[:, :num_reg], feat0[:, num_reg:]) if dn_active else (feat0, None)
        anchor,           dn_anchor = (anc0[:, :num_reg],  anc0[:, num_reg:])  if dn_active else (anc0, None)
        cls_logits,       dn_cls    = (cls0[:, :num_reg],  cls0[:, num_reg:])  if dn_active else (cls0, None)
        quality,          dn_qual   = (q0[:, :num_reg],    q0[:, num_reg:])    if dn_active else (q0, None)

        stage_preds    = [(anchor, cls_logits, quality)]
        dn_stage_preds = [(dn_anchor[0], dn_cls[0], dn_qual[0])] if dn_active else []

        # ---- Merge temporal instances (regular queries only) ----
        instance_feature, anchor = self.instance_bank.update(
            instance_feature, anchor, cls_logits)

        # ---- Temporal stages 1-5 ----
        for temporal_head in self.head_temporal_stages:
            feat_in = _cat(instance_feature, dn_feat)
            anc_in  = _cat(anchor, dn_anchor)
            feat_o, anc_o, cls_o, q_o = temporal_head(
                instance_feature=feat_in, anchor=anc_in, feature_maps=feature_maps,
                projection_mat=proj, image_wh=wh, temp_instance_feature=temp_feat,
                time_interval=time_interval, attn_mask=attn_mask)

            instance_feature = feat_o[:, :num_reg] if dn_active else feat_o
            anchor           = anc_o[:, :num_reg]  if dn_active else anc_o
            cls_logits       = cls_o[:, :num_reg]  if dn_active else cls_o
            quality          = q_o[:, :num_reg]    if dn_active else q_o
            stage_preds.append((anchor, cls_logits, quality))
            if dn_active:
                dn_feat, dn_anchor = feat_o[:, num_reg:], anc_o[:, num_reg:]
                dn_stage_preds.append(
                    (anc_o[:, num_reg:][0], cls_o[:, num_reg:][0], q_o[:, num_reg:][0]))

        # ---- Cache regular queries for next frame ----
        self.instance_bank.cache(
            instance_feature=instance_feature, anchor=anchor, cls_logits=cls_logits,
            ego2global=ego2g, lidar2ego=lidar2ego, timestamp=timestamp)

        results = self.decoder(anchor, cls_logits, quality)
        out = {
            'detections':     results,
            'stage_preds':    stage_preds,
            'dn_stage_preds': dn_stage_preds,
            'dn_labels':      dn_labels,
            'dn_gt_boxes':    dn_gt_boxes,
            'final_anchor':   anchor,       # for the motion-loss Hungarian match
            'final_cls':      cls_logits,
        }
        if self.depth_head is not None:
            out['depth_pred'] = self.depth_head(feature_maps)
        if self.motion_head is not None:
            # QCNet-Laplace trajectories from the final-stage instance feature
            traj, mode_logits = self.motion_head(instance_feature)
            out['trajectories'] = traj
            out['traj_mode_logits'] = mode_logits
        map_feat, map_mask = self._encode_map(img_metas)
        if self.agent_motion is not None:                   # SparseDrive motion
            traj, mode_logits = self.agent_motion(
                instance_feature, anchor, map_feature=map_feat, map_mask=map_mask)
            out['trajectories'] = traj
            out['traj_mode_logits'] = mode_logits
        if self.ego_planner is not None:                    # SparseDrive planning
            ego_traj, ego_logits = self.ego_planner(
                instance_feature, map_feature=map_feat, map_mask=map_mask)
            out['ego_traj'] = ego_traj
            out['ego_logits'] = ego_logits
        return out
