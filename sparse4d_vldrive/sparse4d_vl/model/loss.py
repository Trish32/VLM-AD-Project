"""
Loss functions for Sparse4D fine-tuning — pure PyTorch, MPS-compatible.

Sparse4DLoss
  ├── FocalLoss          (multi-label classification with sigmoid)
  └── Hungarian matcher  (scipy linear_sum_assignment, runs on CPU)

Box regression uses L1 loss in log-space anchor format:
  [x, y, z,  log_w, log_l, log_h,  sin_yaw, cos_yaw,  vx, vy, vz]

This matches the model's anchor representation exactly (detection3d.py),
so no extra encoding/decoding is needed during training.

Usage:
    criterion = Sparse4DLoss()
    loss = criterion(
        pred_anchor  = anchor[0],      # (900, 11)  from model forward
        pred_logits  = cls_logits[0],  # (900, 10)
        gt_boxes     = gt_boxes,       # (M, 11)
        gt_labels    = gt_labels,      # (M,)
    )
    loss.backward()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Focal loss (sigmoid multi-label)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Sigmoid focal loss for multi-label classification.

    Parameters
    ----------
    alpha : float  — weighting factor for the positive class
    gamma : float  — focusing exponent (0 = standard BCE)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred_logits: torch.Tensor,   # (N, C)
        targets:     torch.Tensor,   # (N, C) float binary targets
    ) -> torch.Tensor:
        prob = pred_logits.sigmoid()
        ce   = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        # pt: probability of the "correct" class
        pt   = targets * prob + (1.0 - targets) * (1.0 - prob)
        # (1-pt)^gamma factor down-weights easy negatives so they don't drown the signal.
        loss = self.alpha * (1.0 - pt).pow(self.gamma) * ce
        return loss.sum()


# ---------------------------------------------------------------------------
# Main loss
# ---------------------------------------------------------------------------

class Sparse4DLoss(nn.Module):
    """
    Combined classification + regression loss for Sparse4D fine-tuning.

    Matching strategy: Hungarian assignment minimising a joint cost of
    focal classification cost + L1 box position cost.

    Parameters
    ----------
    num_classes   : int    number of object classes (10 for nuScenes)
    weight_cls    : float  multiplier on classification loss
    weight_reg    : float  multiplier on regression loss
    weight_vel    : float  extra weight on velocity dims (vx, vy) within reg
    focal_alpha   : float  FocalLoss alpha
    focal_gamma   : float  FocalLoss gamma
    cost_cls      : float  cost weight for classification in Hungarian matching
    cost_reg      : float  cost weight for L1 position in Hungarian matching
    """

    def __init__(
        self,
        num_classes:  int   = 10,
        weight_cls:   float = 2.0,
        weight_reg:   float = 0.25,
        weight_vel:   float = 0.2,
        weight_cns:   float = 1.0,
        weight_yns:   float = 1.0,
        focal_alpha:  float = 0.25,
        focal_gamma:  float = 2.0,
        cost_cls:     float = 1.0,
        cost_reg:     float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_cls  = weight_cls
        self.weight_reg  = weight_reg
        self.weight_vel  = weight_vel
        self.weight_cns  = weight_cns
        self.weight_yns  = weight_yns
        self.cost_cls    = cost_cls
        self.cost_reg    = cost_reg
        self.focal       = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    # ------------------------------------------------------------------
    # Hungarian matcher (CPU, no gradient)
    # ------------------------------------------------------------------
    # Matching is a discrete assignment decision, not part of the differentiable graph 
    # — gradients flow through the loss computed after matching, not through the choice of pairs
    @torch.no_grad()
    def _match(
        self,
        pred_anchor:  torch.Tensor,   # (N_q, 11)
        pred_logits:  torch.Tensor,   # (N_q, C)
        gt_boxes:     torch.Tensor,   # (M, 11)
        gt_labels:    torch.Tensor,   # (M,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (pred_indices, gt_indices) — 1-D LongTensors of matched pairs.
        Both are on CPU.
        Hungarian matching resolves the 900-queries-vs-N-objects ambiguity per stage with a gradient-free cost
        """
        M = gt_labels.shape[0]
        if M == 0:
            empty = torch.zeros(0, dtype=torch.long)
            return empty, empty

        # --- Classification cost: negative probability of GT class ---
        # sigmoid prob for each query over GT classes: (N_q, M)
        prob      = pred_logits.sigmoid().detach()            # (N_q, C)
        cls_cost  = -prob[:, gt_labels.cpu()]                 # (N_q, M)

        # --- Regression cost: L1 distance on (x, y, z) position only ---
        # Position match is most discriminative; size/yaw are left to be refined once matched
        reg_cost = torch.cdist(
            pred_anchor[:, :3].detach().cpu().float(),
            gt_boxes[:, :3].cpu().float(),
            p=1,
        )   # (N_q, M)

        cost = (self.cost_cls * cls_cost.cpu() +
                self.cost_reg * reg_cost)         # (N_q, M) on CPU

        pred_idx, gt_idx = linear_sum_assignment(cost.numpy())
        return (
            torch.tensor(pred_idx, dtype=torch.long),
            torch.tensor(gt_idx,   dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @staticmethod
    def quality_loss(
        matched_pred:    torch.Tensor,  # (K, 11)  matched predicted anchors
        matched_gt:      torch.Tensor,  # (K, 11)  matched GT boxes
        matched_quality: torch.Tensor,  # (K, 2)   [centerness, yawness] logits
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        v3 quality-branch targets, matching the official Sparse4Dv3 loss
        (projects/.../detection3d/losses.py):

          centerness target = exp(-||pred_xyz - gt_xyz||_2)   (soft, in (0,1])
          yawness    target = 1[ cos_sim(pred[sin,cos], gt[sin,cos]) > 0 ]

        Both supervised with BCE-with-logits (the official uses
        CrossEntropyLoss(use_sigmoid=True) for cns and GaussianFocalLoss for yns;
        BCE on the binary/soft targets is the transparent pure-PyTorch
        equivalent).  Returns (cns_loss, yns_loss).
        """
        cns = matched_quality[:, 0]
        yns = matched_quality[:, 1]
        cns_target = torch.exp(
            -torch.norm(matched_pred[:, :3] - matched_gt[:, :3], p=2, dim=-1))
        cns_loss = F.binary_cross_entropy_with_logits(cns, cns_target)
        yns_target = (F.cosine_similarity(
            matched_pred[:, 6:8], matched_gt[:, 6:8], dim=-1) > 0).float()
        yns_loss = F.binary_cross_entropy_with_logits(yns, yns_target)
        return cns_loss, yns_loss

    def forward(
        self,
        pred_anchor:  torch.Tensor,   # (N_q, 11)  raw anchor from model head
        pred_logits:  torch.Tensor,   # (N_q, C)   raw class logits
        gt_boxes:     torch.Tensor,   # (M, 11)    log-space anchor format
        gt_labels:    torch.Tensor,   # (M,)       long
        pred_quality: torch.Tensor | None = None,  # (N_q, 2) v3 [cns, yns] logits
    ) -> torch.Tensor:
        """
        Computes the combined focal + regression loss for one frame.

        Both pred tensors and GT tensors must be on the same device.
        Returns a scalar loss tensor with gradient attached to the
        pred tensors.
        
        Each stage matches independently. forward runs its own Hungarian assignment per stage
        because a query's box changes between stages, so the optimal prediction↔GT pairing can differ stage to stage.
        """
        device  = pred_logits.device
        N_q     = pred_logits.shape[0]
        M       = gt_labels.shape[0]

        # ---- Classification loss ----
        # Build (N_q, C) binary target matrix; matched queries get target=1
        #  everything else 0 — so unmatched queries are trained to predict background (all-zero). 
        cls_targets = torch.zeros(N_q, self.num_classes,
                                  dtype=torch.float32, device=device)

        pred_idx, gt_idx = self._match(pred_anchor, pred_logits,
                                       gt_boxes, gt_labels)

        num_matched = len(pred_idx)

        if num_matched > 0:
            cls_targets[pred_idx.to(device),
                        gt_labels[gt_idx].to(device)] = 1.0

        # Normalise by number of GT objects (min 1 to avoid div-by-zero)
        # so the magnitude is stable across frames with different object counts.
        normaliser = max(M, 1)
        loss_cls   = self.focal(pred_logits, cls_targets) / normaliser

        # ---- L1 Regression loss (matched queries only) ----
        if num_matched > 0:
            matched_pred = pred_anchor[pred_idx.to(device)]        # (K, 11)
            matched_gt   = gt_boxes[gt_idx.to(device)]             # (K, 11)

            # Slots 0–7: Position (x, y, z) + log-size (log_w, log_l, log_h)
            # + heading (sin, cos) at full weight
            loss_box = F.l1_loss(matched_pred[:, :8], matched_gt[:, :8],
                                 reduction='mean')

            # Velocity slots 8–11 (vx, vy, vz)
            loss_vel = F.l1_loss(matched_pred[:, 8:11], matched_gt[:, 8:11],
                                 reduction='mean')
            # — down-weighted by weight_vel=0.2 because velocity is noisier and less critical to mAP
            loss_reg = loss_box + self.weight_vel * loss_vel
        else:  # empty-match branch
            loss_reg = pred_anchor.sum() * 0.0   # zero but keeps grad graph alive, so .backward() doesn't error on object-free frames

        total = self.weight_cls * loss_cls + self.weight_reg * loss_reg

        # ---- v3 quality branch (centerness / yawness) on matched queries ----
        if pred_quality is not None and num_matched > 0:
            matched_pred = pred_anchor[pred_idx.to(device)]
            matched_gt   = gt_boxes[gt_idx.to(device)]
            matched_q    = pred_quality[pred_idx.to(device)]
            cns_loss, yns_loss = self.quality_loss(matched_pred, matched_gt, matched_q)
            total = total + self.weight_cns * cns_loss + self.weight_yns * yns_loss

        return total

    # ------------------------------------------------------------------
    # Per-stage (auxiliary) supervision
    # ------------------------------------------------------------------

    def forward_multi(
        self,
        stage_preds: list[tuple[torch.Tensor, torch.Tensor]],  # [(B,N,11),(B,N,C)] × S
        gt_boxes:    torch.Tensor,   # (M, 11)
        gt_labels:   torch.Tensor,   # (M,)
    ) -> torch.Tensor:
        """
        Sum of the single-stage loss over every decoder stage, each stage with its
        own Hungarian matching (mirrors the reference, which supervises every
        'refine' output, not just the last).  
        Mean over stages keeps the loss magnitude comparable to single-stage training.
        """
        # The decoder is 6 stages of iterative refinement (each emits an anchor + logits via stage_preds); 
        # auxiliary supervision at each stage gives every stage a direct gradient signal, 
        # which is what makes the iterative refinement actually learn to converge rather than letting only the last stage carry all the learning.
        # Each stage_pred is (anchor, cls_logits) for v2 or
        # (anchor, cls_logits, quality) for v3 — the 3rd element enables the
        # centerness/yawness supervision above.
        losses = []
        for pred in stage_preds:
            if len(pred) == 3:
                anchor, cls_logits, quality = pred
                q = quality[0] if quality is not None else None
            else:
                anchor, cls_logits = pred
                q = None
            losses.append(self.forward(anchor[0], cls_logits[0],
                                       gt_boxes, gt_labels, pred_quality=q))
        return torch.stack(losses).mean()
