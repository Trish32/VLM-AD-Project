"""
Temporal Instance Denoising (DN) for Sparse4D-v3 — pure PyTorch.

Adapted from DN-DETR / DINO to Sparse4D's anchor queries.  During training we
add `num_groups` NOISED copies of the ground-truth boxes as extra "denoising"
queries alongside the regular 900 object queries, and train the model to recover
the clean box from each noised one.  Because the GT↔query correspondence is
KNOWN (no Hungarian needed), DN gives a stable, noise-free learning signal that
stabilises bipartite matching and accelerates convergence — exactly the benefit
that is otherwise missing on a tiny dataset like nuScenes-mini.

Three pieces live here:
  1. generate_dn_groups : build noised anchors + their target labels/boxes
  2. build_dn_attn_mask : the DINO attention mask that ISOLATES DN groups from
                          the regular queries (and from each other), so DN is
                          invisible to the normal path → inference unaffected
  3. DNLoss             : direct focal + L1 loss with known correspondence

This is SINGLE-FRAME instance denoising (the DN groups are rebuilt each frame
and do not persist through the temporal cache).  That captures the bulk of the
matching-stability benefit; full temporal propagation of DN instances through
InstanceBank is a documented extension (see forward_train in sparse4d_v3.py).

Anchor layout: [x, y, z, log_w, log_l, log_h, sin_yaw, cos_yaw, vx, vy, vz]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import FocalLoss


@torch.no_grad()
def generate_dn_groups(
    gt_boxes:    torch.Tensor,   # (M, 11)  log-space anchor format, lidar frame
    gt_labels:   torch.Tensor,   # (M,)     long
    num_groups:  int   = 5,
    noise_pos:   float = 0.5,    # position noise as a fraction of box size
    noise_size:  float = 0.2,    # log-size jitter (additive in log-space)
    noise_yaw:   float = 0.25,   # yaw jitter (radians, scaled by pi)
    label_flip:  float = 0.1,    # fraction of labels randomly flipped (class noise)
    num_classes: int   = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Returns
    -------
    dn_anchor   : (G*M, 11)  noised anchors
    dn_labels   : (G*M,)     target class for each DN query (label noise applied)
    dn_gt_boxes : (G*M, 11)  the CLEAN box each DN query must regress to
    M           : int        GT count per group
    """
    M = gt_labels.shape[0]
    device = gt_boxes.device
    if M == 0:
        empty = torch.zeros(0, 11, device=device)
        return empty, torch.zeros(0, dtype=torch.long, device=device), empty, 0

    G = num_groups
    # Tile GT G times: (G*M, ·)
    base = gt_boxes.unsqueeze(0).expand(G, M, 11).reshape(G * M, 11).clone()
    dn_gt_boxes = base.clone()
    dn_labels = gt_labels.unsqueeze(0).expand(G, M).reshape(G * M).clone()

    # --- Position noise: ± noise_pos × metric box size ---
    size = base[:, 3:6].exp()                              # (G*M, 3) metric w,l,h
    pos_jitter = (torch.rand_like(base[:, 0:3]) * 2 - 1) * noise_pos * size
    base[:, 0:3] = base[:, 0:3] + pos_jitter

    # --- Size noise: additive jitter in log-space ---
    base[:, 3:6] = base[:, 3:6] + (torch.rand_like(base[:, 3:6]) * 2 - 1) * noise_size

    # --- Yaw noise: jitter the angle, rewrite sin/cos ---
    yaw = torch.atan2(base[:, 6], base[:, 7])             # current angle
    yaw = yaw + (torch.rand_like(yaw) * 2 - 1) * noise_yaw * math.pi
    base[:, 6] = yaw.sin()
    base[:, 7] = yaw.cos()

    # --- Label noise: randomly flip a fraction of classes ---
    if label_flip > 0:
        flip = torch.rand(G * M, device=device) < label_flip
        rand_cls = torch.randint(0, num_classes, (G * M,), device=device)
        dn_labels = torch.where(flip, rand_cls, dn_labels)

    return base, dn_labels, dn_gt_boxes, M


def build_dn_attn_mask(num_reg: int, num_groups: int, M: int,
                       device: torch.device) -> torch.Tensor:
    """
    DINO-style self-attention mask over concat [regular(num_reg), dn(G*M)].

    Returns a bool (Ntot, Ntot) mask where True = NOT allowed to attend:
      • regular queries attend only to regular queries (DN invisible)
      • DN queries attend only within their own group (no cross-group, no regular)

    No row is ever fully masked (regular sees regular; each DN sees its group),
    so attention never produces NaN.
    """
    P = num_groups * M
    Ntot = num_reg + P
    mask = torch.ones(Ntot, Ntot, dtype=torch.bool, device=device)  # block all

    # regular ↔ regular allowed
    mask[:num_reg, :num_reg] = False

    # within-group DN allowed
    for g in range(num_groups):
        s = num_reg + g * M
        e = s + M
        mask[s:e, s:e] = False

    return mask


class DNLoss(nn.Module):
    """
    Denoising loss with KNOWN correspondence (no Hungarian).  Each DN query maps
    to a specific GT, so we apply focal classification + L1 regression directly,
    summed over all decoder stages and averaged.
    """

    def __init__(self, num_classes: int = 10, weight_cls: float = 2.0,
                 weight_reg: float = 0.25, weight_vel: float = 0.2,
                 weight_cns: float = 1.0, weight_yns: float = 1.0,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.weight_cls  = weight_cls
        self.weight_reg  = weight_reg
        self.weight_vel  = weight_vel
        self.weight_cns  = weight_cns
        self.weight_yns  = weight_yns
        self.focal       = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def _stage_loss(self, dn_anchor, dn_logits, dn_labels, dn_gt_boxes,
                    dn_quality=None):
        P = dn_labels.shape[0]
        if P == 0:
            return dn_anchor.sum() * 0.0

        # Classification: one-hot target at the (possibly flipped) label
        cls_targets = torch.zeros(P, self.num_classes, device=dn_logits.device)
        cls_targets[torch.arange(P, device=dn_logits.device), dn_labels] = 1.0
        loss_cls = self.focal(dn_logits, cls_targets) / max(P, 1)

        # Regression: direct L1 to the clean box (known correspondence)
        loss_box = F.l1_loss(dn_anchor[:, :8], dn_gt_boxes[:, :8], reduction='mean')
        loss_vel = F.l1_loss(dn_anchor[:, 8:11], dn_gt_boxes[:, 8:11], reduction='mean')
        loss_reg = loss_box + self.weight_vel * loss_vel

        total = self.weight_cls * loss_cls + self.weight_reg * loss_reg

        # v3 quality branch — every DN query is a positive, so supervise all P
        if dn_quality is not None:
            from .loss import Sparse4DLoss
            cns_loss, yns_loss = Sparse4DLoss.quality_loss(
                dn_anchor, dn_gt_boxes, dn_quality)
            total = total + self.weight_cns * cns_loss + self.weight_yns * yns_loss

        return total

    def forward(self, dn_stage_preds, dn_labels, dn_gt_boxes):
        """
        dn_stage_preds : [(dn_anchor (P,11), dn_logits (P,C)[, dn_quality (P,2)])] × S
        """
        losses = []
        for pred in dn_stage_preds:
            if len(pred) == 3:
                a, c, q = pred
            else:
                a, c = pred
                q = None
            losses.append(self._stage_loss(a, c, dn_labels, dn_gt_boxes, dn_quality=q))
        return torch.stack(losses).mean()
