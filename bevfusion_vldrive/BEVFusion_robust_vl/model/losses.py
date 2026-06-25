"""
Training targets + losses for Anchor3DHead (bevf_pp), pure PyTorch.

- nearest-BEV IoU MaxIoUAssigner (pos 0.6 / neg 0.3 / min_pos 0.3)
- DeltaXYZWLHRBBoxCoder.encode + direction target
- sigmoid FocalLoss (cls), SmoothL1 (bbox, code_weight), CrossEntropy (dir)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def limit_period(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period


def encode_bbox(src, dst):
    """DeltaXYZWLHRBBoxCoder.encode. src/dst (N, >=7)."""
    xa, ya, za, wa, la, ha, ra, *cas = torch.split(src, 1, dim=-1)
    xg, yg, zg, wg, lg, hg, rg, *cgs = torch.split(dst, 1, dim=-1)
    za = za + ha / 2
    zg = zg + hg / 2
    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha
    lt = torch.log(lg / la)
    wt = torch.log(wg / wa)
    ht = torch.log(hg / ha)
    rt = rg - ra
    cts = [g - a for g, a in zip(cgs, cas)]
    return torch.cat([xt, yt, zt, wt, lt, ht, rt, *cts], dim=-1)


def get_direction_target(anchors, reg_targets, dir_offset=0.0, num_bins=2):
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    return torch.clamp(dir_cls, 0, num_bins - 1)


def _bev_aligned_xyxy(boxes):
    """Rotated BEV boxes (N, >=7) -> axis-aligned enclosing xyxy (nearest 3D)."""
    cx, cy = boxes[:, 0], boxes[:, 1]
    w, l = boxes[:, 3], boxes[:, 4]
    yaw = boxes[:, 6]
    cos, sin = torch.cos(yaw).abs(), torch.sin(yaw).abs()
    # enclosing extents of a rotated w x l rectangle
    ex = (w * cos + l * sin) / 2
    ey = (w * sin + l * cos) / 2
    return torch.stack([cx - ex, cy - ey, cx + ex, cy + ey], dim=1)


def bev_iou(anchors, gts):
    """Nearest-BEV IoU between anchors (A,>=7) and gts (G,>=7) -> (A, G)."""
    a = _bev_aligned_xyxy(anchors)
    g = _bev_aligned_xyxy(gts)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    lt = torch.max(a[:, None, :2], g[None, :, :2])
    rb = torch.min(a[:, None, 2:], g[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_g[None, :] - inter
    return inter / union.clamp(min=1e-6)


def assign_targets(anchors, gt_boxes, gt_labels, num_classes,
                   pos_iou_thr=0.6, neg_iou_thr=0.3, min_pos_iou=0.3):
    """MaxIoUAssigner. Returns labels, bbox_targets, bbox_weights,
    dir_targets, dir_weights, pos_inds."""
    A = anchors.shape[0]
    device = anchors.device
    labels = anchors.new_full((A,), num_classes, dtype=torch.long)
    label_weights = anchors.new_zeros(A)
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    dir_targets = anchors.new_zeros(A, dtype=torch.long)
    dir_weights = anchors.new_zeros(A)

    if gt_boxes.shape[0] == 0:
        label_weights[:] = 1.0
        return labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, \
            torch.empty(0, dtype=torch.long, device=device)

    iou = bev_iou(anchors, gt_boxes)            # (A, G)
    max_iou, argmax = iou.max(dim=1)            # best gt per anchor
    gt_max_iou, gt_argmax = iou.max(dim=0)      # best anchor per gt

    # negatives
    neg_mask = max_iou < neg_iou_thr
    label_weights[neg_mask] = 1.0

    # positives by threshold
    pos_mask = max_iou >= pos_iou_thr
    # force each gt's best anchor to be positive if >= min_pos_iou
    for g in range(gt_boxes.shape[0]):
        if gt_max_iou[g] >= min_pos_iou:
            pos_mask[gt_argmax[g]] = True
            argmax[gt_argmax[g]] = g

    pos_inds = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
    if pos_inds.numel() > 0:
        pos_gt = argmax[pos_inds]
        pos_anchors = anchors[pos_inds]
        pos_gt_boxes = gt_boxes[pos_gt]
        bbox_targets[pos_inds] = encode_bbox(pos_anchors, pos_gt_boxes)
        bbox_weights[pos_inds] = 1.0
        dir_targets[pos_inds] = get_direction_target(pos_anchors, bbox_targets[pos_inds])
        dir_weights[pos_inds] = 1.0
        labels[pos_inds] = gt_labels[pos_gt]
        label_weights[pos_inds] = 1.0
    return labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, pos_inds


def sigmoid_focal_loss(pred, target, weight, num_classes, gamma=2.0, alpha=0.25, avg_factor=1.0):
    """pred (N, C) logits; target (N,) in [0, C] (C = background)."""
    N, C = pred.shape
    onehot = torch.zeros_like(pred)
    valid = target < num_classes
    onehot[valid, target[valid]] = 1.0
    p = pred.sigmoid()
    pt = onehot * p + (1 - onehot) * (1 - p)
    at = onehot * alpha + (1 - onehot) * (1 - alpha)
    focal = at * (1 - pt).pow(gamma) * F.binary_cross_entropy_with_logits(
        pred, onehot, reduction='none')
    focal = focal * weight[:, None]
    return focal.sum() / avg_factor


def smooth_l1(pred, target, weight, beta=1.0 / 9.0, avg_factor=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return (loss * weight).sum() / avg_factor


def add_sin_difference(b1, b2):
    rad_pred = torch.sin(b1[..., 6:7]) * torch.cos(b2[..., 6:7])
    rad_tg = torch.cos(b1[..., 6:7]) * torch.sin(b2[..., 6:7])
    b1 = torch.cat([b1[..., :6], rad_pred, b1[..., 7:]], dim=-1)
    b2 = torch.cat([b2[..., :6], rad_tg, b2[..., 7:]], dim=-1)
    return b1, b2
