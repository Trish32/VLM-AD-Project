"""
TransFusion training loss (pure PyTorch) for the det fine-tune smoke test:
- Gaussian heatmap target + gaussian focal loss (dense supervision)
- Hungarian one-to-one matching (cls focal cost + L1 reg cost) in the encoded space
- classification focal loss + bbox L1 (code-weighted) on matched queries
Matches the official TransFusionHead.loss formulation closely enough to verify
the training path (loss decreases). out_size_factor=8.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

CODE_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]


def gaussian_radius(det_size, min_overlap=0.1):
    """How big to make the Gaussian blob for a box of this size: the largest radius such
    that a box shifted by that radius still overlaps the GT by >= min_overlap IoU. Solving
    the 3 overlap configurations (inside/outside/straddle) gives 3 quadratics; take the min.
    (Standard CenterNet heuristic.) Returns an integer radius in BEV cells."""
    h, w = det_size
    a1, b1, c1 = 1, (h + w), w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + (b1 ** 2 - 4 * a1 * c1) ** 0.5) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + (b2 ** 2 - 4 * a2 * c2) ** 0.5) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + (b3 ** 2 - 4 * a3 * c3) ** 0.5) / 2
    return max(0, int(min(r1, r2, r3)))


def draw_gaussian(heatmap, center, radius):
    """Paint a 2D Gaussian bump into heatmap[X,Y] centered at integer (cx,cy), peak=1.
    Uses torch.maximum (not +=) so overlapping blobs keep the stronger value rather than
    summing past 1. The clipping logic handles centers near the grid edge."""
    diameter = 2 * radius + 1
    sigma = diameter / 6
    m = radius
    ys, xs = np.ogrid[-m:m + 1, -m:m + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))   # (d,d) gaussian, peak 1 at center
    g = torch.from_numpy(g).to(heatmap)
    cx, cy = int(center[0]), int(center[1])
    X, Y = heatmap.shape
    # clip the blob to the heatmap bounds (left/right/top/bottom margins)
    l, r = min(cx, radius), min(X - cx, radius + 1)
    t, b = min(cy, radius), min(Y - cy, radius + 1)
    if r <= -l or b <= -t:
        return
    masked = heatmap[cx - l:cx + r, cy - t:cy + b]
    gm = g[radius - l:radius + r, radius - t:radius + b]
    torch.maximum(masked, gm, out=masked)                    # keep max where blobs overlap


def encode(boxes, pc_range, voxel, osf=8):
    """Convert metric GT boxes [x,y,z_bottom,w,l,h,yaw,vx,vy] -> the 10-d target the head
    predicts in its RAW space, mirroring _decode in the head, so loss is computed apples-to-apples with res[*]:
      xy -> BEV cell units; z -> gravity center; wlh -> log; yaw -> (sin,cos); vel as-is."""
    t = boxes.new_zeros((boxes.shape[0], 10))
    t[:, 0] = (boxes[:, 0] - pc_range[0]) / (osf * voxel[0])  # x metres -> cell index
    t[:, 1] = (boxes[:, 1] - pc_range[1]) / (osf * voxel[1])  # y metres -> cell index
    t[:, 2] = boxes[:, 2] + boxes[:, 5] * 0.5                 # z_bottom + h/2 = gravity center
    t[:, 3] = boxes[:, 3].log()                              # log-size (matches dim.exp() in decode)
    t[:, 4] = boxes[:, 4].log()
    t[:, 5] = boxes[:, 5].log()
    t[:, 6] = torch.sin(boxes[:, 6])                         # yaw -> (sin, cos) to avoid wraparound
    t[:, 7] = torch.cos(boxes[:, 6])
    t[:, 8:10] = boxes[:, 7:9]                               # velocity
    return t


def gaussian_focal(pred, target, alpha=2.0, gamma=4.0, eps=1e-12):
    """Penalized-focal loss for the DENSE heatmap (CenterNet style). The target is a soft
    Gaussian, not 0/1: only the exact peak (target==1) is a true positive; every other cell
    is a negative whose penalty is DOWN-WEIGHTED by (1-target)^gamma -- so cells near a peak
    (target close to 1) are barely penalized, distant cells fully. Normalized by #positives."""
    pred = torch.clamp(pred.sigmoid(), eps, 1 - eps)
    pos = target.eq(1).float()                  # exact peaks
    neg = (1 - pos)
    neg_w = (1 - target) ** gamma               # soft down-weight near peaks
    pos_loss = -(1 - pred) ** alpha * torch.log(pred) * pos
    neg_loss = -pred ** alpha * torch.log(1 - pred) * neg_w * neg
    n = pos.sum()
    return (pos_loss.sum() + neg_loss.sum()) / max(n, 1)


def focal_cls(logits, labels, weights, alpha=0.25, gamma=2.0, num_pos=1):
    """Sigmoid focal loss for the SPARSE per-query classification. labels in [0,C]: a value
    == C means 'unmatched / background' (no positive class -> all-zero one-hot). Focal term
    (1-pt)^gamma focuses learning on hard examples; normalized by #matched queries."""
    # logits (N, C), labels (N,) in [0,C], weights (N,)
    N, C = logits.shape
    oh = torch.zeros_like(logits)
    valid = labels < C                          # matched queries get a one-hot; background stays 0
    oh[valid, labels[valid]] = 1.0
    p = logits.sigmoid()
    pt = oh * p + (1 - oh) * (1 - p)            # prob of the correct (target) outcome
    at = oh * alpha + (1 - oh) * (1 - alpha)    # class-balance weight
    loss = at * (1 - pt) ** gamma * F.binary_cross_entropy_with_logits(logits, oh, reduction='none')
    return (loss.sum(1) * weights).sum() / max(num_pos, 1)


def transfusion_loss(res, dense_heatmap, gt_boxes, gt_labels, cfg):
    """Total TransFusion loss = dense heatmap loss + sparse (cls + bbox) loss on the
    Hungarian-matched queries. res: dict of (B=1,*,P). gt_boxes (G,9) lidar, gt_labels (G,)."""
    device = dense_heatmap.device
    pc, vx = cfg.POINT_CLOUD_RANGE, cfg.VOXEL_SIZE
    nc = cfg.NUM_CLASSES if hasattr(cfg, 'NUM_CLASSES') else 10
    nc = 10
    X = Y = cfg.GRID_SIZE[0] // 8        # decoder BEV size (180)
    P = res['center'].shape[-1]          # number of queries (200)

    # ---- dense heatmap target: paint a Gaussian bump at each GT center, in its class channel ----
    hm = dense_heatmap.new_zeros((nc, X, Y))
    if gt_boxes.shape[0] > 0:
        for i in range(gt_boxes.shape[0]):
            w, l = float(gt_boxes[i, 3]), float(gt_boxes[i, 4])
            rad = gaussian_radius((l / (8 * vx[1]), w / (8 * vx[0])), 0.1)   # blob size from box size
            rad = max(2, rad)
            cx = (gt_boxes[i, 0] - pc[0]) / (8 * vx[0])   # GT center -> BEV cell
            cy = (gt_boxes[i, 1] - pc[1]) / (8 * vx[1])
            if 0 <= cx < X and 0 <= cy < Y:
                draw_gaussian(hm[int(gt_labels[i])], (cx, cy), rad)
    loss_hm = gaussian_focal(dense_heatmap[0], hm)         # supervises the heatmap_head

    # ---- per-query raw predictions, assembled into the same 10-d encoded space as the targets ----
    pred = torch.cat([res['center'], res['height'], res['dim'], res['rot'], res['vel']], 1)[0].T  # (P,10)
    cls_logits = res['heatmap'][0].T  # (P, nc)

    # default every query to BACKGROUND (label=nc) with zero bbox target/weight...
    labels = torch.full((P,), nc, dtype=torch.long, device=device)
    label_w = torch.ones(P, device=device)
    bbox_t = torch.zeros((P, 10), device=device)
    bbox_w = torch.zeros((P, 10), device=device)

    if gt_boxes.shape[0] > 0:
        enc = encode(gt_boxes, pc, vx)  # (G,10) targets in encoded space
        # ---- Hungarian matching: build a (P x G) cost, assign one query per GT optimally ----
        # cost = classification focal cost (how unlikely each query calls the GT's class)
        #        + L1 regression cost on the geometric dims (how far the box is)
        p = cls_logits.sigmoid()
        cls_cost = (-(1 - p + 1e-8).log() * 0.25 * p ** 2)[:, gt_labels] \
            - (-(p + 1e-8).log() * 0.75 * (1 - p) ** 2)[:, gt_labels]  # (P,G)
        reg_cost = torch.cdist(pred[:, :8], enc[:, :8], p=1)            # (P,G) L1 distance
        cost = (cls_cost + 0.25 * reg_cost).detach().cpu().numpy()
        ri, ci = linear_sum_assignment(cost)   # optimal 1-to-1: query ri[k] <-> GT ci[k]
        ri = torch.as_tensor(ri, device=device)
        ci = torch.as_tensor(ci, device=device)
        labels[ri] = gt_labels[ci]             # matched queries become POSITIVES with the GT class
        bbox_t[ri] = enc[ci]                   # ...their regression target = the matched GT
        bbox_w[ri] = 1.0                       # ...and only matched queries contribute to bbox loss

    num_pos = max(int((labels < nc).sum()), 1)             # #matched queries (loss normalizer)
    loss_cls = focal_cls(cls_logits, labels, label_w, num_pos=num_pos)      # classify all P queries
    cw = torch.tensor(CODE_WEIGHTS, device=device)         # per-attribute weights (velocity 0.2)
    # L1 box regression, only on matched queries (bbox_w), code-weighted
    loss_bbox = (F.l1_loss(pred, bbox_t, reduction='none') * bbox_w * cw).sum() / num_pos
    return loss_hm, loss_cls, loss_bbox, num_pos
