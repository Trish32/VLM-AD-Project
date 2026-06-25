"""
Rotated-BEV NMS in pure PyTorch, replacing mmdet3d's CUDA `nms_gpu`.

Boxes for NMS are in XYXYR format (x1, y1, x2, y2, ry) as produced by
`xywhr2xyxyr`. We reconstruct the rotated rectangle (cx, cy, w, h, angle),
compute pairwise rotated-IoU via convex-polygon (Sutherland-Hodgman)
intersection, and run greedy NMS — matching the semantics of iou3d's BEV
rotated NMS used by `box3d_multiclass_nms`.
"""
from __future__ import annotations

import torch


def _xyxyr_to_corners(boxes):
    """(N,5) [x1,y1,x2,y2,r] -> (N,4,2) rotated rectangle corners."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = (boxes[:, 2] - boxes[:, 0])
    h = (boxes[:, 3] - boxes[:, 1])
    angle = boxes[:, 4]
    cos, sin = torch.cos(angle), torch.sin(angle)
    # local corners (counter-clockwise)
    dx = torch.stack([-w / 2, w / 2, w / 2, -w / 2], dim=1)  # (N,4)
    dy = torch.stack([-h / 2, -h / 2, h / 2, h / 2], dim=1)
    xs = cx[:, None] + dx * cos[:, None] - dy * sin[:, None]
    ys = cy[:, None] + dx * sin[:, None] + dy * cos[:, None]
    return torch.stack([xs, ys], dim=2)  # (N,4,2)


def _poly_area(poly):
    """Shoelace area of a polygon given as (K,2)."""
    if poly.shape[0] < 3:
        return poly.new_tensor(0.0)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * torch.abs(
        torch.dot(x, torch.roll(y, -1)) - torch.dot(y, torch.roll(x, -1)))


def _clip_polygon(subject, clip):
    """Sutherland-Hodgman: clip `subject` polygon by convex `clip` polygon.
    Both (K,2) tensors with CCW orientation. Returns (M,2)."""
    output = subject
    K = clip.shape[0]
    for i in range(K):
        if output.shape[0] == 0:
            break
        a = clip[i]
        b = clip[(i + 1) % K]
        edge = b - a
        # inside test: cross(edge, p - a) >= 0  (CCW clip => left side inside)
        inp = output
        prev = torch.roll(inp, 1, dims=0)
        # cross products
        cur_cross = edge[0] * (inp[:, 1] - a[1]) - edge[1] * (inp[:, 0] - a[0])
        prev_cross = edge[0] * (prev[:, 1] - a[1]) - edge[1] * (prev[:, 0] - a[0])
        cur_in = cur_cross >= 0
        prev_in = prev_cross >= 0
        new_pts = []
        n = inp.shape[0]
        for j in range(n):
            c_in = bool(cur_in[j])
            p_in = bool(prev_in[j])
            P = prev[j]
            Cp = inp[j]
            if c_in:
                if not p_in:
                    new_pts.append(_intersect(P, Cp, a, b))
                new_pts.append(Cp)
            elif p_in:
                new_pts.append(_intersect(P, Cp, a, b))
        if new_pts:
            output = torch.stack(new_pts, dim=0)
        else:
            output = output.new_zeros((0, 2))
    return output


def _intersect(p1, p2, a, b):
    """Intersection of segment p1-p2 with line a-b."""
    r = p2 - p1
    s = b - a
    denom = r[0] * s[1] - r[1] * s[0]
    if torch.abs(denom) < 1e-9:
        return p2
    t = ((a[0] - p1[0]) * s[1] - (a[1] - p1[1]) * s[0]) / denom
    return p1 + t * r


def _rotated_iou_matrix(corners, areas):
    """Pairwise rotated IoU. corners:(N,4,2), areas:(N,). Returns (N,N)."""
    n = corners.shape[0]
    iou = corners.new_zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter_poly = _clip_polygon(corners[i], corners[j])
            inter = _poly_area(inter_poly)
            union = areas[i] + areas[j] - inter
            v = inter / union if union > 0 else inter.new_tensor(0.0)
            iou[i, j] = v
            iou[j, i] = v
    return iou


def nms_rotated_bev(boxes_xyxyr, scores, thresh, pre_max=None):
    """
    Greedy rotated NMS.
    Args:
        boxes_xyxyr: (N, 5) [x1, y1, x2, y2, r]
        scores: (N,)
        thresh: IoU threshold
    Returns: LongTensor of kept indices (sorted by score desc).
    """
    if boxes_xyxyr.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_xyxyr.device)
    order = torch.argsort(scores, descending=True)
    if pre_max is not None and order.numel() > pre_max:
        order = order[:pre_max]
    corners = _xyxyr_to_corners(boxes_xyxyr)
    areas = torch.stack([_poly_area(corners[i]) for i in range(corners.shape[0])])

    keep = []
    order = order.tolist()
    suppressed = set()
    for _i, i in enumerate(order):
        if i in suppressed:
            continue
        keep.append(i)
        ci = corners[i]
        ai = areas[i]
        for j in order[_i + 1:]:
            if j in suppressed:
                continue
            inter = _poly_area(_clip_polygon(ci, corners[j]))
            union = ai + areas[j] - inter
            iou = inter / union if union > 0 else inter * 0
            if iou > thresh:
                suppressed.add(j)
    return torch.tensor(keep, dtype=torch.long, device=boxes_xyxyr.device)
