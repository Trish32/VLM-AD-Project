"""
Anchor3DHead — pure-PyTorch port of mmdet3d Anchor3DHead for bevf_pp.

Includes AlignedAnchor3DRangeGenerator anchor generation, DeltaXYZWLHRBBoxCoder
decoding, and the test-time get_bboxes decode path (sigmoid scores, top-k,
delta decode, xywhr2xyxyr, rotated multiclass NMS, direction correction).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .nms_bev import nms_rotated_bev


def limit_period(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period


def xywhr2xyxyr(boxes_xywhr):
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2
    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


class AlignedAnchor3DRangeGenerator:
    """Generates anchors aligned to the voxel/feature grid (mmdet3d)."""

    def __init__(self, ranges, sizes, rotations, custom_values=(), scale=1.0):
        self.ranges = ranges
        self.sizes = sizes
        self.rotations = rotations
        self.custom_values = list(custom_values)
        self.scale = scale

    @property
    def num_base_anchors(self):
        return len(self.rotations) * len(self.sizes)

    def anchors_single_range(self, feature_size, anchor_range, sizes,
                             rotations, device):
        # feature_size: [D, H, W] = [z, y, x]
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device, dtype=torch.float32)
        z_centers = torch.linspace(anchor_range[2], anchor_range[5],
                                   feature_size[0] + 1, device=device)
        y_centers = torch.linspace(anchor_range[1], anchor_range[4],
                                   feature_size[1] + 1, device=device)
        x_centers = torch.linspace(anchor_range[0], anchor_range[3],
                                   feature_size[2] + 1, device=device)
        sizes = torch.tensor(sizes, device=device, dtype=torch.float32).reshape(-1, 3) * self.scale
        rotations = torch.tensor(rotations, device=device, dtype=torch.float32)

        # shift centers to voxel center (align_corner=False)
        z_shift = (z_centers[1] - z_centers[0]) / 2
        y_shift = (y_centers[1] - y_centers[0]) / 2
        x_shift = (x_centers[1] - x_centers[0]) / 2
        z_centers = z_centers + z_shift
        y_centers = y_centers + y_shift
        x_centers = x_centers + x_shift

        rets = torch.meshgrid(x_centers[:feature_size[2]],
                              y_centers[:feature_size[1]],
                              z_centers[:feature_size[0]],
                              rotations, indexing='ij')
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])

        if len(self.custom_values) > 0:
            custom = ret.new_zeros([*ret.shape[:-1], len(self.custom_values)])
            ret = torch.cat([ret, custom], dim=-1)
        return ret

    def grid_anchors(self, featmap_size, device):
        # size_per_range: concat anchors for each (range, size) pair
        mr_anchors = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchors.append(
                self.anchors_single_range(featmap_size, anchor_range,
                                          [anchor_size], self.rotations, device))
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        return mr_anchors.reshape(-1, mr_anchors.size(-1))


def decode_bbox(anchors, deltas):
    """DeltaXYZWLHRBBoxCoder.decode. anchors/deltas (N, >=7)."""
    box_ndim = anchors.shape[-1]
    if box_ndim > 7:
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
    else:
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)
        cas, cts = [], []
    za = za + ha / 2
    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    lg = torch.exp(lt) * la
    wg = torch.exp(wt) * wa
    hg = torch.exp(ht) * ha
    rg = rt + ra
    zg = zg - hg / 2
    cgs = [t + a for t, a in zip(cts, cas)]
    return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)


class Anchor3DHead(nn.Module):
    def __init__(self, num_classes, in_channels, feat_channels,
                 anchor_generator, code_size=9,
                 dir_offset=0.7854, dir_limit_offset=0.0, test_cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.box_code_size = code_size
        self.anchor_generator = anchor_generator
        self.num_anchors = anchor_generator.num_base_anchors
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.test_cfg = test_cfg
        self.use_sigmoid_cls = True

        self.conv_cls = nn.Conv2d(feat_channels, self.num_anchors * num_classes, 1)
        self.conv_reg = nn.Conv2d(feat_channels, self.num_anchors * code_size, 1)
        self.conv_dir_cls = nn.Conv2d(feat_channels, self.num_anchors * 2, 1)

    def forward(self, x):
        return self.conv_cls(x), self.conv_reg(x), self.conv_dir_cls(x)

    @torch.no_grad()
    def get_bboxes_single(self, cls_score, bbox_pred, dir_cls_pred, device):
        cfg = self.test_cfg
        featmap_size = cls_score.shape[-2:]
        anchors = self.anchor_generator.grid_anchors(featmap_size, device)

        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)
        scores = cls_score.sigmoid()
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.box_code_size)

        nms_pre = cfg['nms_pre']
        if 0 < nms_pre < scores.shape[0]:
            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]
            dir_cls_score = dir_cls_score[topk_inds]

        bboxes = decode_bbox(anchors, bbox_pred)
        bev = bboxes[:, [0, 1, 3, 4, 6]]
        bboxes_for_nms = xywhr2xyxyr(bev)

        # multiclass rotated NMS
        out_bboxes, out_scores, out_labels, out_dir = [], [], [], []
        for c in range(self.num_classes):
            cls_inds = scores[:, c] > cfg['score_thr']
            if not cls_inds.any():
                continue
            _scores = scores[cls_inds, c]
            _boxes_nms = bboxes_for_nms[cls_inds, :]
            keep = nms_rotated_bev(_boxes_nms, _scores, cfg['nms_thr'])
            _bboxes = bboxes[cls_inds, :][keep]
            out_bboxes.append(_bboxes)
            out_scores.append(_scores[keep])
            out_labels.append(torch.full((keep.numel(),), c,
                                         dtype=torch.long, device=device))
            out_dir.append(dir_cls_score[cls_inds][keep])

        if out_bboxes:
            bboxes = torch.cat(out_bboxes, 0)
            scores = torch.cat(out_scores, 0)
            labels = torch.cat(out_labels, 0)
            dir_scores = torch.cat(out_dir, 0)
            if bboxes.shape[0] > cfg['max_num']:
                _, inds = scores.sort(descending=True)
                inds = inds[:cfg['max_num']]
                bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
                dir_scores = dir_scores[inds]
            # direction correction
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = dir_rot + self.dir_offset + np.pi * dir_scores.to(bboxes.dtype)
        else:
            bboxes = cls_score.new_zeros((0, self.box_code_size))
            scores = cls_score.new_zeros((0,))
            labels = cls_score.new_zeros((0,), dtype=torch.long)
        return bboxes, scores, labels
