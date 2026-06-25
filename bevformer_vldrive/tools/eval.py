#!/usr/bin/env python3
"""
Evaluate BEVFormer-Tiny on nuScenes mini val split.

Loads backbone + neck + encoder + embeddings + decoder self-attention
from the official fp16 checkpoint. Reports:
  • nuScenes mAP + NDS  (via devkit NuScenesEval)
  • BEV occupancy mIoU per class  (rasterised on the 50×50 grid)

Usage:
    python tools/eval.py \
        --checkpoint model/checkpoints/bevformer_tiny_fp16_epoch_24.pth \
        --dataroot /Users/trish/Downloads/nuScenes_miniV1.0 \
        [--force-cpu] [--max-total-frames N]

Weight coverage from checkpoint
---------------------------------
  Backbone (ResNet-50)          ✓  fully loaded
  FPN neck                      ✓  fully loaded
  BEV queries / pos-enc         ✓  fully loaded
  CAN-bus MLP                   ✓  fully loaded
  Encoder TSA (all 3 layers)    ✓  shapes match — fully loaded
  Encoder SCA (all 3 layers)    ✓  shapes match — fully loaded
  Decoder self-attn (all 6)     ✓  fully loaded
  Decoder cross-attn (all 6)    ✗  architecture differs (deformable vs std MHA)
  cls_branch (final linear)     ✓  loaded (only last of 3-layer official branch)
  reg_branch (first + last)     ✓  loaded (first and last linear of 3-layer branch)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.utils.splits import create_splits_scenes

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import BEVFormerTiny
from data import NuScenesMiniLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PC_RANGE    = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
BEV_H = BEV_W = 50

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# nuScenes annotation category → detection class name
NUSC_CAT_MAP = {
    'movable_object.barrier':               'barrier',
    'vehicle.bicycle':                      'bicycle',
    'vehicle.bus.bendy':                    'bus',
    'vehicle.bus.rigid':                    'bus',
    'vehicle.car':                          'car',
    'vehicle.construction':                 'construction_vehicle',
    'vehicle.motorcycle':                   'motorcycle',
    'human.pedestrian.adult':               'pedestrian',
    'human.pedestrian.child':               'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer':      'pedestrian',
    'movable_object.trafficcone':           'traffic_cone',
    'vehicle.trailer':                      'trailer',
    'vehicle.truck':                        'truck',
}

CLASS_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

ATTR_DEFAULT = {
    'car':                   'vehicle.parked',
    'truck':                 'vehicle.parked',
    'construction_vehicle':  'vehicle.parked',
    'bus':                   'vehicle.parked',
    'trailer':               'vehicle.parked',
    'motorcycle':            'cycle.without_rider',
    'bicycle':               'cycle.without_rider',
    'pedestrian':            'pedestrian.standing',
    'barrier':               '',
    'traffic_cone':          '',
}


# ---------------------------------------------------------------------------
# Weight loading — extended remap from official checkpoint
# ---------------------------------------------------------------------------

def _build_remap(raw: dict) -> dict:
    """
    Returns a dict  { our_model_key : weight_tensor }  covering every
    parameter from the official checkpoint that can be loaded into our
    pure-PyTorch model without shape conflicts.
    """
    m: dict = {}

    def _take(src_key: str, dst_key: str, transform=None):
        if src_key not in raw:
            return
        v = raw[src_key].float()           # fp16 → fp32
        if transform is not None:
            v = transform(v)
        m[dst_key] = v

    # ---- Backbone & neck --------------------------------------------------
    for k in raw:
        if k.startswith('img_backbone.'):
            m['backbone.' + k[len('img_backbone.'):]] = raw[k].float()
        elif k.startswith('img_neck.'):
            m['neck.' + k[len('img_neck.'):]] = raw[k].float()

    # ---- BEV embeddings ---------------------------------------------------
    _take('pts_bbox_head.bev_embedding.weight',               'bev_queries')
    _take('pts_bbox_head.positional_encoding.row_embed.weight', 'bev_pos_enc.row.weight')
    _take('pts_bbox_head.positional_encoding.col_embed.weight', 'bev_pos_enc.col.weight')
    _take('pts_bbox_head.transformer.cams_embeds',             'cams_embeds')
    # Official has 4 multi-scale levels; we use 1 → take first row
    _take('pts_bbox_head.transformer.level_embeds', 'level_embeds',
          transform=lambda v: v[:1])

    # ---- CAN-bus MLP  (Sequential: Linear, ReLU, Linear, LayerNorm) ------
    for suf in ('weight', 'bias'):
        _take(f'pts_bbox_head.transformer.can_bus_mlp.0.{suf}', f'can_bus_mlp.0.{suf}')
        _take(f'pts_bbox_head.transformer.can_bus_mlp.2.{suf}', f'can_bus_mlp.2.{suf}')
        _take(f'pts_bbox_head.transformer.can_bus_mlp.norm.{suf}', f'can_bus_mlp.4.{suf}')

    # ---- Query embedding + reference points ------------------------------
    _take('pts_bbox_head.query_embedding.weight',         'det_head.query_embed.weight')
    _take('pts_bbox_head.transformer.reference_points.weight', 'det_head.ref_points.weight')
    _take('pts_bbox_head.transformer.reference_points.bias',   'det_head.ref_points.bias')

    # ---- Encoder layers (3×) ----------------------------------------------
    # Layer i: TSA (attentions.0) + SCA (attentions.1) + FFN + 3 norms
    for i in range(3):
        src = f'pts_bbox_head.transformer.encoder.layers.{i}'
        dst = f'encoder.layers.{i}'

        # TSA ------------------------------------------
        for suf in ('weight', 'bias'):
            _take(f'{src}.attentions.0.sampling_offsets.{suf}',   f'{dst}.tsa.sampling_offsets.{suf}')
            _take(f'{src}.attentions.0.attention_weights.{suf}',   f'{dst}.tsa.attention_weights.{suf}')
            _take(f'{src}.attentions.0.value_proj.{suf}',          f'{dst}.tsa.value_proj.{suf}')
            _take(f'{src}.attentions.0.output_proj.{suf}',         f'{dst}.tsa.output_proj.{suf}')

        # SCA ------------------------------------------
        for suf in ('weight', 'bias'):
            _take(f'{src}.attentions.1.deformable_attention.sampling_offsets.{suf}',
                  f'{dst}.sca.sampling_offsets.{suf}')
            _take(f'{src}.attentions.1.deformable_attention.attention_weights.{suf}',
                  f'{dst}.sca.attention_weights.{suf}')
            _take(f'{src}.attentions.1.deformable_attention.value_proj.{suf}',
                  f'{dst}.sca.value_proj.{suf}')
            _take(f'{src}.attentions.1.output_proj.{suf}',
                  f'{dst}.sca.output_proj.{suf}')

        # FFN: official ffns.0.layers.0.0 → our ffn.0  (first Linear)
        #      official ffns.0.layers.1   → our ffn.3  (second Linear)
        for suf in ('weight', 'bias'):
            _take(f'{src}.ffns.0.layers.0.0.{suf}', f'{dst}.ffn.0.{suf}')
            _take(f'{src}.ffns.0.layers.1.{suf}',   f'{dst}.ffn.3.{suf}')

        # Norms
        for ni, nn_name in enumerate(['norm1', 'norm2', 'norm3']):
            for suf in ('weight', 'bias'):
                _take(f'{src}.norms.{ni}.{suf}', f'{dst}.{nn_name}.{suf}')

    # ---- Decoder layers (6×): self-attn + BEV deformable cross-attn + FFN --
    for i in range(6):
        src = f'pts_bbox_head.transformer.decoder.layers.{i}'
        dst = f'det_head.decoder_layers.{i}'

        # Self-attention (standard MHA — weight-compatible)
        for suf in ('weight', 'bias'):
            _take(f'{src}.attentions.0.attn.in_proj_{suf}',   f'{dst}.self_attn.in_proj_{suf}')
        _take(f'{src}.attentions.0.attn.out_proj.weight', f'{dst}.self_attn.out_proj.weight')
        _take(f'{src}.attentions.0.attn.out_proj.bias',   f'{dst}.self_attn.out_proj.bias')

        # BEV deformable cross-attention (now weight-compatible!)
        for suf in ('weight', 'bias'):
            _take(f'{src}.attentions.1.sampling_offsets.{suf}',  f'{dst}.cross_attn.sampling_offsets.{suf}')
            _take(f'{src}.attentions.1.attention_weights.{suf}', f'{dst}.cross_attn.attention_weights.{suf}')
            _take(f'{src}.attentions.1.value_proj.{suf}',        f'{dst}.cross_attn.value_proj.{suf}')
            _take(f'{src}.attentions.1.output_proj.{suf}',       f'{dst}.cross_attn.output_proj.{suf}')

        # FFN
        for suf in ('weight', 'bias'):
            _take(f'{src}.ffns.0.layers.0.0.{suf}', f'{dst}.linear1.{suf}')
            _take(f'{src}.ffns.0.layers.1.{suf}',   f'{dst}.linear2.{suf}')

        # Norms
        for ni, nn_name in enumerate(['norm1', 'norm2', 'norm3']):
            for suf in ('weight', 'bias'):
                _take(f'{src}.norms.{ni}.{suf}', f'{dst}.{nn_name}.{suf}')

    # ---- Per-layer reg branches: indices 0, 2, 4 all now present in our model --
    for i in range(6):
        for suf in ('weight', 'bias'):
            _take(f'pts_bbox_head.reg_branches.{i}.0.{suf}', f'det_head.reg_branches.{i}.0.{suf}')
            _take(f'pts_bbox_head.reg_branches.{i}.2.{suf}', f'det_head.reg_branches.{i}.2.{suf}')
            _take(f'pts_bbox_head.reg_branches.{i}.4.{suf}', f'det_head.reg_branches.{i}.4.{suf}')

    # ---- cls_branch: now a full 7-module Sequential matching official cls_branches.5 --
    # Official indices: 0=Linear, 1=LN, (2=ReLU no params), 3=Linear, 4=LN, (5=ReLU), 6=Linear
    for idx in (0, 1, 3, 4, 6):
        for suf in ('weight', 'bias'):
            _take(f'pts_bbox_head.cls_branches.5.{idx}.{suf}', f'det_head.cls_branch.{idx}.{suf}')

    return m


def load_checkpoint(model: BEVFormerTiny, path: str) -> None:
    ckpt = torch.load(path, map_location='cpu')
    raw  = ckpt.get('state_dict', ckpt)

    remap = _build_remap(raw)
    result = model.load_state_dict(remap, strict=False)

    loaded = len(remap) - len(result.unexpected_keys)
    total  = sum(1 for _ in model.parameters())
    print(f'[ckpt] mapped   : {len(remap)} keys from {Path(path).name}')
    print(f'[ckpt] missing  : {len(result.missing_keys)} model keys (random init)')
    if result.unexpected_keys:
        print(f'[ckpt] rejected : {len(result.unexpected_keys)} (shape mismatch)')


# ---------------------------------------------------------------------------
# Prediction decoding  (NMSFreeCoder-style)
# ---------------------------------------------------------------------------

def decode_predictions(
    cls_logits: torch.Tensor,   # (Q, num_classes)
    reg_preds:  torch.Tensor,   # (Q, 10)  — deltas; xyz refined separately via ref_pts
    ref_pts:    torch.Tensor,   # (Q, 3)   — final refined reference points in [0,1]
    pc_range:   list,
    score_thr:  float = 0.1,
    max_num:    int   = 300,
) -> list[dict]:
    """
    Decode model output to box dicts (LiDAR frame).

    NMSFreeCoder reg_preds format: [dx, dy, log_w, log_l, dz, log_h, sin_yaw, cos_yaw, vx, vy]
      x, y, z  ← ref_pts[0:3] * range + min     (already refined via indices 0,1,4 in forward)
      w        ← exp(reg_preds[2])
      l        ← exp(reg_preds[3])
      h        ← exp(reg_preds[5])
      yaw      ← atan2(reg_preds[6], reg_preds[7])
      vx, vy   ← reg_preds[8:10]
    """
    scores, labels = cls_logits.sigmoid().max(-1)   # (Q,)
    order = scores.argsort(descending=True)[:max_num]

    xr = PC_RANGE[3] - PC_RANGE[0]
    yr = PC_RANGE[4] - PC_RANGE[1]
    zr = PC_RANGE[5] - PC_RANGE[2]

    results: list[dict] = []
    for idx in order:
        s = float(scores[idx])
        if s < score_thr:
            break
        r = reg_preds[idx].float()
        p = ref_pts[idx].float()

        x  = float(p[0]) * xr + PC_RANGE[0]
        y  = float(p[1]) * yr + PC_RANGE[1]
        z  = float(p[2]) * zr + PC_RANGE[2]
        w  = float(r[2].exp())   # log_w at index 2
        l  = float(r[3].exp())   # log_l at index 3
        h  = float(r[5].exp())   # log_h at index 5
        # BEVFormer trains with SECOND-format yaw: second_yaw = -nusc_lidar_yaw - π/2
        # Invert to get nuScenes LiDAR-frame yaw: nusc_lidar_yaw = -second_yaw - π/2
        yaw = float(-torch.atan2(r[6], r[7]) - math.pi / 2)
        vx  = float(r[8])
        vy  = float(r[9])

        w = float(np.clip(w, 0.1, 20.0))
        l = float(np.clip(l, 0.1, 20.0))
        h = float(np.clip(h, 0.1, 10.0))

        results.append({
            'score':   s,
            'label':   int(labels[idx]),
            'box3d':   (x, y, z, w, l, h, yaw),
            'vel':     (vx, vy),
        })
    return results


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def get_lidar2global(nusc: NuScenes, sample_token: str):
    """
    Returns (lidar2global 4×4 float64, yaw_total float)
    yaw_total = yaw_lidar2ego + yaw_ego2global  (used to rotate yaw predictions)
    """
    sample   = nusc.get('sample', sample_token)
    lidar_t  = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_t)
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    q_l2e = Quaternion(lidar_cs['rotation'])
    t_l2e = np.array(lidar_cs['translation'])
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = q_l2e.rotation_matrix
    lidar2ego[:3,  3] = t_l2e

    q_e2g = Quaternion(lidar_ep['rotation'])
    t_e2g = np.array(lidar_ep['translation'])
    ego2global = np.eye(4)
    ego2global[:3, :3] = q_e2g.rotation_matrix
    ego2global[:3,  3] = t_e2g

    lidar2global = ego2global @ lidar2ego
    yaw_total    = q_l2e.yaw_pitch_roll[0] + q_e2g.yaw_pitch_roll[0]
    return lidar2global, yaw_total


def preds_to_nuscenes(dets: list[dict], sample_token: str,
                      lidar2global: np.ndarray, yaw_total: float) -> list[dict]:
    """Convert decoded LiDAR-frame boxes to nuScenes submission records."""
    R = lidar2global[:3, :3]
    records = []
    for d in dets:
        x, y, z, w, l, h, yaw = d['box3d']
        vx, vy = d['vel']
        cls_name = CLASS_NAMES[d['label']]

        # Transform centre to global
        c_global = lidar2global @ np.array([x, y, z, 1.0])

        # Rotate yaw to global
        yaw_global = yaw + yaw_total
        q = Quaternion(axis=[0, 0, 1], angle=yaw_global)

        # Rotate velocity to global
        v_global = R[:2, :2] @ np.array([vx, vy])

        records.append({
            'sample_token':    sample_token,
            'translation':     [float(c_global[0]), float(c_global[1]), float(c_global[2])],
            'size':            [w, l, h],
            'rotation':        [q.w, q.x, q.y, q.z],
            'velocity':        [float(v_global[0]), float(v_global[1])],
            'detection_name':  cls_name,
            'detection_score': d['score'],
            'attribute_name':  ATTR_DEFAULT[cls_name],
        })
    return records


# ---------------------------------------------------------------------------
# BEV occupancy mIoU
# ---------------------------------------------------------------------------

def _rasterize_boxes(boxes_lidar: list, pc_range: list,
                     bev_h: int, bev_w: int) -> np.ndarray:
    """
    boxes_lidar: list of (cx, cy, w, l, yaw, class_idx)
    Returns (num_classes, bev_h, bev_w) uint8 occupancy masks.
    """
    grid = np.zeros((len(CLASS_NAMES), bev_h, bev_w), dtype=np.uint8)
    xmin, ymin = pc_range[0], pc_range[1]
    xmax, ymax = pc_range[3], pc_range[4]

    for (cx, cy, w, l, yaw, cls_idx) in boxes_lidar:
        if cls_idx < 0 or cls_idx >= len(CLASS_NAMES):
            continue
        # Pixel coordinates (origin = top-left, y-axis flipped)
        gx = (cx - xmin) / (xmax - xmin) * bev_w
        gy = (cy - ymin) / (ymax - ymin) * bev_h
        gw = w / (xmax - xmin) * bev_w
        gl = l / (ymax - ymin) * bev_h

        ca, sa = np.cos(yaw), np.sin(yaw)
        hw, hl = gw / 2.0, gl / 2.0
        corners = np.array([
            [ ca * hw - sa * hl,  sa * hw + ca * hl],
            [-ca * hw - sa * hl, -sa * hw + ca * hl],
            [-ca * hw + sa * hl, -sa * hw - ca * hl],
            [ ca * hw + sa * hl,  sa * hw - ca * hl],
        ], dtype=np.float32)
        corners[:, 0] += gx
        corners[:, 1] += gy
        pts = corners.reshape(1, 4, 2).astype(np.int32)
        cv2.fillPoly(grid[cls_idx], pts, 1)

    return grid


def _get_gt_boxes_lidar(nusc: NuScenes, sample_token: str,
                        lidar2global: np.ndarray) -> list:
    """
    Returns list of (cx, cy, w, l, yaw, class_idx) in the LiDAR frame.
    """
    sample   = nusc.get('sample', sample_token)
    g2l      = np.linalg.inv(lidar2global)
    q_g2l    = Quaternion(matrix=g2l[:3, :3])
    yaw_g2l  = q_g2l.yaw_pitch_roll[0]

    boxes = []
    for ann_token in sample['anns']:
        ann      = nusc.get('sample_annotation', ann_token)
        cat_name = ann['category_name']
        if cat_name not in NUSC_CAT_MAP:
            continue
        cls_name = NUSC_CAT_MAP[cat_name]
        cls_idx  = CLASS_IDX[cls_name]

        center_global = np.array(ann['translation'] + [1.0])
        center_lidar  = (g2l @ center_global)[:3]
        cx, cy        = center_lidar[0], center_lidar[1]

        w, l, _       = ann['size']          # nuScenes: [width, length, height]
        yaw_global    = Quaternion(ann['rotation']).yaw_pitch_roll[0]
        yaw_lidar     = yaw_global + yaw_g2l

        boxes.append((cx, cy, w, l, yaw_lidar, cls_idx))
    return boxes


def compute_bev_miou(nusc: NuScenes, val_tokens: list[str],
                     all_preds: dict[str, list[dict]]) -> dict:
    """
    Computes BEV occupancy mIoU per class.

    Rasterises GT and predicted boxes on the BEV grid for every val sample
    and accumulates pixel-level intersection / union counts per class.

    Returns dict with per-class IoU and mean IoU.
    """
    nc           = len(CLASS_NAMES)
    intersect    = np.zeros(nc, dtype=np.float64)
    union        = np.zeros(nc, dtype=np.float64)

    for tok in val_tokens:
        l2g, yaw_total = get_lidar2global(nusc, tok)

        # GT boxes → LiDAR frame
        gt_boxes   = _get_gt_boxes_lidar(nusc, tok, l2g)
        gt_grid    = _rasterize_boxes(gt_boxes, PC_RANGE, BEV_H, BEV_W)

        # Predicted boxes (already in LiDAR frame)
        dets       = all_preds.get(tok, [])
        pred_boxes = []
        for d in dets:
            x, y, _, w, l, _, yaw = d['box3d']
            pred_boxes.append((x, y, w, l, yaw, d['label']))
        pred_grid  = _rasterize_boxes(pred_boxes, PC_RANGE, BEV_H, BEV_W)

        for ci in range(nc):
            g = gt_grid[ci].astype(bool)
            p = pred_grid[ci].astype(bool)
            intersect[ci] += float((g & p).sum())
            union[ci]     += float((g | p).sum())

    iou_per_class = {}
    valid_ious    = []
    for ci, name in enumerate(CLASS_NAMES):
        if union[ci] > 0:
            iou = intersect[ci] / union[ci]
        else:
            iou = float('nan')
        iou_per_class[name] = iou
        if not np.isnan(iou):
            valid_ious.append(iou)

    iou_per_class['mIoU'] = float(np.mean(valid_ious)) if valid_ious else float('nan')
    return iou_per_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default='model/checkpoints/bevformer_tiny_fp16_epoch_24.pth')
    parser.add_argument('--dataroot',
                        default='/Users/trish/Downloads/nuScenes_miniV1.0')
    parser.add_argument('--score-thr', type=float, default=0.1,
                        help='Detection score threshold')
    parser.add_argument('--max-per-sample', type=int, default=300)
    parser.add_argument('--out-dir', default='eval_results')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Run entirely on CPU to rule out MPS numerical issues')
    parser.add_argument('--max-total-frames', type=int, default=0,
                        help='Stop after this many frames total (0 = no limit).')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Device -----------------------------------------------------------
    if args.force_cpu:
        device = torch.device('cpu')
        print('[INFO] --force-cpu: overriding device to CPU')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'[INFO] device : {device}')

    # ---- Model ------------------------------------------------------------
    # BEVFormerTiny.__init__ auto-selects MPS/CUDA; if --force-cpu we move
    # it back to CPU and patch model.device so all internal .to(self.device)
    # calls in forward() also land on CPU.
    model = BEVFormerTiny(pretrained_backbone=False)
    model.eval()
    load_checkpoint(model, args.checkpoint)
    if args.force_cpu:
        model.to('cpu')
        model.device = torch.device('cpu')
        print('[INFO] model moved to CPU')
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[INFO] model  : {n_params:.1f} M params')

    # ---- nuScenes init ----------------------------------------------------
    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)

    # Use ALL mini scenes (all 10 scenes, ~404 samples).
    # NuScenesEval will later filter to mini_val for the official mAP metric.
    splits      = create_splits_scenes()
    mini_val_scenes   = set(splits.get('mini_val',   []))
    mini_train_scenes = set(splits.get('mini_train', []))
    all_scenes_set    = mini_val_scenes | mini_train_scenes

    # Map scene name → scene index in the mini dataset
    scene_name_to_idx = {s['name']: i for i, s in enumerate(nusc.scene)}

    # ---- Collect all sample tokens (all 10 mini scenes) -------------------
    val_tokens: list[str] = []
    for scene in nusc.scene:
        tok = scene['first_sample_token']
        while tok:
            val_tokens.append(tok)
            tok = nusc.get('sample', tok)['next'] or None

    n_val_tok  = sum(1 for tok in val_tokens
                     if nusc.get('sample', tok) and
                     nusc.get('scene', nusc.get('sample', tok)['scene_token'])['name']
                     in mini_val_scenes)
    print(f'[INFO] all mini samples : {len(val_tokens)}  '
          f'(mini_val subset: {n_val_tok})')

    # ---- Inference --------------------------------------------------------
    loader = NuScenesMiniLoader(args.dataroot)

    all_preds: dict[str, list[dict]] = {t: [] for t in val_tokens}
    processed_tokens: list[str] = []   # tokens for which we actually ran inference

    frame_limit = args.max_total_frames  # 0 = no limit
    if frame_limit:
        print(f'[INFO] frame limit : {frame_limit} total frames')

    print('\n[INFO] running inference on all mini scenes...')
    total_frames = 0
    total_ms     = 0.0
    done         = False   # early-exit flag shared across both loop levels

    with torch.no_grad():
        for scene in nusc.scene:
            if done:
                break
            scene_idx       = scene_name_to_idx[scene['name']]
            prev_bev        = None
            scene_frame_idx = 0

            for sample in loader.iter_scene(scene_idx=scene_idx):
                if done:
                    break
                sample_token = sample['img_metas'][0]['sample_token']

                imgs      = sample['imgs']
                img_metas = sample['img_metas']

                t0 = time.perf_counter()
                out = model(imgs, img_metas, prev_bev=prev_bev)
                if device.type == 'mps':
                    torch.mps.synchronize()
                elif device.type == 'cuda':
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000

                # NaN/Inf guard — print which tensor is bad and its finite value range
                _bad = False
                for _key in ('cls_logits', 'reg_preds', 'ref_pts'):
                    _t = out[_key]
                    _n = int(torch.isnan(_t).sum())
                    _i = int(torch.isinf(_t).sum())
                    if _n or _i:
                        _fin = _t[_t.isfinite()]
                        _rng = (f'[{float(_fin.min()):.4f}, {float(_fin.max()):.4f}]'
                                if _fin.numel() else 'all-bad')
                        print(f'  [WARN] frame {total_frames} | {_key}: '
                              f'{_n} NaN  {_i} Inf  finite-range={_rng}')
                        _bad = True
                if _bad:
                    print(f'  [WARN] frame {total_frames} — dropping prev_bev, skipping')
                    prev_bev = None
                    total_frames += 1
                    if frame_limit and total_frames >= frame_limit:
                        done = True
                    continue

                prev_bev = out['bev_feat'].detach()

                dets = decode_predictions(
                    out['cls_logits'][0].cpu(),
                    out['reg_preds'][0].cpu(),
                    out['ref_pts'][0].cpu(),
                    PC_RANGE,
                    score_thr=args.score_thr,
                    max_num=args.max_per_sample,
                )
                all_preds[sample_token] = dets
                processed_tokens.append(sample_token)

                scene_frame_idx += 1
                total_frames    += 1
                total_ms        += elapsed_ms
                det_str = f'{len(dets):3d} dets'
                print(f'  {scene["name"]} | frame {total_frames:3d} | '
                      f'{elapsed_ms:6.1f} ms | {det_str} | {sample_token[:8]}')

                if frame_limit and total_frames >= frame_limit:
                    done = True

    avg_ms = total_ms / max(total_frames, 1)
    partial = done and bool(frame_limit)
    print(f'\n[INFO] processed {total_frames} frames'
          + (f'  (partial run — {frame_limit}-frame limit)' if partial else '')
          + f', avg {avg_ms:.1f} ms/frame')

    # ---- BEV occupancy mIoU -----------------------------------------------
    # Scoped to processed_tokens only so partial runs get a valid number.
    miou_scope     = processed_tokens
    miou_scope_lbl = (f'{len(miou_scope)} processed frames'
                      if partial else 'all 10 scenes')
    print(f'\n{"=" * 60}')
    print(f'BEV Occupancy mIoU  (50×50 grid, 2.048 m/cell)  — {miou_scope_lbl}')
    print('=' * 60)

    bev_miou = compute_bev_miou(nusc, miou_scope, all_preds)

    print()
    for name in CLASS_NAMES:
        v = bev_miou[name]
        bar = ('█' * int(v * 30) if not np.isnan(v) else '  N/A')
        val_str = f'{v:.4f}' if not np.isnan(v) else ' N/A '
        print(f'  {name:<25s}: {val_str}  {bar}')
    print(f'\n  mIoU  : {bev_miou["mIoU"]:.4f}')

    # ---- nuScenes detection evaluation — skipped on partial runs ----------
    # NuScenesEval requires all tokens for the split to be present; a partial
    # run (--max-total-frames) does not satisfy this, so we skip it.
    all_metrics: dict = {}
    if partial:
        print(f'\n[INFO] Skipping NuScenesEval — partial run '
              f'({total_frames}/{len(val_tokens)} tokens). '
              f'Re-run without --max-total-frames for full mAP.')
    else:
        # Build per-token submission records
        all_records: dict[str, tuple[str, list]] = {}
        for tok in val_tokens:
            sn = nusc.get('scene', nusc.get('sample', tok)['scene_token'])['name']
            l2g, yaw_total = get_lidar2global(nusc, tok)
            records = preds_to_nuscenes(all_preds[tok], tok, l2g, yaw_total)
            all_records[tok] = (sn, records)

        meta = {
            'use_camera':   True, 'use_lidar':    False,
            'use_radar':    False, 'use_map':      False, 'use_external': True,
        }

        def _write_submission(scene_set: set, filename: str) -> str:
            rd = {tok: recs for tok, (sn, recs) in all_records.items()
                  if sn in scene_set}
            path = os.path.join(args.out_dir, filename)
            with open(path, 'w') as f:
                json.dump({'meta': meta, 'results': rd}, f)
            return path

        path_val   = _write_submission(mini_val_scenes,   'results_mini_val.json')
        path_train = _write_submission(mini_train_scenes, 'results_mini_train.json')

        import shutil
        shutil.copy(path_val, os.path.join(args.out_dir, 'nuscenes_results.json'))

        def _run_nusc_eval(result_path: str, eval_set: str, out_subdir: str):
            os.makedirs(out_subdir, exist_ok=True)
            ev = NuScenesEval(
                nusc,
                config=config_factory('detection_cvpr_2019'),
                result_path=result_path,
                eval_set=eval_set,
                output_dir=out_subdir,
                verbose=False,
            )
            return ev.evaluate()[0]

        for eval_set, result_path in [('mini_val',   path_val),
                                       ('mini_train', path_train)]:
            print(f'\n{"=" * 60}')
            print(f'nuScenes Detection Evaluation — {eval_set}')
            print('=' * 60)
            m = _run_nusc_eval(result_path, eval_set,
                               os.path.join(args.out_dir, eval_set))
            all_metrics[eval_set] = m
            print(f'\n  NDS : {m.nd_score:.4f}   mAP : {m.mean_ap:.4f}')
            print('  Per-class AP:')
            for cls_name, ap in sorted(m.mean_dist_aps.items()):
                print(f'    {cls_name:<25s}: {ap:.4f}')

    # ---- Summary table -----------------------------------------------------
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    if all_metrics:
        for split_name, m in all_metrics.items():
            print(f'  [{split_name}]  NDS: {m.nd_score:.4f}   mAP: {m.mean_ap:.4f}')
    else:
        print('  [NuScenesEval]  skipped (partial run)')
    print(f'  mIoU (BEV, {miou_scope_lbl}) : {bev_miou["mIoU"]:.4f}')
    print(f'  Avg latency : {avg_ms:.1f} ms/frame  ({1000/avg_ms:.1f} FPS)')
    print(f'  Device      : {device}  |  frames processed: {total_frames}')
    print()

    # ---- Save summary JSON -------------------------------------------------
    summary: dict = {
        'device':         str(device),
        'frames_processed': total_frames,
        'partial_run':    partial,
        'mIoU_bev':       bev_miou['mIoU'],
        'per_class_mIoU': {k: (float(v) if not np.isnan(v) else None)
                           for k, v in bev_miou.items()},
        'avg_latency_ms': avg_ms,
    }
    if all_metrics:
        for split_name, m in all_metrics.items():
            summary[split_name] = {
                'NDS': m.nd_score,
                'mAP': m.mean_ap,
                'per_class_AP': m.mean_dist_aps,
            }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[INFO] summary saved → {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
