#!/usr/bin/env python3
"""
Deep probe of SCA internals: find the first operation that diverges between CPU and MPS.

Checks in order:
  1. point_sampling outputs (ref_cam, bev_mask)
  2. cam_indices (visibility mask as indices)
  3. v_proj (value projection)
  4. offsets / attn
  5. sampling_locs (ref + offsets)
  6. ms_deform_attn_core output
  7. slots after scatter-back
  8. final SCA output (after count normalisation + residual)

Usage:
    conda run -n simple_bev_vldrive python tools/probe_sca.py \
        --checkpoint model/checkpoints/bevformer_tiny_fp16_epoch_24.pth \
        --dataroot /Users/trish/Downloads/nuScenes_miniV1.0
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from model.bevformer_tiny import BEVFormerTiny, NUSCENES_IMG_MEAN, NUSCENES_IMG_STD
from model.deform_attn    import ms_deform_attn_core
from data import NuScenesMiniLoader
from tools.eval import _build_remap


# ── helpers ───────────────────────────────────────────────────────────────────

def _cossim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu().flatten()
    b = b.detach().float().cpu().flatten()
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max())

def _reldiff(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.detach().float().cpu() - b.detach().float().cpu()).abs().max()
    n = a.detach().float().cpu().abs().max().clamp(min=1e-8)
    return float(d / n)

def _report(tag: str, cpu_t: torch.Tensor, mps_t: torch.Tensor) -> None:
    cs = _cossim(cpu_t, mps_t)
    md = _maxdiff(cpu_t, mps_t)
    rd = _reldiff(cpu_t, mps_t)
    flag = "  *** DIVERGE ***" if cs < 0.9999 or rd > 1e-4 else ""
    print(f"  {tag:<40s}  cos={cs:.8f}  max_abs={md:.6f}  rel={rd:.6f}{flag}")


# ── Manual SCA forward on two devices ────────────────────────────────────────

def run_sca_staged(sca_module, query, value, reference_points_cam, bev_mask,
                   spatial_shapes, level_start_index, query_pos, device_tag):
    """
    Manually execute SCA step by step and return a dict of intermediate tensors.
    Everything expected to already be on the target device.
    """
    steps = {}

    B, L, _ = query.shape
    num_cams = value.shape[0]
    num_Z    = reference_points_cam.shape[3]

    identity = query

    if query_pos is not None:
        query = query + query_pos

    steps['query+pos'] = query.detach().cpu()

    slots = torch.zeros(B, L, sca_module.embed_dim,
                        device=query.device, dtype=query.dtype)

    cam_active = bev_mask.any(dim=-1)   # (num_cams, B, L)
    steps['bev_mask']   = bev_mask.detach().cpu().float()
    steps['cam_active'] = cam_active.detach().cpu().float()

    cam_indices = []
    for i in range(num_cams):
        idx = cam_active[i, 0].nonzero(as_tuple=False).squeeze(-1)
        cam_indices.append(idx)

    steps['cam_indices_sizes'] = torch.tensor([len(idx) for idx in cam_indices],
                                               dtype=torch.float32)

    max_len = max((len(idx) for idx in cam_indices), default=0)
    if max_len == 0:
        steps['slots'] = slots.detach().cpu()
        return steps

    q_rebatch   = query.new_zeros(B, num_cams, max_len, sca_module.embed_dim)
    ref_rebatch = reference_points_cam.new_zeros(B, num_cams, max_len, num_Z, 2)
    for j in range(B):
        for i, idx in enumerate(cam_indices):
            if len(idx) == 0:
                continue
            q_rebatch[j, i,   :len(idx)] = query[j, idx]
            ref_rebatch[j, i, :len(idx)] = reference_points_cam[i, j, idx]

    steps['q_rebatch']   = q_rebatch.detach().cpu()
    steps['ref_rebatch'] = ref_rebatch.detach().cpu()

    q_flat   = q_rebatch.view(B * num_cams, max_len, sca_module.embed_dim)
    ref_flat = ref_rebatch.view(B * num_cams, max_len, num_Z, 2)

    v_flat = value.permute(2, 0, 1, 3).reshape(B * num_cams, -1, sca_module.embed_dim)
    v_proj = sca_module.value_proj(v_flat)
    steps['v_proj'] = v_proj.detach().cpu()

    v_proj = v_proj.view(B * num_cams, v_proj.shape[1], sca_module.num_heads, sca_module.head_dim)

    offsets = sca_module.sampling_offsets(q_flat).view(
        B * num_cams, max_len, sca_module.num_heads, sca_module.num_levels, sca_module.num_points, 2
    )
    steps['offsets'] = offsets.detach().cpu()

    attn = sca_module.attention_weights(q_flat).view(
        B * num_cams, max_len, sca_module.num_heads, sca_module.num_levels * sca_module.num_points
    ).softmax(-1).view(
        B * num_cams, max_len, sca_module.num_heads, sca_module.num_levels, sca_module.num_points
    )
    steps['attn'] = attn.detach().cpu()

    offset_normalizer = torch.stack(
        [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
    ).to(query.device)

    num_Z_actual = num_Z
    pts_per_z = sca_module.num_points // num_Z_actual

    ref_exp = ref_flat[:, :, None, None, None, :, :]
    offsets_r = offsets / offset_normalizer[None, None, None, :, None, :]
    offsets_r = offsets_r.view(
        B * num_cams, max_len, sca_module.num_heads, sca_module.num_levels, pts_per_z, num_Z_actual, 2
    )

    sampling_locs = ref_exp + offsets_r
    sampling_locs = sampling_locs.view(
        B * num_cams, max_len, sca_module.num_heads, sca_module.num_levels, sca_module.num_points, 2
    )
    steps['sampling_locs'] = sampling_locs.detach().cpu()

    # Run ms_deform_attn_core on CPU regardless of current device
    # to isolate whether grid_sample is the issue
    v_cpu   = v_proj.detach().cpu()
    sp_cpu  = spatial_shapes.detach().cpu()
    sl_cpu  = sampling_locs.detach().cpu()
    at_cpu  = attn.detach().cpu()
    steps[f'deform_attn_out_{device_tag}_forced_cpu'] = ms_deform_attn_core(
        v_cpu, sp_cpu, sl_cpu, at_cpu
    ).detach().cpu()

    # Run on native device
    out_native = ms_deform_attn_core(v_proj, spatial_shapes, sampling_locs, attn)
    steps['deform_attn_out'] = out_native.detach().cpu()

    out = out_native.view(B, num_cams, max_len, sca_module.embed_dim)
    for j in range(B):
        for i, idx in enumerate(cam_indices):
            if len(idx) > 0:
                slots[j, idx] += out[j, i, :len(idx)]

    steps['slots_before_norm'] = slots.detach().cpu()

    count = cam_active.float().permute(1, 2, 0).sum(-1).clamp(min=1.0)
    slots_normed = slots / count.unsqueeze(-1)
    steps['slots_after_norm'] = slots_normed.detach().cpu()

    final = sca_module.dropout(sca_module.output_proj(slots_normed)) + identity
    steps['final'] = final.detach().cpu()

    return steps


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default='model/checkpoints/bevformer_tiny_fp16_epoch_24.pth')
    parser.add_argument('--dataroot',
                        default='/Users/trish/Downloads/nuScenes_miniV1.0')
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print('[ERROR] MPS not available')
        sys.exit(1)

    cpu = torch.device('cpu')
    mps = torch.device('mps')

    # ---- Load model on CPU --------------------------------------------------
    print('[probe_sca] loading model ...')
    model = BEVFormerTiny(pretrained_backbone=False)
    model.to('cpu')
    model.device = cpu
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    raw  = ckpt.get('state_dict', ckpt)
    remap = _build_remap(raw)
    result = model.load_state_dict(remap, strict=False)
    print(f'[probe_sca] loaded {len(remap)} keys  missing={len(result.missing_keys)}')

    # ---- Load one frame -------------------------------------------------------
    loader = NuScenesMiniLoader(args.dataroot)
    sample = next(iter(loader.iter_scene(scene_idx=0)))
    imgs_cpu  = sample['imgs'].float()
    img_metas = sample['img_metas']
    print(f'[probe_sca] frame shape: {list(imgs_cpu.shape)}')

    # ---- Prepare feature map (backbone + neck) — identical on CPU & MPS ------
    B, N, _, H, W = imgs_cpu.shape
    mean = NUSCENES_IMG_MEAN[None, None, :, None, None]
    std  = NUSCENES_IMG_STD [None, None, :, None, None]
    imgs_norm_cpu = (imgs_cpu.float() - mean) / std
    imgs_flat_cpu = imgs_norm_cpu.view(B * N, 3, H, W)

    with torch.no_grad():
        feat_bb  = model.backbone(imgs_flat_cpu)
        feat_cpu = model.neck(feat_bb)         # (B*N, 256, 15, 25)

    _, C, Hf, Wf = feat_cpu.shape
    feat_v = feat_cpu.view(B, N, C, Hf, Wf)
    feat_v = feat_v + model.cams_embeds[None, :, :, None, None]
    feat_v = feat_v + model.level_embeds.view(1, 1, -1, 1, 1)
    feat_flat = feat_v.permute(1, 0, 2, 3, 4).flatten(3).permute(0, 3, 1, 2)
    # (N, S, B, C)

    spatial_shapes    = torch.tensor([[Hf, Wf]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)

    # ---- BEV queries + pos enc -----------------------------------------------
    can_bus  = torch.from_numpy(
        np.stack([m['can_bus'] for m in img_metas]).astype(np.float32)
    )
    bev_q = (model.bev_queries.unsqueeze(1).expand(-1, B, -1)
             + model.can_bus_mlp(can_bus).unsqueeze(0))          # (L, B, C)
    bev_pos = model.bev_pos_enc(model.BEV_H, model.BEV_W, cpu).flatten(2).permute(0, 2, 1)
    # (1, L, C)

    bev_q_batch = bev_q.permute(1, 0, 2)   # (B, L, C)

    # ---- point_sampling on CPU & MPS -----------------------------------------
    enc = model.encoder
    ref_3d = enc.get_reference_points(
        enc.bev_h, enc.bev_w, Z=enc.pc_range[5]-enc.pc_range[2],
        num_z=enc.num_z, dim='3d', bs=B, device=cpu, dtype=torch.float32
    )

    with torch.no_grad():
        ref_cam_cpu, bev_mask_cpu = enc.point_sampling(ref_3d, img_metas, cpu)

    ref_3d_mps = ref_3d.to(mps)
    with torch.no_grad():
        ref_cam_mps, bev_mask_mps = enc.point_sampling(ref_3d_mps, img_metas, mps)

    print('\n=== point_sampling comparison ===')
    _report('ref_cam',  ref_cam_cpu,  ref_cam_mps)
    print(f'  bev_mask agreement: {(bev_mask_cpu == bev_mask_mps.cpu()).all().item()}')
    n_diff_mask = int((bev_mask_cpu != bev_mask_mps.cpu()).sum())
    print(f'  bev_mask differing bits: {n_diff_mask} / {bev_mask_cpu.numel()}')

    # ---- Run SCA layer 0 on both devices ----------------------------------------
    sca = model.encoder.layers[0].sca

    # CPU run
    with torch.no_grad():
        steps_cpu = run_sca_staged(
            sca, bev_q_batch.clone(), feat_flat.clone(),
            ref_cam_cpu, bev_mask_cpu,
            spatial_shapes, level_start_index,
            bev_pos, device_tag='cpu'
        )

    # MPS run — move everything to MPS
    model.to(mps)
    bev_q_mps    = bev_q_batch.to(mps)
    feat_flat_mps = feat_flat.to(mps)
    ref_cam_mps_d  = ref_cam_mps
    bev_mask_mps_d = bev_mask_mps
    bev_pos_mps    = bev_pos.to(mps)
    ss_mps = spatial_shapes.to(mps)
    lsi_mps = level_start_index.to(mps)

    with torch.no_grad():
        steps_mps = run_sca_staged(
            sca, bev_q_mps, feat_flat_mps,
            ref_cam_mps_d, bev_mask_mps_d,
            ss_mps, lsi_mps,
            bev_pos_mps, device_tag='mps'
        )
    model.to(cpu)

    print('\n=== SCA layer 0 step-by-step comparison ===')
    common_keys = [k for k in steps_cpu if k in steps_mps
                   and not k.startswith('deform_attn_out_')]
    for k in common_keys:
        _report(k, steps_cpu[k], steps_mps[k])

    # Compare forced-CPU deform_attn vs native per device
    print('\n=== deform_attn forced-CPU vs native ===')
    key_c = 'deform_attn_out_cpu_forced_cpu'
    key_m = 'deform_attn_out_mps_forced_cpu'
    if key_c in steps_cpu and key_m in steps_mps:
        _report('CPU-native vs MPS-forced-cpu', steps_cpu['deform_attn_out'],
                steps_mps[key_m])
        _report('CPU-native vs CPU-native(mps)',
                steps_cpu['deform_attn_out'], steps_cpu[key_c])
        _report('MPS-native vs MPS-forced-cpu',
                steps_mps['deform_attn_out'], steps_mps[key_m])

    print('\n[probe_sca] done.')


if __name__ == '__main__':
    main()
