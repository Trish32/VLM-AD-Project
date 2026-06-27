#!/usr/bin/env python3
"""
Staged CPU vs MPS divergence probe.

Runs an identical forward pass on CPU and MPS, capturing activations at each
major module boundary:
  1. After _normalize
  2. After backbone
  3. After neck
  4. After cam/level embed addition
  5. After BEV encoder
  6. After det_head (cls_logits, reg_preds, ref_pts)

At each stage, prints cosine-similarity and max-abs-difference to identify
exactly where CPU and MPS outputs first diverge significantly.

Usage:
    conda run -n simple_bev_vldrive python tools/probe_mps.py \
        --checkpoint model/checkpoints/bevformer_tiny_fp16_epoch_24.pth \
        --dataroot /Users/trish/Downloads/nuScenes_miniV1.0 \
        [--scene 0]  [--frame 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from model.bevformer_tiny import BEVFormerTiny, NUSCENES_IMG_MEAN, NUSCENES_IMG_STD
from data import NuScenesMiniLoader
from tools.eval import _build_remap


# ── helpers ──────────────────────────────────────────────────────────────────

def _cossim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu().flatten()
    b = b.detach().float().cpu().flatten()
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float().cpu() - b.float().cpu()).abs().max())


def _reldiff(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a.float().cpu() - b.float().cpu()).abs().max()
    den = a.float().cpu().abs().max().clamp(min=1e-8)
    return float(num / den)


def _report(tag: str, cpu_t: torch.Tensor, mps_t: torch.Tensor) -> None:
    cs  = _cossim(cpu_t, mps_t)
    md  = _maxdiff(cpu_t, mps_t)
    rd  = _reldiff(cpu_t, mps_t)
    fin_cpu = cpu_t.isfinite().all().item()
    fin_mps = mps_t.isfinite().all().item()
    flag = "  *** DIVERGE ***" if cs < 0.99 or md > 1.0 else ""
    print(f"  {tag:<35s}  cos={cs:.6f}  max_abs={md:.4f}  rel={rd:.4f}"
          f"  finite(cpu={fin_cpu}, mps={fin_mps}){flag}")


# ── staged forward ────────────────────────────────────────────────────────────

def staged_forward(model: BEVFormerTiny, imgs_cpu: torch.Tensor,
                   img_metas: list) -> dict:
    """
    Runs each major stage on both CPU and MPS, storing (cpu_out, mps_out) pairs.
    model must be on CPU; internally we mirror to MPS where needed.
    """
    assert not torch.backends.mps.is_available() or True  # MPS may be available
    mps = torch.device('mps')
    cpu = torch.device('cpu')

    stages: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    B, N, _, H, W = imgs_cpu.shape

    # ---- Stage 1: normalize ------------------------------------------------
    mean = NUSCENES_IMG_MEAN[None, None, :, None, None]   # (1,1,3,1,1)
    std  = NUSCENES_IMG_STD [None, None, :, None, None]
    imgs_norm_cpu = (imgs_cpu.float() - mean) / std       # on CPU

    imgs_norm_mps = (imgs_cpu.float().to(mps) - mean.to(mps)) / std.to(mps)

    stages['1_normalize'] = (imgs_norm_cpu, imgs_norm_mps.cpu())

    # ---- Stage 2: backbone -------------------------------------------------
    model_cpu = model
    model_cpu.eval()

    imgs_flat_cpu = imgs_norm_cpu.view(B * N, 3, H, W)
    with torch.no_grad():
        bb_cpu = model_cpu.backbone(imgs_flat_cpu)  # (B*N, 2048, Hb, Wb)

    # Move model to MPS, run, move back
    model.to(mps)
    imgs_flat_mps = imgs_norm_mps.view(B * N, 3, H, W)
    with torch.no_grad():
        bb_mps = model.backbone(imgs_flat_mps)
    model.to(cpu)

    stages['2_backbone'] = (bb_cpu.cpu(), bb_mps.cpu())

    # ---- Stage 3: FPN neck -------------------------------------------------
    with torch.no_grad():
        neck_cpu = model_cpu.neck(bb_cpu)  # (B*N, 256, Hf, Wf)

    model.to(mps)
    with torch.no_grad():
        neck_mps = model.neck(bb_mps)
    model.to(cpu)

    stages['3_neck'] = (neck_cpu.cpu(), neck_mps.cpu())

    # ---- Stage 4: cam/level embed addition ---------------------------------
    _, C, Hf, Wf = neck_cpu.shape
    feat_cpu = neck_cpu.view(B, N, C, Hf, Wf)
    feat_cpu = feat_cpu + model_cpu.cams_embeds[None, :, :, None, None]
    feat_cpu = feat_cpu + model_cpu.level_embeds.view(1, 1, -1, 1, 1)
    feat_flat_cpu = feat_cpu.permute(1, 0, 2, 3, 4).flatten(3).permute(0, 3, 1, 2)
    # (N, Hf*Wf, B, C)

    neck_mps2 = neck_mps.to(mps)
    feat_mps = neck_mps2.view(B, N, C, Hf, Wf)
    feat_mps = feat_mps + model_cpu.cams_embeds.to(mps)[None, :, :, None, None]
    feat_mps = feat_mps + model_cpu.level_embeds.to(mps).view(1, 1, -1, 1, 1)
    feat_flat_mps = feat_mps.permute(1, 0, 2, 3, 4).flatten(3).permute(0, 3, 1, 2)

    stages['4_feat_embeds'] = (feat_flat_cpu.cpu(), feat_flat_mps.cpu())

    # ---- Stage 5: BEV encoder ----------------------------------------------
    spatial_shapes    = torch.tensor([[Hf, Wf]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)

    bev_q   = model_cpu.bev_queries.unsqueeze(1).expand(-1, B, -1)
    bev_pos = model_cpu.bev_pos_enc(model_cpu.BEV_H, model_cpu.BEV_W, cpu)
    bev_pos = bev_pos.flatten(2).permute(0, 2, 1)  # (1, L, C)

    can_bus = torch.from_numpy(
        np.stack([m['can_bus'] for m in img_metas], axis=0).astype(np.float32)
    )
    bev_q_cpu = bev_q + model_cpu.can_bus_mlp(can_bus).unsqueeze(0)

    with torch.no_grad():
        bev_feat_cpu = model_cpu.encoder(
            bev_q_cpu, feat_flat_cpu,
            spatial_shapes, level_start_index,
            img_metas, prev_bev=None, shift=None, bev_pos=bev_pos,
        )

    # MPS encoder
    model.to(mps)
    bev_q_mps = (model.bev_queries.unsqueeze(1).expand(-1, B, -1)
                 + model.can_bus_mlp(can_bus.to(mps)).unsqueeze(0))
    bev_pos_mps = model.bev_pos_enc(model.BEV_H, model.BEV_W, mps)
    bev_pos_mps = bev_pos_mps.flatten(2).permute(0, 2, 1)

    with torch.no_grad():
        bev_feat_mps = model.encoder(
            bev_q_mps, feat_flat_mps,
            spatial_shapes.to(mps), level_start_index.to(mps),
            img_metas, prev_bev=None, shift=None, bev_pos=bev_pos_mps,
        )
    model.to(cpu)

    stages['5_bev_encoder'] = (bev_feat_cpu.cpu(), bev_feat_mps.cpu())

    # ---- Stage 6: detection head -------------------------------------------
    with torch.no_grad():
        head_cpu = model_cpu.det_head(bev_feat_cpu)

    model.to(mps)
    with torch.no_grad():
        head_mps = model.det_head(bev_feat_mps.to(mps))
    model.to(cpu)

    stages['6_cls_logits']  = (head_cpu['cls_logits'].cpu(), head_mps['cls_logits'].cpu())
    stages['6_reg_preds']   = (head_cpu['reg_preds'].cpu(),  head_mps['reg_preds'].cpu())

    return stages


# ── sub-module hooks for encoder layer-by-layer comparison ───────────────────

def probe_encoder_layers(model: BEVFormerTiny, imgs_cpu: torch.Tensor,
                         img_metas: list,
                         feat_flat_cpu: torch.Tensor,
                         feat_flat_mps: torch.Tensor) -> None:
    """Compare each encoder layer's output between CPU and MPS."""
    cpu = torch.device('cpu')
    mps = torch.device('mps')
    B = imgs_cpu.shape[0]

    # FPN C5 spatial size: IMG_H=480, IMG_W=800, ResNet stride=32 → 15×25
    Hf, Wf = 15, 25
    spatial_shapes    = torch.tensor([[Hf, Wf]], dtype=torch.long)

    layer_outputs_cpu = {}
    layer_outputs_mps = {}

    def make_hook(store: dict, lid: int):
        def hook(module, inp, out):
            store[lid] = out.detach().cpu() if isinstance(out, torch.Tensor) else out[0].detach().cpu()
        return hook

    # Register hooks on encoder layers AND their TSA/SCA sub-modules
    handles_cpu = []
    handles_mps = []

    tsa_cpu: dict = {}
    tsa_mps: dict = {}
    sca_cpu: dict = {}
    sca_mps: dict = {}

    for i, layer in enumerate(model.encoder.layers):
        handles_cpu.append(layer.register_forward_hook(make_hook(layer_outputs_cpu, i)))
        handles_cpu.append(layer.tsa.register_forward_hook(make_hook(tsa_cpu, i)))
        handles_cpu.append(layer.sca.register_forward_hook(make_hook(sca_cpu, i)))

    can_bus = torch.from_numpy(
        np.stack([m['can_bus'] for m in img_metas], axis=0).astype(np.float32)
    )

    level_start_index = torch.tensor([0], dtype=torch.long)

    bev_q_cpu = (model.bev_queries.unsqueeze(1).expand(-1, B, -1)
                 + model.can_bus_mlp(can_bus).unsqueeze(0))
    bev_pos_cpu = model.bev_pos_enc(model.BEV_H, model.BEV_W, cpu).flatten(2).permute(0, 2, 1)

    with torch.no_grad():
        _ = model.encoder(
            bev_q_cpu, feat_flat_cpu,
            spatial_shapes, level_start_index,
            img_metas, prev_bev=None, shift=None, bev_pos=bev_pos_cpu,
        )
    for h in handles_cpu:
        h.remove()

    # MPS encoder layer-by-layer
    model.to(mps)
    for i, layer in enumerate(model.encoder.layers):
        handles_mps.append(layer.register_forward_hook(make_hook(layer_outputs_mps, i)))
        handles_mps.append(layer.tsa.register_forward_hook(make_hook(tsa_mps, i)))
        handles_mps.append(layer.sca.register_forward_hook(make_hook(sca_mps, i)))

    bev_q_mps = (model.bev_queries.unsqueeze(1).expand(-1, B, -1)
                 + model.can_bus_mlp(can_bus.to(mps)).unsqueeze(0))
    bev_pos_mps = model.bev_pos_enc(model.BEV_H, model.BEV_W, mps).flatten(2).permute(0, 2, 1)

    with torch.no_grad():
        _ = model.encoder(
            bev_q_mps, feat_flat_mps.to(mps),
            spatial_shapes.to(mps), level_start_index.to(mps),
            img_metas, prev_bev=None, shift=None, bev_pos=bev_pos_mps,
        )
    for h in handles_mps:
        h.remove()
    model.to(cpu)

    print("\n  [Encoder layer-by-layer breakdown]")
    for i in range(len(model.encoder.layers)):
        if i in tsa_cpu and i in tsa_mps:
            _report(f'  layer[{i}].tsa', tsa_cpu[i], tsa_mps[i])
        if i in sca_cpu and i in sca_mps:
            _report(f'  layer[{i}].sca', sca_cpu[i], sca_mps[i])
        if i in layer_outputs_cpu and i in layer_outputs_mps:
            _report(f'  layer[{i}] (full)', layer_outputs_cpu[i], layer_outputs_mps[i])


# ── backbone sub-layer hooks ──────────────────────────────────────────────────

def probe_backbone_sublayers(model: BEVFormerTiny, imgs_norm_cpu: torch.Tensor) -> None:
    """Hook each ResNet block and BN layer to find first divergence inside backbone."""
    cpu = torch.device('cpu')
    mps = torch.device('mps')
    B, N, _, H, W = imgs_norm_cpu.shape
    imgs_flat_cpu = imgs_norm_cpu.view(B * N, 3, H, W)
    imgs_flat_mps = imgs_flat_cpu.to(mps)

    store_cpu: dict = {}
    store_mps: dict = {}

    tag_map = {}
    hooks_cpu = []
    hooks_mps = []

    backbone = model.backbone

    def reg(module, name: str):
        tag_map[id(module)] = name

        def hook_cpu(m, inp, out):
            store_cpu[tag_map[id(m)]] = out.detach().cpu() if isinstance(out, torch.Tensor) else out[0].detach().cpu()
        def hook_mps(m, inp, out):
            store_mps[tag_map[id(m)]] = out.detach().cpu() if isinstance(out, torch.Tensor) else out[0].detach().cpu()

        hooks_cpu.append(module.register_forward_hook(hook_cpu))
        hooks_mps.append(module.register_forward_hook(hook_mps))

    # Register on major sub-blocks
    reg(backbone.conv1,  'bb.conv1')
    reg(backbone.bn1,    'bb.bn1')
    for i, layer in enumerate(backbone.layer1):
        reg(layer, f'bb.layer1[{i}]')
    for i, layer in enumerate(backbone.layer2):
        reg(layer, f'bb.layer2[{i}]')
    for i, layer in enumerate(backbone.layer3):
        reg(layer, f'bb.layer3[{i}]')
    for i, layer in enumerate(backbone.layer4):
        reg(layer, f'bb.layer4[{i}]')

    with torch.no_grad():
        _ = backbone(imgs_flat_cpu)
    for h in hooks_cpu:
        h.remove()

    model.to(mps)
    with torch.no_grad():
        _ = model.backbone(imgs_flat_mps)
    for h in hooks_mps:
        h.remove()
    model.to(cpu)

    print("\n  [Backbone sub-layer breakdown — stops at first divergence]")
    found_diverge = False
    for tag in ['bb.conv1', 'bb.bn1',
                'bb.layer1[0]', 'bb.layer1[1]', 'bb.layer1[2]',
                'bb.layer2[0]', 'bb.layer2[1]', 'bb.layer2[2]', 'bb.layer2[3]',
                'bb.layer3[0]', 'bb.layer3[1]', 'bb.layer3[2]', 'bb.layer3[3]',
                'bb.layer3[4]', 'bb.layer3[5]',
                'bb.layer4[0]', 'bb.layer4[1]', 'bb.layer4[2]']:
        if tag not in store_cpu or tag not in store_mps:
            continue
        a, b = store_cpu[tag], store_mps[tag]
        cs = _cossim(a, b)
        md = _maxdiff(a, b)
        rd = _reldiff(a, b)
        flag = ""
        if cs < 0.999 or rd > 0.01:
            flag = "  *** FIRST DIVERGE ***"
            found_diverge = True
        print(f"    {tag:<25s}  cos={cs:.6f}  max_abs={md:.4f}  rel={rd:.4f}{flag}")
        if found_diverge:
            break


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default='model/checkpoints/bevformer_tiny_fp16_epoch_24.pth')
    parser.add_argument('--dataroot',
                        default='/Users/trish/Downloads/nuScenes_miniV1.0')
    parser.add_argument('--scene', type=int, default=0,
                        help='Scene index (0-based within mini dataset)')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index within scene (0-based)')
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print('[ERROR] MPS not available — this probe requires an Apple Silicon Mac.')
        sys.exit(1)

    # ---- Load model on CPU (we'll mirror to MPS per-stage) -----------------
    print('[probe] loading model on CPU ...')
    model = BEVFormerTiny(pretrained_backbone=False)
    # Override device to CPU so __init__ doesn't move to MPS
    model.to('cpu')
    model.device = torch.device('cpu')
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    raw  = ckpt.get('state_dict', ckpt)
    from tools.eval import _build_remap
    remap = _build_remap(raw)
    result = model.load_state_dict(remap, strict=False)
    print(f'[probe] loaded {len(remap)} keys  missing={len(result.missing_keys)}')

    # ---- Load one frame ----------------------------------------------------
    print(f'[probe] loading scene={args.scene}, frame={args.frame} ...')
    loader = NuScenesMiniLoader(args.dataroot)
    sample = None
    for fi, s in enumerate(loader.iter_scene(scene_idx=args.scene)):
        if fi == args.frame:
            sample = s
            break
    if sample is None:
        print('[ERROR] frame not found')
        sys.exit(1)

    imgs_cpu  = sample['imgs'].float()          # (1, 6, 3, H, W)
    img_metas = sample['img_metas']
    print(f'[probe] imgs shape: {list(imgs_cpu.shape)}')

    # ---- Staged forward ----------------------------------------------------
    print('\n=== Staged CPU vs MPS comparison ===\n')
    stages = staged_forward(model, imgs_cpu, img_metas)
    for tag, (c, m) in stages.items():
        _report(tag, c, m)

    # ---- Drill into backbone if stage 2 diverges ---------------------------
    mean = NUSCENES_IMG_MEAN[None, None, :, None, None]
    std  = NUSCENES_IMG_STD [None, None, :, None, None]
    imgs_norm_cpu = (imgs_cpu.float() - mean) / std

    cos2 = _cossim(*stages['2_backbone'])
    if cos2 < 0.9999:
        print('\n=== Backbone diverges — drilling into sub-layers ===')
        probe_backbone_sublayers(model, imgs_norm_cpu)
    else:
        print('\n[probe] Backbone OK — divergence is downstream of backbone')

    # ---- Drill into encoder if stage 5 diverges but not backbone/neck ------
    cos3 = _cossim(*stages['3_neck'])
    cos5 = _cossim(*stages['5_bev_encoder'])
    if cos3 > 0.9999 and cos5 < 0.99:
        print('\n=== Encoder diverges — comparing layer-by-layer ===')
        feat_flat_cpu = stages['4_feat_embeds'][0]
        feat_flat_mps = stages['4_feat_embeds'][1]
        # feat_flat is (N, S, B, C) — check shape
        B, N = imgs_cpu.shape[0], imgs_cpu.shape[1]
        Hf, Wf = 15, 25
        probe_encoder_layers(model, imgs_cpu, img_metas, feat_flat_cpu, feat_flat_mps)

    print('\n[probe] done.')


if __name__ == '__main__':
    main()
