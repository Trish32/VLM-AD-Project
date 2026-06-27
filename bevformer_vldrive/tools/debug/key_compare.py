#!/usr/bin/env python3
"""
Compare official BEVFormer-tiny checkpoint keys against our model's state_dict.

Prints:
  1. Mapped keys             — checkpoint key → model key, with shape pair
  2. Shape mismatches        — mapped but tensor shapes differ
  3. Unmapped checkpoint keys — checkpoint keys that _build_remap ignores
  4. Uninitialised model keys — model keys that receive no checkpoint weight

Usage:
    python tools/key_compare.py \
        --checkpoint model/checkpoints/bevformer_tiny_fp16_epoch_24.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from model import BEVFormerTiny
from eval import _build_remap


def _section(title: str):
    print()
    print('=' * 70)
    print(f'  {title}')
    print('=' * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default='model/checkpoints/bevformer_tiny_fp16_epoch_24.pth')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load checkpoint and model
    # ------------------------------------------------------------------ #
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    raw  = ckpt.get('state_dict', ckpt)

    model = BEVFormerTiny(pretrained_backbone=False)
    model_sd = model.state_dict()

    print(f'Checkpoint keys : {len(raw)}')
    print(f'Model keys      : {len(model_sd)}')

    # ------------------------------------------------------------------ #
    # 2. Run remap
    # ------------------------------------------------------------------ #
    remap = _build_remap(raw)   # { model_key: tensor }

    # Invert remap: model_key → set of ckpt_keys that contributed
    # (build_remap is injective, one ckpt_key per model_key)
    ckpt_key_for_model: dict[str, str] = {}
    for ck, v in raw.items():
        # We need to find which model key each ckpt key ended up at.
        # Easiest: rebuild with tracking.
        pass

    # Rebuild with tracking (patch _take to record the mapping)
    mapping: list[tuple[str, str, tuple, tuple]] = []   # (ckpt_k, model_k, ckpt_shape, model_shape)
    shape_mismatches: list[tuple] = []

    for model_k, tensor in remap.items():
        model_shape = tuple(model_sd[model_k].shape) if model_k in model_sd else None
        ckpt_shape  = tuple(tensor.shape)
        if model_shape is None:
            shape_mismatches.append((None, model_k, ckpt_shape, None))
        elif ckpt_shape != model_shape:
            shape_mismatches.append((None, model_k, ckpt_shape, model_shape))

    # Find which raw key produced each remap entry by re-running with tracking
    tracked_mapping: list[tuple[str, str, tuple, tuple | None]] = []

    def _build_remap_tracked(raw_: dict):
        result: dict[str, torch.Tensor] = {}
        src_map: dict[str, str] = {}   # model_key → ckpt_key

        def _take(sk, dk, transform=None):
            if sk not in raw_:
                return
            v = raw_[sk].float()
            if transform is not None:
                v = transform(v)
            result[dk] = v
            src_map[dk] = sk

        for k in raw_:
            if k.startswith('img_backbone.'):
                dk = 'backbone.' + k[len('img_backbone.'):]
                result[dk] = raw_[k].float()
                src_map[dk] = k
            elif k.startswith('img_neck.'):
                dk = 'neck.' + k[len('img_neck.'):]
                result[dk] = raw_[k].float()
                src_map[dk] = k

        _take('pts_bbox_head.bev_embedding.weight',                 'bev_queries')
        _take('pts_bbox_head.positional_encoding.row_embed.weight', 'bev_pos_enc.row.weight')
        _take('pts_bbox_head.positional_encoding.col_embed.weight', 'bev_pos_enc.col.weight')
        _take('pts_bbox_head.transformer.cams_embeds',              'cams_embeds')
        _take('pts_bbox_head.transformer.level_embeds', 'level_embeds',
              transform=lambda v: v[:1])
        for suf in ('weight', 'bias'):
            _take(f'pts_bbox_head.transformer.can_bus_mlp.0.{suf}',      f'can_bus_mlp.0.{suf}')
            _take(f'pts_bbox_head.transformer.can_bus_mlp.2.{suf}',      f'can_bus_mlp.2.{suf}')
            _take(f'pts_bbox_head.transformer.can_bus_mlp.norm.{suf}',   f'can_bus_mlp.4.{suf}')
        _take('pts_bbox_head.query_embedding.weight',              'det_head.query_embed.weight')
        _take('pts_bbox_head.transformer.reference_points.weight', 'det_head.ref_points.weight')
        _take('pts_bbox_head.transformer.reference_points.bias',   'det_head.ref_points.bias')

        for i in range(3):
            src = f'pts_bbox_head.transformer.encoder.layers.{i}'
            dst = f'encoder.layers.{i}'
            for suf in ('weight', 'bias'):
                _take(f'{src}.attentions.0.sampling_offsets.{suf}',  f'{dst}.tsa.sampling_offsets.{suf}')
                _take(f'{src}.attentions.0.attention_weights.{suf}', f'{dst}.tsa.attention_weights.{suf}')
                _take(f'{src}.attentions.0.value_proj.{suf}',        f'{dst}.tsa.value_proj.{suf}')
                _take(f'{src}.attentions.0.output_proj.{suf}',       f'{dst}.tsa.output_proj.{suf}')
                _take(f'{src}.attentions.1.deformable_attention.sampling_offsets.{suf}', f'{dst}.sca.sampling_offsets.{suf}')
                _take(f'{src}.attentions.1.deformable_attention.attention_weights.{suf}',f'{dst}.sca.attention_weights.{suf}')
                _take(f'{src}.attentions.1.deformable_attention.value_proj.{suf}',       f'{dst}.sca.value_proj.{suf}')
                _take(f'{src}.attentions.1.output_proj.{suf}',       f'{dst}.sca.output_proj.{suf}')
                _take(f'{src}.ffns.0.layers.0.0.{suf}', f'{dst}.ffn.0.{suf}')
                _take(f'{src}.ffns.0.layers.1.{suf}',   f'{dst}.ffn.3.{suf}')
            for ni, nn_name in enumerate(['norm1', 'norm2', 'norm3']):
                for suf in ('weight', 'bias'):
                    _take(f'{src}.norms.{ni}.{suf}', f'{dst}.{nn_name}.{suf}')

        for i in range(6):
            src = f'pts_bbox_head.transformer.decoder.layers.{i}'
            dst = f'det_head.decoder_layers.{i}'
            for suf in ('weight', 'bias'):
                _take(f'{src}.attentions.0.attn.in_proj_{suf}', f'{dst}.self_attn.in_proj_{suf}')
            _take(f'{src}.attentions.0.attn.out_proj.weight', f'{dst}.self_attn.out_proj.weight')
            _take(f'{src}.attentions.0.attn.out_proj.bias',   f'{dst}.self_attn.out_proj.bias')
            for suf in ('weight', 'bias'):
                _take(f'{src}.attentions.1.sampling_offsets.{suf}',  f'{dst}.cross_attn.sampling_offsets.{suf}')
                _take(f'{src}.attentions.1.attention_weights.{suf}', f'{dst}.cross_attn.attention_weights.{suf}')
                _take(f'{src}.attentions.1.value_proj.{suf}',        f'{dst}.cross_attn.value_proj.{suf}')
                _take(f'{src}.attentions.1.output_proj.{suf}',       f'{dst}.cross_attn.output_proj.{suf}')
                _take(f'{src}.ffns.0.layers.0.0.{suf}', f'{dst}.linear1.{suf}')
                _take(f'{src}.ffns.0.layers.1.{suf}',   f'{dst}.linear2.{suf}')
            for ni, nn_name in enumerate(['norm1', 'norm2', 'norm3']):
                for suf in ('weight', 'bias'):
                    _take(f'{src}.norms.{ni}.{suf}', f'{dst}.{nn_name}.{suf}')
            for suf in ('weight', 'bias'):
                _take(f'pts_bbox_head.reg_branches.{i}.0.{suf}', f'det_head.reg_branches.{i}.0.{suf}')
                _take(f'pts_bbox_head.reg_branches.{i}.2.{suf}', f'det_head.reg_branches.{i}.2.{suf}')
                _take(f'pts_bbox_head.reg_branches.{i}.4.{suf}', f'det_head.reg_branches.{i}.4.{suf}')

        for idx in (0, 1, 3, 4, 6):
            for suf in ('weight', 'bias'):
                _take(f'pts_bbox_head.cls_branches.5.{idx}.{suf}', f'det_head.cls_branch.{idx}.{suf}')

        return result, src_map

    remap2, src_map = _build_remap_tracked(raw)

    # Build full mapping table
    for model_k, tensor in remap2.items():
        ckpt_k     = src_map.get(model_k, '<transform/prefix>')
        ckpt_shape = tuple(tensor.shape)
        model_shape = tuple(model_sd[model_k].shape) if model_k in model_sd else None
        tracked_mapping.append((ckpt_k, model_k, ckpt_shape, model_shape))

    # ------------------------------------------------------------------ #
    # 3. SECTION A — Successfully mapped keys
    # ------------------------------------------------------------------ #
    _section('A. MAPPED KEYS  (checkpoint → model, both shapes)')
    ok    = [(a, b, c, d) for a, b, c, d in tracked_mapping if d and c == d]
    print(f'  {len(ok)} keys mapped cleanly\n')
    for ckpt_k, model_k, cs, ms in ok:
        print(f'  {ckpt_k}')
        print(f'    → {model_k}   shape {list(cs)}')

    # ------------------------------------------------------------------ #
    # 4. SECTION B — Shape mismatches
    # ------------------------------------------------------------------ #
    _section('B. SHAPE MISMATCHES  (mapped but shapes differ → will be rejected)')
    bad = [(a, b, c, d) for a, b, c, d in tracked_mapping if d and c != d]
    if not bad:
        print('  None — all mapped keys have matching shapes.')
    for ckpt_k, model_k, cs, ms in bad:
        print(f'  MISMATCH  {ckpt_k}')
        print(f'    → {model_k}')
        print(f'    ckpt shape  : {list(cs)}')
        print(f'    model shape : {list(ms)}')

    # ------------------------------------------------------------------ #
    # 5. SECTION C — Model keys with no mapping
    # ------------------------------------------------------------------ #
    _section('C. MODEL KEYS WITH NO CHECKPOINT MAPPING  (random init)')
    mapped_model_keys = set(remap2.keys())
    uninit = sorted(k for k in model_sd if k not in mapped_model_keys)
    print(f'  {len(uninit)} model parameters are randomly initialised\n')
    for k in uninit:
        print(f'  {k}   shape {list(model_sd[k].shape)}')

    # ------------------------------------------------------------------ #
    # 6. SECTION D — Checkpoint keys that go unused
    # ------------------------------------------------------------------ #
    _section('D. CHECKPOINT KEYS NOT USED BY _build_remap')
    used_ckpt_keys = set(src_map.values())
    unused = sorted(k for k in raw if k not in used_ckpt_keys)
    print(f'  {len(unused)} checkpoint keys are ignored\n')

    # Group by prefix for readability
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for k in unused:
        parts = k.split('.')
        prefix = '.'.join(parts[:3]) if len(parts) >= 3 else k
        by_prefix[prefix].append(k)

    for prefix in sorted(by_prefix):
        keys = by_prefix[prefix]
        print(f'  [{prefix}]  ({len(keys)} keys)')
        for k in keys[:6]:   # cap at 6 per group
            shape = list(raw[k].shape)
            print(f'    {k}   {shape}')
        if len(keys) > 6:
            print(f'    ... and {len(keys) - 6} more')

    # ------------------------------------------------------------------ #
    # 7. Quick stats
    # ------------------------------------------------------------------ #
    _section('SUMMARY')
    n_ok    = len(ok)
    n_bad   = len(bad)
    n_uninit= len(uninit)
    n_unused= len(unused)
    print(f'  Mapped + matching shape : {n_ok}')
    print(f'  Shape mismatches        : {n_bad}   ← these are rejected at load time')
    print(f'  Model keys unloaded     : {n_uninit}  ← random init')
    print(f'  Checkpoint keys unused  : {n_unused}')
    print()


if __name__ == '__main__':
    main()
