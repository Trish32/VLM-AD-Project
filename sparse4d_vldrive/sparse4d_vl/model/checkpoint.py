"""
Checkpoint loading utilities for Sparse4D v1 / v2.

Maps official Sparse4D checkpoint keys (flat head.layers.* ModuleList,
img_backbone.*, img_neck.*) to our pure-PyTorch model's shared-weight
architecture (head_single.*, head_temporal.*, backbone.backbone.*, ...).

Reference checkpoint key layout
--------------------------------
Backbone / neck:
  img_backbone.*  →  backbone.backbone.*
  img_neck.*      →  backbone.neck.*

Instance bank:
  head.instance_bank.anchor           →  instance_bank.anchors
  head.instance_bank.instance_feature →  instance_bank.instance_feature

Anchor encoder (shared, loaded into both heads for v2):
  head.anchor_encoder.*  →  head_single.anchor_encoder.*
                          →  head_temporal.anchor_encoder.*

v2 operation_order flat layer indices:
  Stage 0 (single-frame): 0=deform, 1=ffn, 2=norm, 3=refine
  Stage k (temporal, k=1..5): 4+(k-1)*7 = temp_gnn, +1=gnn, +2=norm,
                               +3=deform_temporal, +4=ffn, +5=norm, +6=refine
  We use last temporal stage (k=5): layers 32-38.

v1 operation_order: ['gnn','norm','deformable','norm','ffn','norm','refine'] × 6
  Last stage (stage 5): 35=gnn, 36=norm, 37=deform, 38=norm, 39=ffn, 40=norm, 41=refine
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_remap(raw: dict, version: str = 'v2') -> dict:
    """
    Build a state-dict mapping from official checkpoint keys to our model keys.

    Parameters
    ----------
    raw     : raw checkpoint state_dict (from torch.load)
    version : 'v1' or 'v2'

    Returns
    -------
    dict { our_model_key : weight_tensor }
    """
    remap: dict[str, torch.Tensor] = {}

    def _take(src: str, dst: str) -> None:
        """
        Copy one exact key (used where the sub-structure differs, like FFN)
        """
        if src in raw:
            remap[dst] = raw[src].float()  # The .float() on every copy forces fp32 for MPS

    def _prefix(src_pfx: str, dst_pfx: str, exclude: str | None = None) -> None:
        """Copy all keys with src_pfx to dst_pfx, optionally skipping a sub-prefix.
         (used where our submodule mirrors the reference's)"""
        for k, v in raw.items():
            if k.startswith(src_pfx):
                if exclude and k.startswith(src_pfx + exclude):
                    continue
                suffix = k[len(src_pfx):]
                remap[dst_pfx + suffix] = v.float()  # The .float() on every copy forces fp32 for MPS

    # ------------------------------------------------------------------
    # Backbone + FPN neck (same key structure, different prefix)
    # ------------------------------------------------------------------
    _prefix('img_backbone.', 'backbone.backbone.')
    _prefix('img_neck.',     'backbone.neck.')

    # ------------------------------------------------------------------
    # Instance bank
    # ------------------------------------------------------------------
    _take('head.instance_bank.anchor',           'instance_bank.anchors')
    _take('head.instance_bank.instance_feature', 'instance_bank.instance_feature')

    # ------------------------------------------------------------------
    # v2-specific: flat 39-layer head.layers → head_single(stage 0) / head_temporal(stages 1–5)
    # ------------------------------------------------------------------
    if version == 'v2':
        # Anchor encoder — shared in checkpoint, copy to stage 0 head
        _prefix('head.anchor_encoder.', 'head_single.anchor_encoder.')

        # ----- Stage 0 → head_single (0=deform, 1=ffn, 2=norm, 3=refine) -----
        # Layer 0: DeformableFeatureAggregation (no temp_module)
        _prefix('head.layers.0.weights_fc.',                   'head_single.deform.weights_fc.')
        _prefix('head.layers.0.output_proj.',                  'head_single.deform.output_proj.')
        _prefix('head.layers.0.camera_encoder.',               'head_single.deform.camera_encoder.')
        _prefix('head.layers.0.kps_generator.learnable_fc.',   'head_single.deform.key_pts_gen.learnable_fc.')

        # Layer 1: AsymmetricFFN (reference keys: layers.0.0 is mmcv's Sequential(Linear, …) nesting → fc1, layers.1 → fc2)
        _take('head.layers.1.layers.0.0.weight', 'head_single.ffn.fc1.weight')
        _take('head.layers.1.layers.0.0.bias',   'head_single.ffn.fc1.bias')
        _take('head.layers.1.layers.1.weight',   'head_single.ffn.fc2.weight')
        _take('head.layers.1.layers.1.bias',     'head_single.ffn.fc2.bias')
        _prefix('head.layers.1.pre_norm.',       'head_single.ffn.pre_norm.')
        _prefix('head.layers.1.identity_fc.',    'head_single.ffn.identity_fc.')

        # Layer 2: LayerNorm → norms.0
        _prefix('head.layers.2.', 'head_single.norms.0.')

        # Layer 3: SparseBox3DRefinementModule
        _prefix('head.layers.3.layers.',     'head_single.refine.layers.')
        _prefix('head.layers.3.cls_layers.', 'head_single.refine.cls_layers.')

        # ----- Temporal stages 1-5 → head_temporal_stages.{0..4} -----
        # Checkpoint flat-list layout per temporal stage k (1-indexed):
        #   base = 4 + (k-1)*7  each temporal stage has 7, so 4 + 5×7 = 39 — exactly the flat count.
        #   base+0: temp_gnn, base+1: gnn, base+2: norm,
        #   base+3: deformable, base+4: ffn, base+5: norm, base+6: refine
        for k in range(1, 6):
            si  = k - 1                 # 0-indexed stage slot
            b   = 4 + (k - 1) * 7      # base layer index in checkpoint
            pfx = f'head_temporal_stages.{si}.'

            # Anchor encoder — one shared encoder in checkpoint, copy to each temporal stage's own anchor_encoder
            _prefix('head.anchor_encoder.', f'{pfx}anchor_encoder.')

            # temp_gnn
            _prefix(f'head.layers.{b}.', f'{pfx}temp_gnn.')

            # gnn
            _prefix(f'head.layers.{b+1}.', f'{pfx}gnn.')

            # norm → norms.0
            _prefix(f'head.layers.{b+2}.', f'{pfx}norms.0.')

            # DeformableFeatureAggregation (no temp_module in any stage)
            _prefix(f'head.layers.{b+3}.weights_fc.',                 f'{pfx}deform.weights_fc.')
            _prefix(f'head.layers.{b+3}.output_proj.',                f'{pfx}deform.output_proj.')
            _prefix(f'head.layers.{b+3}.camera_encoder.',             f'{pfx}deform.camera_encoder.')
            _prefix(f'head.layers.{b+3}.kps_generator.learnable_fc.', f'{pfx}deform.key_pts_gen.learnable_fc.')

            # AsymmetricFFN (layers.0.0 = fc1, layers.1 = fc2)
            _take(f'head.layers.{b+4}.layers.0.0.weight', f'{pfx}ffn.fc1.weight')
            _take(f'head.layers.{b+4}.layers.0.0.bias',   f'{pfx}ffn.fc1.bias')
            _take(f'head.layers.{b+4}.layers.1.weight',   f'{pfx}ffn.fc2.weight')
            _take(f'head.layers.{b+4}.layers.1.bias',     f'{pfx}ffn.fc2.bias')
            _prefix(f'head.layers.{b+4}.pre_norm.',        f'{pfx}ffn.pre_norm.')
            _prefix(f'head.layers.{b+4}.identity_fc.',     f'{pfx}ffn.identity_fc.')

            # norm → norms.1
            _prefix(f'head.layers.{b+5}.', f'{pfx}norms.1.')

            # SparseBox3DRefinementModule
            _prefix(f'head.layers.{b+6}.layers.',     f'{pfx}refine.layers.')
            _prefix(f'head.layers.{b+6}.cls_layers.', f'{pfx}refine.cls_layers.')

    # ------------------------------------------------------------------
    # v3-specific: same 39-layer flat layout as v2, plus
    #   - addition 1: decoupled attention: fc_before / fc_after (shared in ckpt → copied
    #     into every stage head), 512-dim gnn / temp_gnn
    #   - addition 2: v3 anchor encoder (pos_fc/size_fc/yaw_fc/vel_fc, cat mode)
    #   - addition 3: quality_layers in every refine module
    # ------------------------------------------------------------------
    elif version == 'v3':
        # Anchor encoder → stage 0 + every temporal stage (shared in ckpt)
        _prefix('head.anchor_encoder.', 'head_single.anchor_encoder.')

        # fc_before / fc_after → every head (shared pair in checkpoint)
        _prefix('head.fc_before.', 'head_single.fc_before.')
        _prefix('head.fc_after.',  'head_single.fc_after.')

        # ----- Stage 0 → head_single (layers 0-3) -----
        _prefix('head.layers.0.weights_fc.',                 'head_single.deform.weights_fc.')
        _prefix('head.layers.0.output_proj.',                'head_single.deform.output_proj.')
        _prefix('head.layers.0.camera_encoder.',             'head_single.deform.camera_encoder.')
        _prefix('head.layers.0.kps_generator.learnable_fc.', 'head_single.deform.key_pts_gen.learnable_fc.')
        _take('head.layers.0.kps_generator.fix_scale',       'head_single.deform.key_pts_gen.fix_scale')  # v2's fix_scale is a config constant absent from its checkpoint; v3 ships it trained

        _take('head.layers.1.layers.0.0.weight', 'head_single.ffn.fc1.weight')
        _take('head.layers.1.layers.0.0.bias',   'head_single.ffn.fc1.bias')
        _take('head.layers.1.layers.1.weight',   'head_single.ffn.fc2.weight')
        _take('head.layers.1.layers.1.bias',     'head_single.ffn.fc2.bias')
        _prefix('head.layers.1.pre_norm.',       'head_single.ffn.pre_norm.')
        _prefix('head.layers.1.identity_fc.',    'head_single.ffn.identity_fc.')

        _prefix('head.layers.2.', 'head_single.norms.0.')

        _prefix('head.layers.3.layers.',         'head_single.refine.layers.')
        _prefix('head.layers.3.cls_layers.',     'head_single.refine.cls_layers.')
        _prefix('head.layers.3.quality_layers.', 'head_single.refine.quality_layers.')

        # ----- Temporal stages 1-5 → head_temporal_stages.{0..4} -----
        for k in range(1, 6):
            si  = k - 1
            b   = 4 + (k - 1) * 7
            pfx = f'head_temporal_stages.{si}.'

            _prefix('head.anchor_encoder.', f'{pfx}anchor_encoder.')
            _prefix('head.fc_before.',      f'{pfx}fc_before.')
            _prefix('head.fc_after.',       f'{pfx}fc_after.')

            _prefix(f'head.layers.{b}.',   f'{pfx}temp_gnn.')
            _prefix(f'head.layers.{b+1}.', f'{pfx}gnn.')
            _prefix(f'head.layers.{b+2}.', f'{pfx}norms.0.')

            _prefix(f'head.layers.{b+3}.weights_fc.',                 f'{pfx}deform.weights_fc.')
            _prefix(f'head.layers.{b+3}.output_proj.',                f'{pfx}deform.output_proj.')
            _prefix(f'head.layers.{b+3}.camera_encoder.',             f'{pfx}deform.camera_encoder.')
            _prefix(f'head.layers.{b+3}.kps_generator.learnable_fc.', f'{pfx}deform.key_pts_gen.learnable_fc.')
            _take(f'head.layers.{b+3}.kps_generator.fix_scale',       f'{pfx}deform.key_pts_gen.fix_scale')

            _take(f'head.layers.{b+4}.layers.0.0.weight', f'{pfx}ffn.fc1.weight')
            _take(f'head.layers.{b+4}.layers.0.0.bias',   f'{pfx}ffn.fc1.bias')
            _take(f'head.layers.{b+4}.layers.1.weight',   f'{pfx}ffn.fc2.weight')
            _take(f'head.layers.{b+4}.layers.1.bias',     f'{pfx}ffn.fc2.bias')
            _prefix(f'head.layers.{b+4}.pre_norm.',        f'{pfx}ffn.pre_norm.')
            _prefix(f'head.layers.{b+4}.identity_fc.',     f'{pfx}ffn.identity_fc.')

            _prefix(f'head.layers.{b+5}.', f'{pfx}norms.1.')

            _prefix(f'head.layers.{b+6}.layers.',         f'{pfx}refine.layers.')
            _prefix(f'head.layers.{b+6}.cls_layers.',     f'{pfx}refine.cls_layers.')
            _prefix(f'head.layers.{b+6}.quality_layers.', f'{pfx}refine.quality_layers.')

    # ------------------------------------------------------------------
    # v1-specific: flat 42-layer head.layers → head.*
    # Last stage (stage 5): 35=gnn, 36=norm, 37=deform, 38=norm,
    #                        39=ffn, 40=norm, 41=refine
    # ------------------------------------------------------------------
    elif version == 'v1':
        _prefix('head.anchor_encoder.', 'head.anchor_encoder.')

        # gnn (layer 35)
        _prefix('head.layers.35.', 'head.gnn.')

        # norm (layer 36) → norms.0
        _prefix('head.layers.36.', 'head.norms.0.')

        # deformable (layer 37)
        _prefix('head.layers.37.weights_fc.',     'head.deform.weights_fc.')
        _prefix('head.layers.37.output_proj.',    'head.deform.output_proj.')
        _prefix('head.layers.37.camera_encoder.', 'head.deform.camera_encoder.')

        # norm (layer 38) → norms.1
        _prefix('head.layers.38.', 'head.norms.1.')

        # ffn (layer 39)
        _take('head.layers.39.layers.0.weight', 'head.ffn.fc1.weight')
        _take('head.layers.39.layers.0.bias',   'head.ffn.fc1.bias')
        _take('head.layers.39.layers.3.weight', 'head.ffn.fc2.weight')
        _take('head.layers.39.layers.3.bias',   'head.ffn.fc2.bias')

        # norm (layer 40) → norms.2
        _prefix('head.layers.40.', 'head.norms.2.')

        # refine (layer 41)
        _prefix('head.layers.41.layers.',     'head.refine.layers.')
        _prefix('head.layers.41.cls_layers.', 'head.refine.cls_layers.')

    return remap


def load_checkpoint(
    model: nn.Module,
    path: str,
    version: str = 'v2',
    strict: bool = False,
) -> None:
    """
    Load an official Sparse4D checkpoint into our model.

    Parameters
    ----------
    model   : Sparse4Dv1 or Sparse4Dv2 instance
    path    : path to the .pth checkpoint file
    version : 'v1' or 'v2' (must match model type)
    strict  : if True, raise on missing/unexpected keys
    """
    # weights_only=False: these are our own / official checkpoints (trusted).
    # Finetuned checkpoints embed an optimizer state that can contain numpy
    # scalars (e.g. a numpy-typed LR), which the PyTorch>=2.6 safe loader rejects.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    # Finetuned checkpoints (from train_finetune.py) store model under 'model' key
    # with our already-correct names, so they bypass the remap entirely
    if 'model' in ckpt and isinstance(ckpt['model'], dict):
        # Drop temporal cache buffers — they are runtime state and must be reset
        # Persisting them would inject one scene's temporal cache into a fresh run.
        ft_state = {k: v for k, v in ckpt['model'].items()
                    if '_cached_' not in k}
        result = model.load_state_dict(ft_state, strict=False)
        print(f'[ckpt] {path}  (finetuned checkpoint, epoch={ckpt.get("epoch","?")}'
              f'  loss={ckpt.get("loss", float("nan")):.4f})')
        return

    raw  = ckpt.get('state_dict', ckpt)

    remap  = _build_remap(raw, version=version)  # Official chekpoints
    result = model.load_state_dict(remap, strict=strict)

    loaded  = len(remap)
    missing = len(result.missing_keys)
    extra   = len(result.unexpected_keys)
    print(f'[ckpt] {path}')
    print(f'       mapped={loaded}  missing={missing}  unexpected={extra}')
    if missing > 0 and not strict:
        print('       (missing keys expected for unshared/optional modules)')
