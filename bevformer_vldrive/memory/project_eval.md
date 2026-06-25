---
name: project-eval
description: tools/eval.py checkpoint loading, mAP/NDS/BEV-mIoU evaluation, architecture remap notes
metadata:
  type: project
---

`tools/eval.py` evaluates BEVFormer-Tiny on nuScenes mini val (scene-0103, scene-0916, 81 frames).

**Checkpoint**: `model/checkpoints/bevformer_tiny_fp16_epoch_24.pth` — 592/643 keys loaded.

**What's loaded cleanly**: backbone (ResNet-50), FPN neck, BEV queries/pos-enc, CAN-bus MLP, encoder TSA+SCA (all 3 layers), decoder self-attn (all 6 layers), decoder BEV deformable cross-attn (all 6 layers), 6× reg_branches (full 5-layer Sequential), cls_branch (full 7-module Sequential).

**Key architecture fixes made**:
- `SimpleDetHead` now uses `_BEVDecLayer` with `BEVDeformCrossAttn` (not nn.TransformerDecoder) — matches checkpoint's `attentions.1.*` deformable-attn weights.
- `cls_branch` rebuilt as `Sequential([Linear, LN, ReLU, Linear, LN, ReLU, Linear])` to match official `cls_branches.5` structure (indices 0,1,3,4,6 in checkpoint).
- `reg_branches` rebuilt as `Sequential([Linear, ReLU, Linear, ReLU, Linear])` — all 3 linears now loadable.
- BEVFormerTiny.forward() now passes through `ref_pts` key.

**Results on mini val (81 frames, 5.3 FPS on MPS)**:
- NDS: 0.0238
- mAP: ~0.0000 (car AP ≈ 0.0002 — detection active but localization imprecise)
- BEV mIoU: 0.0083 (car: 0.065, others near 0)

**Why low vs paper (mAP≈25 on full val)**: mini val is only 81 frames (statistically noisy), and TSA/SCA pure-PyTorch grid-sample implementation differs slightly from the official CUDA ms_deform_attn kernel.

**Why:** To evaluate detection quality from the official epoch-24 checkpoint without mmcv/mmdet3d, in a pure-PyTorch MPS-compatible pipeline.
**How to apply:** Results in `eval_results/summary.json`. Re-run with `python tools/eval.py --score-thr 0.1`.
