# BEVFormer-Tiny — Model Architecture

Pure-PyTorch, MPS-compatible re-implementation of **BEVFormer-Tiny** (ResNet-50
backbone). It lifts 6 surround-view cameras into a 50×50 bird's-eye-view (BEV)
feature map and decodes 3-D bounding boxes — **without any `mmcv`, `mmdet3d`, or
custom CUDA/C++ extensions**. The official `bevformer_tiny_fp16` checkpoint loads
directly via a prefix remap (592/643 keys).

Tiny vs. Base: R50 backbone (not R101-DCN), 50×50 BEV (not 200×200), 3 encoder
layers (not 6), 480×800 input (not 1600×900), single-scale C5 features.

---

## Architecture overview

```
                6 camera images  (B, 6, 3, 480, 800)  float[0,255]
                          │
          ┌───────────────▼───────────────┐
          │  ResNet-50 backbone  → C5      │  ○ torchvision        backbone.py
          │  FPN neck (2048→256, 1 level)  │  ○ standard conv      backbone.py
          └───────────────┬───────────────┘
                          │  per-cam feature maps (6, 15×25, B, 256)
                          │  + camera embeds + level embed
                          ▼
   BEV queries 50×50 ──►┌─────────────────────────────────────────┐
   (2500, B, 256)       │           BEVFormerEncoder  × 3          │  encoder.py
   + learned BEV pos    │                                         │
   + CAN-bus MLP        │   ┌───────────────────────────────────┐ │
   + temporal prev_bev  │   │ TSA  Temporal Self-Attention      │★│  tsa.py
                        │   │   (deformable over [prev, curr])  │ │
                        │   ├───────────────────────────────────┤ │
                        │   │ SCA  Spatial Cross-Attention      │★│  sca.py
                        │   │   geometric lift-and-project to   │ │
                        │   │   each camera, deformable sample  │ │
                        │   ├───────────────────────────────────┤ │
                        │   │ FFN + LayerNorms                  │○│
                        │   └───────────────────────────────────┘ │
                        └────────────────────┬────────────────────┘
                          BEV feature map (B, 2500, 256)
                                             ▼
          ┌──────────────────────────────────────────────────────┐
          │  SimpleDetHead — DETR-style decoder × 6               │  bevformer_tiny.py
          │   • self-attention   (nn.MultiheadAttention)      ○   │
          │   • BEV deformable cross-attention                ★   │  BEVDeformCrossAttn
          │   • per-layer reg_branch → iterative ref refine   ○   │
          │   • cls_branch on final query                     ○   │
          └────────────────────────┬─────────────────────────────┘
                                   ▼
        cls_logits (B, 900, 10)  ·  reg_preds (B, 900, 10)  ·  ref_pts (B, 900, 3)

   ★ = pure-PyTorch / NumPy rewrite of an mmcv / mmdet3d CUDA-or-C++ op
   ○ = built on stock torch / torchvision
```

All deformable attention (TSA, SCA, decoder cross-attn) routes through a single
shared kernel — **`ms_deform_attn_core`** in [`deform_attn.py`](deform_attn.py) —
which replaces the mmcv `MultiScaleDeformableAttention` CUDA extension with
`F.grid_sample` bilinear interpolation that runs on MPS / CUDA / CPU.

### Why deformable attention ≠ `nn.MultiheadAttention`

They are different mechanisms, not two implementations of one. `nn.MultiheadAttention`
is **dense and content-based**: it computes `softmax(QKᵀ/√d)` over *every* key
(O(N²)). Deformable attention computes **no query–key dot products at all** — from
each query, two linear layers directly predict (a) a few sampling **offsets**
around a reference point and (b) the attention **weights** (`Linear(query).softmax()`),
then it bilinearly samples the value map at only those few sub-pixel locations
(`F.grid_sample`). So there is no QKᵀ matrix for `nn.MultiheadAttention` to provide,
and it cannot sample at learned continuous coordinates — which is exactly what TSA,
SCA, and the BEV cross-attn need (sparse, geometry-aware, linear cost). It also
matches the checkpoint, whose weights are `sampling_offsets` / `attention_weights` /
`value_proj` / `output_proj` (no `q_proj`/`k_proj`). `nn.MultiheadAttention` *is*
used — but only for the decoder's dense query-to-query self-attention.

---

## Step-by-step: camera input → final output

Implemented in [`bevformer_tiny.py`](bevformer_tiny.py) `BEVFormerTiny.forward()`.
Data flow (★ = pure-PyTorch rewrite, ○ = stock torch/torchvision):

```
 cameras ─►[backbone+neck]─► image feats ─►[ BEV encoder ×3 ]─► BEV map ─►[ decoder ×6 ]─► boxes
                                            TSA ★→ SCA ★→ FFN
```

Each row is one stage; the **Tensor out** column tracks the shape as data flows
top-to-bottom (`B` = batch):

| # | Stage | Operation | Tensor out | Where |
|:-:|---|---|---|---|
| 1 | Normalize | nuScenes mean/std, on-device FP32 | `(B, 6, 3, 480, 800)` | `bevformer_tiny.py` |
| 2 | Backbone + neck ○ | ResNet-50 → C5 (stride 32); FPN `2048→256` | `(6, 15×25, B, 256)` | `backbone.py` |
| 3 | Cam / level embeds | + learned per-camera + level embedding | `(6, 15×25, B, 256)` | `bevformer_tiny.py` |
| 4 | BEV queries | `50×50` grid + BEV pos-enc + CAN-bus MLP `(18→256)` | `(2500, B, 256)` | `bevformer_tiny.py` |
| 5 | Temporal align ★ | rotate `prev_bev` by Δyaw; shift TSA refs by Δxy | `(B, 2500, 256)` | `bevformer_tiny.py` |
| 6a | **TSA** ★ | deformable self-attn over `[prev ‖ curr]` BEV queue | `(B, 2500, 256)` | `tsa.py` |
| 6b | **SCA** ★ | lift: project pillars → cameras, deformable sample image feats | `(B, 2500, 256)` | `sca.py` |
| 6c | FFN ○ | feed-forward + LayerNorms | `(B, 2500, 256)` | `encoder.py` |
| ↻ | ×3 layers | repeat 6a–6c; NaN-guard the final BEV map | `(B, 2500, 256)` | `encoder.py` |
| 7 | Decoder ×6 | self-attn ○ + BEV deform cross-attn ★ + iterative ref refine | queries `(B, 900, 256)` | `bevformer_tiny.py` |
| 8 | Heads | `cls_branch` + final-layer `reg_branch` | see **Outputs** | `bevformer_tiny.py` |

**Step 6 lift geometry** ([`encoder.py`](encoder.py)): `get_reference_points`
builds the 3-D pillar anchor grid (4 height levels) + the 2-D BEV grid;
`point_sampling` ★ projects each anchor through the per-camera `lidar2img`
matrices, perspective-divides, normalizes to image coords, and builds a
field-of-view mask — replacing the mmdet3d projection utilities.

**Step 7 refinement:** each decoder layer reads the BEV map via
[`BEVDeformCrossAttn`](bevformer_tiny.py) ★ and updates the 3-D reference points
as `inverse_sigmoid → +Δ → sigmoid` (clamped for MPS stability).

**Outputs**

| Tensor | Shape | Meaning |
|---|---|---|
| `cls_logits` | `(B, 900, 10)` | class scores over the 10 nuScenes classes |
| `reg_preds` | `(B, 900, 10)` | box deltas (center, log-size, sin/cos yaw, velocity) |
| `ref_pts` | `(B, 900, 3)` | refined reference centers, normalized `[0,1]` |
| `bev_feat` | `(B, 2500, 256)` | recurrent state → next frame's `prev_bev` |

Decoding to drawable 3-D boxes (score threshold, denormalization to metres) and
all visualization happen downstream in `tools/` (see `../README.md`).

---

## Pure-PyTorch / NumPy replacements

The official BEVFormer depends on `mmcv` / `mmdet3d` and a custom CUDA kernel.
Every such op is rewritten in native PyTorch, so the whole model runs on Apple Silicon (MPS):

| Module / file | Original (mmcv / mmdet3d) | Pure-PyTorch / NumPy rewrite |
|---|---|---|
| [`deform_attn.py`](deform_attn.py) `ms_deform_attn_core` | `MultiScaleDeformableAttention` CUDA ext (`ms_deform_attn_forward`) | `F.grid_sample` bilinear sampling — the shared kernel for **all** deformable attention ★ |
| [`tsa.py`](tsa.py) `TemporalSelfAttention` | mmcv TSA (CUDA deformable attn) | deformable attention over the `[prev ‖ curr]` BEV queue; **no `nn.MultiheadAttention`** ★ |
| [`sca.py`](sca.py) `SpatialCrossAttention` | mmcv SCA + ms-deform-attn | per-camera rebatching + deformable sampling, valid-camera-count normalization ★ |
| [`encoder.py`](encoder.py) `point_sampling` | mmdet3d projection / coord transforms | explicit `lidar2img` matmul + perspective divide + FOV mask (FP32, no TF32) ★ |
| [`BEVDeformCrossAttn`](bevformer_tiny.py) | decoder deformable cross-attn (`attentions.1.*`) | 2-D deformable attention on the BEV map via the shared kernel ★ |
| `_rotate_prev_bev`, `_compute_shift` ([`bevformer_tiny.py`](bevformer_tiny.py)) | mmcv temporal alignment | `torchvision.transforms.functional.rotate` + NumPy CAN-bus geometry ★ |
| [`backbone.py`](backbone.py) ResNet-50 + FPN | mmdet `ResNet` / `FPN` | stock torchvision ResNet-50 (C5) + minimal `ConvModule` neck, attribute names kept so checkpoint keys map ○ |

**MPS-specific notes:** everything is pinned to FP32 (no autocast / fp16, which
MPS handles unreliably); `nan_to_num` is only called on contiguous tensors
(MPS miscomputes it on non-contiguous views); and the encoder output is
NaN-guarded before the detection head.

---

## Reimplementation challenges & fixes

The hard part was not the math but (a) matching the official checkpoint's exact
module layout so weights load, and (b) making deformable attention and geometry
numerically stable on MPS. The main issues hit during the port:

| Problem | Cause | Fix |
|---|---|---|
| No deformable-attention kernel on MPS | mmcv `MultiScaleDeformableAttention` is a compiled CUDA/C++ extension | Rewrote as `ms_deform_attn_core` with `F.grid_sample` (bilinear, `padding_mode='zeros'`); shared by TSA, SCA, and the decoder |
| Decoder weights wouldn't load | Used `nn.TransformerDecoder`, but the official decoder uses deformable cross-attn (`attentions.1.*`) | Built `_BEVDecLayer` around `BEVDeformCrossAttn` (heads=8, levels=1, points=4) to match the keys |
| `cls_branch` / `reg_branches` key mismatch | Official `cls_branches` = `[Linear,LN,ReLU,Linear,LN,ReLU,Linear]`, reg = `[Linear,ReLU,Linear,ReLU,Linear]` | Rebuilt the exact `Sequential` layouts so checkpoint indices (0,1,3,4,6 / 0,2,4) map cleanly |
| Silent NaNs in mixed precision | MPS doesn't support autocast / fp16 / bf16 reliably | Pin the whole model and all inputs to **FP32**; no amp anywhere |
| `nan_to_num` returns wrong values on MPS | MPS miscomputes it on non-contiguous views | Call `nan_to_num` only on **contiguous** tensors (e.g. inside `point_sampling`) |
| `inverse_sigmoid → ±Inf` during ref-point refinement | Iterative refinement could push `ref_pts` to exactly 0 or 1 | Clamp per-layer deltas to ±4 and final logits to ±6 before `sigmoid`, then `detach` |
| First-frame / projection cold-start NaNs | Pillars projecting outside every camera, or no `prev_bev` yet | NaN-guard at the encoder→decoder boundary (`nan_to_num` + warn) plus a downstream prediction guard |
| Geometric drift in projection | TF32 matmul lowers precision of `lidar2img` projection | `point_sampling` forced FP32 (no TF32), run under `torch.no_grad()` |
| Temporal misalignment between frames | Ego translates and rotates between frames | `_rotate_prev_bev` (`TF.rotate` by yaw delta) + `_compute_shift` (CAN-bus translation → BEV-normalized shift on TSA refs) |
| `query_embed` misinterpreted | Official packs `2*C` as `[query_pos ‖ content]` | Split first `C` = positional, last `C` = content query (and feed `query_pos` into self/cross-attn) |
| `ref_pts` unavailable downstream | `forward()` didn't expose the refined reference points | Pass `ref_pts` through the output dict for the decoder/visualizer |
| Lower mAP than the paper | pure-PyTorch `grid_sample` differs slightly from the CUDA `ms_deform_attn`; mini-val is only 81 frames | Accepted as a known gap — see eval notes in `../README.md` (detection is active, localization imprecise) |

---

## Files

| File | Contents |
|---|---|
| `bevformer_tiny.py` | `BEVFormerTiny` (full pipeline), `SimpleDetHead`, `BEVDeformCrossAttn`, CAN-bus MLP, temporal rotate/shift, image normalization |
| `encoder.py` | `BEVFormerEncoder`, `BEVFormerLayer`, `get_reference_points`, `point_sampling` |
| `tsa.py` | `TemporalSelfAttention` (deformable temporal self-attention) |
| `sca.py` | `SpatialCrossAttention` (geometric lift-and-project) |
| `deform_attn.py` | `ms_deform_attn_core` — shared `grid_sample` deformable kernel |
| `backbone.py` | `ResNet50Backbone` (C5), `FPNNeck` (single level) |
| `checkpoints/` | official `bevformer_tiny_fp16_epoch_24.pth` (gitignored) |
