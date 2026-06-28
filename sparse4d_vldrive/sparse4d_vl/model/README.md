# `model/` — Sparse4D v1/v2/v3 internals

For the end-to-end diagram, eval numbers and findings see the
[project README](../../README.md).

---

## Module map

| file | role |
|------|------|
| `sparse4d_base.py` | shared base: ResNet-50 + 4-level FPN, image normalisation, meta-tensor helpers (used by v1/v2 and v3) |
| `backbone.py` | `ResNet50Backbone` (C2–C5) + `FPNNeck` (4 levels, 256-ch) |
| `sparse4d_v2.py` | `Sparse4Dv1` (single-frame) and `Sparse4Dv2` (temporal) end-to-end models |
| `sparse4d_v3.py` | `Sparse4Dv3` — adds decoupled attention, v3 anchor encoder, quality, optional motion/planning |
| `sparse4d_head_v2.py` | `Sparse4DHead` — one iterative decoder stage (v1/v2), driven by an `operation_order` list |
| `sparse4d_head_v3.py` | `Sparse4DHeadV3` — decoder stage with **decoupled** attention |
| `blocks.py` | `DeformableFeatureAggregation` (DFA), `SparseBox3DKeyPointsGenerator`, `AsymmetricFFN`, graph/decoupled attention |
| `instance_bank.py` | `InstanceBank` — 900-query management, temporal propagation, **track-id assignment** |
| `detection3d.py` | v2 box encoder / refinement / decoder (`boxes_3d` decode, NMS-free top-k) |
| `detection3d_v3.py` | v3 box encoder (per-component cat) + refinement (adds **centerness/yawness** quality) + decoder |
| `denoising.py` | Temporal Instance Denoising (DN) — training-only noised GT query groups + attention mask |
| `depth_head.py` | optional dense per-camera depth head (v2 aux supervision) |
| `motion_head.py` | QCNet-style multi-modal trajectory head (probabilistic) |
| `motion_planning.py` | SparseDrive `AgentMotionHead`, `EgoPlanner`, `MapEncoder` |
| `loss.py` | `Sparse4DLoss` (focal cls + L1 box + quality), DN loss, motion/planning losses |
| `checkpoint.py` | maps official flat `head.layers.*` checkpoints → this module layout |

---

## Key mechanisms

### Sparse queries & the InstanceBank (`instance_bank.py`)
The model never builds a dense BEV — it carries **900 sparse queries** (256-D
instance feature + an 11-D anchor each):

```
anchor = [x, y, z,  log_w, log_l, log_h,  sinθ, cosθ,  vx, vy, vz]
```

Per frame the bank returns **600 temporal** instances (propagated from the
previous frame, ego-motion compensated) + **300 fresh** K-means priors. After the
single-frame stage, `update()` merges the cached 600 with the top-300 fresh;
`cache()` keeps the top-600 (confidence multiplicatively decayed by 0.6) for the
next frame — temporal hysteresis that lets a briefly-occluded object survive.

**Tracking is built in.** `get_instance_id()` lets cached instances inherit the
id they carried last frame and mints new sequential ids for fresh confident
detections — detection-and-tracking in one pass, **no Hungarian/IoU matcher**.

### Deformable Feature Aggregation (`blocks.py`) — the CUDA replacement
For each anchor, `SparseBox3DKeyPointsGenerator` produces **13 keypoints** (7
fixed box-relative + 6 learned from the instance feature). `project_points()`
projects them into all 6 cameras via `projection_mat` (lidar→pixel), and
`_sample_features()` samples the **4 FPN levels** with `F.grid_sample`
(`padding_mode='zeros'` handles out-of-FOV). Samples are combined by learned
weights across **cameras × levels × keypoints**, grouped into **8** attention
groups — the pure-PyTorch stand-in for Sparse4D's CUDA `deformable_aggregation`.

### Decoder stages — v2 vs v3
A stage is a list of ops applied in order:

```
Stage 0 (single-frame):  ['deformable', 'ffn', 'norm', 'refine']
Stages 1-5 (temporal) :  ['temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine']
```

- **v2** (`sparse4d_head_v2.py`): standard multi-head self-attention over queries.
- **v3** (`sparse4d_head_v3.py`): **decoupled attention** — q/k built by
  concatenating instance + anchor embeddings (512-D) through shared
  `fc_before`/`fc_after`, reducing the embedding-aliasing of v2.
- **v3 quality** (`detection3d_v3.py`): refinement also emits
  `(centerness, yawness)`; the decoder re-ranks final scores by
  `sigmoid(cls) · sigmoid(centerness)`.

Each of the 6 stages has **unique weights** (flat `head.layers.0..38` in the
checkpoint), mapped by `checkpoint.py`.

### Anchor size slots are length-first (a gotcha)
The decoded box is *named* `[x,y,z, w,l,h, yaw, vx,vy]`, but the GT encoder stores
**slot 3 = length** (extent along heading) and **slot 4 = width** (see the parent
README's Findings). Consumers must draw `box[3]` along `yaw`; trusting the names
rotates every box 90°.

### Heads (optional, built on the instance features)
- **Motion** — `AgentMotionHead` (SparseDrive) or `TrajectoryHead` (QCNet-style):
  multi-modal future trajectories per agent, optionally map-conditioned.
- **Planning** — `EgoPlanner`: a command-conditioned (left/straight/right) ego
  trajectory with 3 modes, consuming agent + `MapEncoder` tokens.
- **Depth** — `DepthHead`: dense per-camera depth, a v2 training aux signal.

### Training-only: Temporal Instance Denoising (`denoising.py`)
`generate_dn_groups()` appends groups of noised GT queries after the 900 regular
queries, isolated by `build_dn_attn_mask()` so they never leak into the temporal
cache. (On the mini split this *hurt* — see the parent README's Findings.)

---

## Pure-PyTorch / MPS replacements

| original (Sparse4D ref) | here |
|---|---|
| CUDA `deformable_aggregation` | `DeformableFeatureAggregation` via `F.grid_sample` (`blocks.py`) |
| mmdet `ResNet` / `FPN` | torchvision ResNet-50 + minimal FPN (`backbone.py`) |
| mmdet3d box coders / projection | explicit anchor encode + `lidar2img` projection (`detection3d*.py`, `blocks.py`) |
| official flat checkpoint keys | prefix/index remap in `checkpoint.py` |

Everything is pinned to FP32 (MPS has no reliable autocast/fp16).

---

## Loading the official checkpoint

```python
from sparse4d_vl.model.sparse4d_v3 import Sparse4Dv3
from sparse4d_vl.model.checkpoint import load_checkpoint

model = Sparse4Dv3(pretrained_backbone=False).eval()
load_checkpoint(model, "sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth", version="v3")
```

