# Sparse4D-v3 — pure-PyTorch, MPS end-to-end perception

A from-scratch, **pure-PyTorch / MPS-compatible** reimplementation of **Sparse4D-v3**
— sparse, anchor-based, multi-view **detection + tracking + motion + planning** —
running on Apple Silicon **without `mmcv`, `mmdet3d`, or any custom CUDA/C++
extension**. The custom 4-D Deformable Aggregation is replaced with vectorized
coordinate projection + `torch.nn.functional.grid_sample`.

The model loads the official `sparse4dv3_r50.pth` checkpoint and reproduces its
detection / tracking metrics on the nuScenes mini split.

![Sparse4D-v3 tracking](sparse4d_track_outputs/scene0_track.gif)

*Detection **and** tracking on nuScenes scene-0. **Left:** the 6 surround cameras
with projected 3-D boxes. **Right:** a top-down BEV (LIDAR frame, ego at origin,
forward = up). Every box — on the cameras and in the BEV — is coloured by its
**persistent track id** (assigned by the temporal instance bank), and the fading
BEV trail is that track's past centres, so identity persistence is visible across
the whole clip.*

---

## End-to-end architecture

```
   6 surround cameras  (1, 6, 3, 256, 704)
              │
   ┌──────────▼───────────┐
   │ ResNet-50 + 4-lvl FPN │   C2..C5 → 256-ch features at strides 4/8/16/32
   └──────────┬───────────┘     (64×176, 32×88, 16×44, 8×22)
              │  multi-scale image features
   ┌──────────▼─────────────────────────────────────────────┐
   │ Sparse query decoder — 900 anchors (256-D)              │
   │   = 600 temporal (from InstanceBank) + 300 fresh priors │
   │                                                         │
   │   Stage 0 (single-frame):  DFA → FFN → norm → refine    │
   │   ── InstanceBank.update(): merge [cached 600 | top-300]│
   │   Stages 1-5 (temporal), unique weights each:           │
   │      temp_gnn → gnn → norm → DFA → FFN → norm → refine   │
   │                                                         │
   │   DFA  : 13 keypoints/anchor → project to 6 cams →      │
   │          grid_sample 4 FPN levels → grouped (8) weighted│
   │          aggregation   (the MPS replacement for CUDA DFA)│
   │   v3   : decoupled attention, per-component anchor       │
   │          encoder, quality head (centerness, yawness)     │
   └──────────┬──────────────────────────────────────────────┘
              │  refined instance features + anchors
   ┌──────────▼───────────┐   InstanceBank.cache() → next frame
   │ heads / decode        │   + get_instance_id() → track ids
   ├───────────────────────┤
   │ • Detection  → boxes_3d [x,y,z,w,l,h,yaw,vx,vy], score, label
   │ • Tracking   → persistent track ids (ID propagation, no matcher)
   │ • Motion     → multi-modal agent trajectories (map-conditioned)
   │ • Planning   → command-conditioned ego trajectory (3 modes)
   └───────────────────────┘
```

**Per-frame flow** (`sparse4d_vl/model/sparse4d_v3.py` `forward`):

1. **Backbone + FPN** — ResNet-50 to C2–C5, a 4-level FPN gives 256-channel
   feature maps at strides 4/8/16/32 ([`model/backbone.py`](sparse4d_vl/model/backbone.py)).
2. **Sparse queries** — the **InstanceBank** supplies 900 anchors: 600 propagated
   from previous frames + 300 fresh K-means priors. Anchors are 11-D
   `[x,y,z, log_w,log_l,log_h, sinθ,cosθ, vx,vy,vz]`
   ([`model/instance_bank.py`](sparse4d_vl/model/instance_bank.py)).
3. **Stage 0** — a single-frame decoder layer refines the fresh priors; then
   `InstanceBank.update()` merges the cached 600 with the top-300 fresh.
4. **Stages 1-5** — temporal decoder layers (each with its own checkpoint
   weights) attend to current + cached instances and the image features via DFA.
5. **Deformable Feature Aggregation** — for each anchor, 13 keypoints (7 fixed +
   6 learned) are projected into all 6 cameras and the 4 FPN levels are sampled
   with `grid_sample`, then aggregated per group
   ([`model/blocks.py`](sparse4d_vl/model/blocks.py)) — this is the pure-PyTorch
   stand-in for Sparse4D's CUDA `deformable_aggregation`.
6. **Cache + track** — `InstanceBank.cache()` keeps the top-600 (confidence
   decayed) for the next frame; `get_instance_id()` lets cached instances inherit
   their id and mints new ids for fresh confident detections → **tracking with no
   separate association step**.
7. **Heads** — the decoder emits boxes/scores/labels (quality-reranked); optional
   SparseDrive **motion** (`AgentMotionHead`) and **planning** (`EgoPlanner`,
   command-conditioned) heads consume the instance features + HD-map tokens
   ([`model/motion_planning.py`](sparse4d_vl/model/motion_planning.py)).

Training adds **temporal instance denoising** (DN groups, isolated by an
attention mask) — see [`model/denoising.py`](sparse4d_vl/model/denoising.py).

Key dims: `EMBED_DIMS=256`, `NUM_CAMS=6`, `NUM_LEVELS=4`, `NUM_GROUPS=8`,
`NUM_PTS=13`, `NUM_CLASSES=10`, 900 anchors (600 temporal + 300 fresh).

---

## Tracking task visualization

Sparse4D-v3 is **detection-and-tracking in one network**: the temporal instance
bank carries the top-600 instances across frames, so a detected object keeps the
**same query — and the same id — for as long as it stays confident**, with no
Hungarian/IoU matcher. The GIF above colours every box by that id and draws each
track's recent trail, so identity persistence is visible directly.

Regenerate it:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
  python sparse4d_vl/tools/visualize_track.py --scene 0 --max-frames 20 \
         --score-thresh 0.45
# → sparse4d_track_outputs/scene0_track.gif
```

---

## Evaluation results

nuScenes **mini-val**, official devkit evaluators, from `sparse4dv3_r50.pth`,
pure-PyTorch on Apple M3 Max (MPS, ~250 ms/frame).

**Detection** — `NuScenesEval` ([`tools/eval.py`](sparse4d_vl/tools/eval.py)):

| metric | value |
|---|---|
| **mAP** | **0.463** |
| **NDS** | **0.480** |

| class | AP | | class | AP |
|---|---|---|---|---|
| traffic_cone | 0.819 | | motorcycle | 0.683 |
| car | 0.738 | | pedestrian | 0.655 |
| truck | 0.707 | | bicycle | 0.331 |
| bus | 0.699 | | trailer / constr. / barrier | 0.000\* |

\* trailer, construction_vehicle and barrier are rare/absent in the mini split,
so their AP is 0.

**Tracking** — `TrackingEval`, 7 tracking classes
([`tools/eval_track.py`](sparse4d_vl/tools/eval_track.py)):

| AMOTA | AMOTP | MOTA | recall |
|---|---|---|---|
| **0.627** | 0.772 | 0.631 | 0.709 |

Reproduce:

```bash
# detection (mAP / NDS)
python sparse4d_vl/tools/eval.py --version v3 \
  --checkpoint sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth --eval-set mini_val
# tracking (AMOTA / AMOTP)
python sparse4d_vl/tools/eval_track.py \
  --checkpoint sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth --eval-set mini_val
```

---

## Findings

### Temporal instance denoising (DN) hurts on the mini-split fine-tune

DN appends groups of noised ground-truth queries (isolated by an attention mask)
to stabilize bipartite matching and speed convergence — a win on **full**
nuScenes (~28 k frames). To test it here, the official checkpoint was fine-tuned
for **6 epochs on nuScenes mini** under one recipe (unfrozen backbone, quality +
depth aux losses), changing **only** `--dn_groups`:

| run | DN groups | mAP | NDS |
|---|:--:|:--:|:--:|
| pretrained baseline (no fine-tune) | – | **0.463** | **0.480** |
| fine-tune, **no** DN  (`dn_groups=0`) | 0 | 0.453 | 0.476 |
| fine-tune, **+ DN**   (`dn_groups=5`) | 5 | 0.394 | 0.439 |

*(`tools/eval.py` on mini-val; see `train_logs/eval_v3_compare.log`.)*

**Takeaway.** On this tiny budget DN was **counter-productive** — it dropped mAP
~6 points (0.453 → 0.394), while a plain no-DN fine-tune stayed essentially
lossless vs the strong pretrained baseline (0.463 → 0.453). With only a few
hundred frames and 6 epochs, the extra DN optimization burden perturbs
already-good weights before its convergence benefit — which needs large-scale
data — can pay off. DN is a **scale-dependent** trick, not a free improvement; for
small-data fine-tuning, leave it off.

### Anchor size slots are length-first (a w/l-swap gotcha)

The box is *named* `[x,y,z, w,l,h, yaw, …]`, but the GT encoder
([`data/finetune_loader.py`](sparse4d_vl/data/finetune_loader.py)) stores **slot 3
= length** (extent along heading) and **slot 4 = width**. Any visualiser/consumer
that trusts the names and draws `box[4]` along `yaw` renders every box rotated
90°. Confirmed empirically (moving vehicles: `box[3]≈4.5 m`, `box[4]≈1.9 m`, with
`yaw` matching the velocity direction). The tracking visualiser and the
downstream KBM simulator both draw `box[3]` along the heading for this reason.

---

## Layout

| path | role |
|------|------|
| `sparse4d_vl/model/` | model: backbone, FPN, decoder heads (v2/v3), `blocks.py` (DFA), `instance_bank.py` (temporal + tracking), `motion_planning.py`, `denoising.py`, `loss.py` |
| `sparse4d_vl/data/` | nuScenes loaders (`loader.py`, `finetune_loader.py`), K-means anchors, motion anchors |
| `sparse4d_vl/tools/` | `eval.py` (mAP/NDS), `eval_track.py` (AMOTA), `eval_motion.py`, `visualizer.py` (camera 3-D boxes), `visualize_track.py` (BEV tracking GIF), `test.py` |
| `sparse4d_vl/infer.py` | quick scene inference + simple BEV dump |
| `train_v2.py` / `train_v3.py` / `train_finetune.py` | training entry points |
| `checkpoints/` | `sparse4dv3_r50.pth`, `sparse4dv2_r50_*.pth` (gitignored — large) |
