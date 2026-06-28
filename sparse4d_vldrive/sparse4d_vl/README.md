# `sparse4d_vl/` — the Sparse4D implementation package

Pure-PyTorch, MPS-compatible implementation of the **Sparse4D** family
(v1 → v2 → v3) plus SparseDrive-style **motion forecasting** and **ego planning**,
runnable on Apple Silicon with **no `mmcv` / `mmdet3d` / CUDA extensions**.

This page is the package map and run guide. For the architecture diagram, eval
results and findings see the [project README](../README.md); for model internals
see [`model/README.md`](model/README.md).

---

## Version lineage

| version | adds | entry class |
|---|---|---|
| **v1** | single-frame sparse 3-D detection (no temporal) | `model.sparse4d_v2.Sparse4Dv1` |
| **v2** | temporal instance propagation, dense-depth aux | `model.sparse4d_v2.Sparse4Dv2` |
| **v3** | decoupled attention, per-component anchor encoder, quality (centerness/yawness), temporal instance denoising | `model.sparse4d_v3.Sparse4Dv3` |
| **+ SparseDrive** | multi-modal agent motion + command-conditioned ego planning + HD-map encoder | `Sparse4Dv3(with_planning=True, with_map=True)` |

All versions share one backbone (ResNet-50 + 4-level FPN) and the 900-query
sparse decoder.

---

## Layout

```
sparse4d_vl/
├── model/      # network: backbone, decoder heads (v1/v2/v3), DFA, instance bank,
│               #          motion/planning, denoising, loss, checkpoint  → model/README.md
├── data/       # nuScenes loaders + anchors
│   ├── loader.py            # base mini loader (images, projection_mat, ego poses)
│   ├── finetune_loader.py   # adds GT boxes / futures / ego plan / HD-map polylines
│   ├── lidar_depth.py       # LiDAR → sparse depth GT (v2 depth aux)
│   ├── nuscenes_kmeans900.npy   # 900 K-means anchor priors
│   └── motion_anchors*.npz      # motion-mode anchors
├── tools/      # inference, evaluation, visualisation
│   ├── infer.py             # quick scene inference + simple BEV dump
│   ├── test.py              # run all 10 mini scenes, report stats
│   ├── eval.py              # detection mAP / NDS (nuScenes devkit)
│   ├── eval_track.py        # tracking AMOTA / AMOTP
│   ├── eval_motion.py       # motion + planning metrics
│   ├── visualizer.py        # 3-D boxes projected onto the 6 cameras
│   └── visualize_track.py   # tracking GIF (cameras + BEV, colour = track id)
└── bench2drive/             # closed-loop planning eval harness
```

---

## Run

All commands from the repo root, env active, MPS CPU-fallback on:

```bash
conda activate simple_bev_vldrive
export PYTORCH_ENABLE_MPS_FALLBACK=1
CKPT=sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth
```

```bash
# Inference + simple BEV dump on one scene
python sparse4d_vl/tools/infer.py --version v3 --scene 0

# Detection eval (mAP / NDS)
python sparse4d_vl/tools/eval.py --version v3 --checkpoint $CKPT --eval-set mini_val

# Tracking eval (AMOTA / AMOTP)
python sparse4d_vl/tools/eval_track.py --checkpoint $CKPT --eval-set mini_val

# Tracking visualisation → sparse4d_track_outputs/scene0_track.gif
python sparse4d_vl/tools/visualize_track.py --scene 0 --max-frames 20

# Camera-projected 3-D boxes
python sparse4d_vl/tools/visualizer.py --version v2 --checkpoint $CKPT --scene 0
```

Training entry points live one level up: `train_v2.py`, `train_v3.py`,
`train_finetune.py`.

---

## Data & checkpoints

- **Dataset**: nuScenes **mini** (`v1.0-mini`) at
  `/Users/trish/Downloads/nuScenes_miniV1.0`, 6 cameras resized to 256×704.
- **Checkpoints** (`model/checkpoints/`, gitignored — large): `sparse4dv3_r50.pth`,
  `sparse4dv2_r50_HInf_256x704.pth`. Loaded via `model.checkpoint.load_checkpoint`.
