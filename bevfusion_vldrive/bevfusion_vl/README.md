# BEVFusion (MIT) — pure-PyTorch camera+LiDAR fusion on Apple MPS

A from-scratch, **pure-PyTorch / MPS-compatible** port of **MIT-HAN-LAB
[BEVFusion](https://github.com/mit-han-lab/bevfusion)** (ICRA 2023) — unified
camera+LiDAR BEV fusion for **3-D detection *and* BEV map segmentation** — running
on Apple Silicon with **no `mmcv`, `mmdet3d`, `spconv`, `bev_pool`, or any custom
CUDA/C++ extension**. Both official checkpoints load with **0 missing / 0
unexpected** keys.

![BEVFusion detections](viz_out/bevfusion_mit_scene.gif)

*nuScenes-mini scene. **Left:** the 6 surround cameras with the predicted 3-D
boxes projected on (cars red, cones/barriers yellow, pedestrians green, …).
**Right:** the LiDAR-frame BEV — accumulated point cloud (height-shaded) with the
same boxes and the ego at the centre, forward = up.*

---

## Architecture

```
   6 cameras ─ Swin-T ─ GeneralizedLSSFPN ─ LSS depth-lift ─ BEVPool ─┐ camera BEV (80ch)
                                                                      ├─ ConvFuser ─ SECOND/FPN ─┬─ TransFusion head ─ 3-D boxes
   LiDAR ─ voxelize(mean VFE) ─ SparseEncoder (3-D sparse conv) ──────┘ lidar BEV (256ch)        └─ BEV seg head ─ map masks
```

- **Camera branch** ([`model/swin.py`](model/swin.py), [`model/lss_fpn.py`](model/lss_fpn.py),
  [`model/vtransform.py`](model/vtransform.py), [`model/bev_pool.py`](model/bev_pool.py))
  — Swin-T + `GeneralizedLSSFPN`; the LSS transform predicts a per-pixel depth
  distribution and `BEVPool`s the lifted features into an 80-channel BEV grid.
- **LiDAR branch** ([`model/voxelize.py`](model/voxelize.py),
  [`model/sparse_encoder.py`](model/sparse_encoder.py), [`model/spconv.py`](model/spconv.py),
  [`model/second.py`](model/second.py)) — the "VFE" is just a **mean over points**
  per voxel; a pure-PyTorch `SparseEncoder` (`spconv.py` reimplements `SubMConv3d`/
  `SparseConv3d`, verified vs `F.conv3d` to ~1e-5) produces a 256-channel BEV.
- **Fusion + heads** ([`model/bevfusion.py`](model/bevfusion.py),
  [`model/transfusion_head.py`](model/transfusion_head.py), [`model/seg_head.py`](model/seg_head.py))
  — a `ConvFuser` concatenates the camera/LiDAR BEVs; a shared SECOND/SECONDFPN
  backbone feeds a **TransFusion** detection head and a **BEV segmentation** head.

---

## Evaluation

nuScenes **mini-val**, official devkit, pure-PyTorch on MPS:

| task | metric | this port (mini-val) | official (full val) |
|---|---|:--:|:--:|
| Detection | mAP / NDS | **0.578 / 0.575** | 0.685 / 0.714 |
| Map seg.  | mIoU | **0.712** | 0.627 |

Detection per-class: bus .99, car .92, ped .93, cone .90, truck .81, moto .70,
bike .53 (trailer / construction_vehicle / barrier absent in mini). Seg per-class:
drivable 88.9, ped-crossing 79.9, walkway 71.6, carpark 76.4, divider 58.5,
stop-line 52.1 — matching/exceeding the official full-val per-class numbers.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
  python tools/eval_det.py --device cpu        # mAP / NDS
  python tools/eval_seg.py --device cpu        # mIoU
```

---

## Finding — the Swin `PatchMerging` bug

The camera branch was silently broken: a wrong `PatchMerging` ordering in the
Swin backbone corrupted the image features. Fixing it lifted **detection mAP
0.508 → 0.578** and **segmentation mIoU 0.302 → 0.712** (e.g. cone AP .51 → .90,
car .85 → .92). The lesson: a quietly-degraded camera stream hurts *both* the
detection and segmentation heads, since they share the fused BEV — and the
checkpoint still loads 0/0, so only the metrics reveal it.

---

## Visualize / run

```bash
conda activate simple_bev_vldrive
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 6-camera + LiDAR-BEV detection GIF → viz_out/bevfusion_mit_scene.gif
python visualize.py --max-frames 15 --device cpu
```

(Checkpoints `model/checkpoints/bevfusion-{det,seg}.pth` are gitignored — large.)
