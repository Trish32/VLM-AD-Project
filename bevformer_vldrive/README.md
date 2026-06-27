# BEVFormer-VLDrive

A pure-PyTorch, MPS-compatible re-implementation of **BEVFormer-Tiny** (ResNet-50
backbone) for nuScenes, paired with a **Qwen2.5VL-7B** vision-language planner
(served locally via Ollama). BEVFormer lifts the 6 surround-view cameras into a
bird's-eye-view feature map and decodes 3-D detections; the VLM reads the
predicted BEV and emits a driving decision (`PROCEED` / `SLOW_DOWN` / `YIELD` /
`STOP`) with one-sentence reasoning.

![BEVFormer-VLDrive demo](bev_outputs/scene_gifs/scene07_scene-1077_bev.gif)

*Per-frame composite: predicted BEV (left) with short class tags (`veh`, `ped`,
`bar`, `cone`) | ground-truth ego trajectory (right), the 6-camera surround view
below with projected 3-D boxes, and the VLM reasoning / decision overlaid on the
BACK camera.*

---

## Generating visualization outputs

All commands run from this directory (`bevformer_vldrive/`) with the conda env
active. MPS needs the CPU-fallback flag for the few ops without Metal kernels:

```bash
conda activate simple_bev_vldrive
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 1. Composite scene GIFs (BEV + cameras + VLM reasoning)

```bash
# One animated composite per scene → bev_outputs/scene_gifs/scene{NN}_{name}_bev.gif
python tools/make_composite_gif.py --scenes 0 1 2 3 4 5 6 7 8 9 --vl

# A single scene:
python tools/make_composite_gif.py --scenes 5 --vl
```

Useful flags:
- `--no-vl` — skip Ollama and reuse the cached per-scene decisions in
  `tools/decisions/decisions_scene{N}.jsonl` (fast; reruns are resumable).
- `--cam-gif` — also write a camera-only GIF per scene.
- `--frame-ms 800` — per-frame duration.

Requires Ollama running with the VLM pulled: `ollama pull qwen2.5vl:7b`.

### 2. Per-frame BEV raster + camera mosaic

```bash
# bev_outputs/bev_xxx.png (BEV detections) and bev_outputs/cameras/cams_xxx.png
python tools/infer.py \
    --dataroot /Users/trish/Downloads/nuScenes_miniV1.0 \
    --checkpoint model/checkpoints/bevformer_tiny_fp16_epoch_24.pth \
    --scene 5 --max-frames 10 --score-thr 0.25 \
    --out-dir bev_outputs --save-cams
```

### 3. Pred-BEV canvas (`vis_xxx.png`) + live VLM composite

```bash
# bev_outputs/vis_xxx.png (scene canvas) and the live composite latest_bev_grid.jpg
python tools/vis_infer.py --scene 5 --max-frames 10                  # BEV only
python tools/vis_infer.py --scene 5 --max-frames 10 --vl \
    --log tools/decisions/decisions_scene5.jsonl                     # + streaming VLM
```

---

## Evaluation

`tools/eval.py` evaluates BEVFormer-Tiny on nuScenes mini val (scene-0103,
scene-0916, 81 frames) from the official epoch-24 checkpoint
(`model/checkpoints/bevformer_tiny_fp16_epoch_24.pth`, 592/643 keys loaded), in a
pure-PyTorch pipeline with no `mmcv` / `mmdet3d`.

**Results on mini val (81 frames, 5.3 FPS on MPS):**
- NDS: 0.0238
- mAP: ~0.0000 (car AP ≈ 0.0002 — detection active but localization imprecise)
- BEV mIoU: 0.0083 (car: 0.065, others near 0)

**Why low vs paper (mAP ≈ 25 on full val):** mini val is only 81 frames
(statistically noisy), and the TSA/SCA pure-PyTorch grid-sample implementation
differs slightly from the official CUDA `ms_deform_attn` kernel.

Results are written to `eval_results/summary.json`. Re-run with:

```bash
python tools/eval.py --score-thr 0.1
```
