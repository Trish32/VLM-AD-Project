# BEVFormer-VLDrive

A pure-PyTorch, MPS-compatible re-implementation of **BEVFormer-Tiny** (ResNet-50
backbone) for nuScenes, paired with a **Qwen2.5VL-7B** vision-language planner
(served locally via Ollama). BEVFormer lifts the 6 surround-view cameras into a
bird's-eye-view feature map and decodes 3-D detections.

The VLM then receives four channels, split by what each component is actually
good at: the **rendered BEV canvas** for layout, drawn ego-centric with the
heading straight up; the **forward camera** for the
semantics a BEV raster physically cannot carry (brake lights, signage,
construction, pedestrian intent, and which signal governs this lane); a
**map-projected crop** zoomed on the traffic light, because signal state is what
range destroys first; and the **decoded detections as text** — ranges, bearings
and closing rates treated as authoritative, so the VLM is not asked to re-estimate
by eye the geometry the detector already measured. It emits a traffic-light
reading, a driving decision (`PROCEED` / `SLOW_DOWN` / `YIELD` / `STOP`) and
one-sentence reasoning.

See **[RESULT.md](RESULT.md)** for the measured cost of each channel, the
ablations behind the design, and every caveat on the numbers.

![BEVFormer-VLDrive demo](bev_outputs/scene_gifs/scene04_scene-0757_bev.gif)

*scene-0757 — a red light with a completely clear road, which a BEV-only planner
cannot get right even in principle.*

*Top: the 6-camera surround view with projected 3-D boxes; VLM reasoning, the
`LIGHT` chip read from the forward camera, and the decision are overlaid on the
BACK cell. Bottom, all three sharing one **ego-centric forward-up** frame —
predicted BEV | ground-truth trajectory and GT boxes | the DiffusionDrive
planning view: the 3x6 kmeans anchor vocabulary (grey), the scored modes at the
speed the car is actually doing (teal), the top-1 plan after the safety filter
(orange), and the logged human path (white).*

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
  `tools/reasoning_decisions/decisions_scene{N}.jsonl` (fast; reruns are resumable).
- `--cam-gif` — also write a camera-only GIF per scene.
- `--frame-ms 800` — per-frame duration.

Requires Ollama running with the VLM pulled: `ollama pull qwen2.5vl:7b`.

### What the VLM receives

Four channels, split by what each component is actually good at: the **BEV
canvas** (ego-centric, heading up) for layout, the **forward camera** for the
semantics a BEV raster cannot carry, a **map-projected crop** zoomed on the
traffic light because signal state is what range destroys first, and the
**decoded detections as text** — treated as authoritative so the VLM is not
asked to re-estimate geometry the detector already measured.

The light is read in its **own call** with the camera alone: a single call
carrying detection text answers "no traffic lights visible" on frames with an
obvious red. That state then enters the decision call as text.

Full channel-by-channel costs, the ablations behind this design, the temporal
stability work and the detection numbers are in **[RESULT.md](RESULT.md)**.


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
    --log tools/reasoning_decisions/decisions_scene5.jsonl           # + streaming VLM
```

---

---

## Results

mini val **NDS 0.2255**, mAP **0.2334** over the 7 classes actually present
(vs the official tiny's 0.252); BEV mIoU 0.1902; 5.3 FPS on MPS. The red-light
case reads `red -> STOP` on 5/5 frames of scene-0757.

Method, ablations, retractions and caveats: **[RESULT.md](RESULT.md)**.

```bash
python tools/eval.py --score-thr 0.1      # -> eval_results/summary.json
```
