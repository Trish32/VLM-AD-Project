# VLMProjects

Pure-PyTorch ports of several autonomous-driving perception, prediction, and
planning models, all re-implemented to run on **Apple Silicon (MPS)** without
CUDA, `mmcv`, `mmdet3d`, or `spconv` — plus **two integrated pipelines** built
on top of them.

### Component ports

| Project | What it is |
|---------|------------|
| [`bevformer_vldrive`](bevformer_vldrive) | BEVFormer-Tiny (camera BEV detection) + Qwen2.5-VL planner, nuScenes mini |
| [`bevfusion_vldrive`](bevfusion_vldrive) | BEVFusion, two heads (MIT TransFusion, ADLab anchor3d), custom sparse conv |
| [`sparse4d_vldrive`](sparse4d_vldrive) | Sparse4D v2/v3 detection + tracking + motion + ego-planning |
| [`Occupancy/FlashOcc`](Occupancy) | FlashOcc BEVDetOCC 3-D occupancy (18-class Occ3D) |
| [`simple_bev_vldrive`](simple_bev_vldrive) | Simple-BEV lift-splat BEV segmentation |
| [`motionForecasting/QCNet`](motionForecasting) | QCNet motion forecasting (Argoverse 2) |
| [`diffusiondrive_planner`](diffusiondrive_planner) | DiffusionDrive truncated-diffusion planning head |
| [`simulator`](simulator) | Kinematic bicycle model closed-loop sim |

### Integrated pipelines

| Pipeline | Question it answers |
|---|---|
| [**BEV → VLM → DiffusionDrive**](#pipeline-1-bev--vlm--diffusiondrive) | Does better 3-D perception produce better *language-mediated* driving decisions? |
| [**e2e_pipeline**](#pipeline-2-e2e_pipeline) | Can a modular stack with calibrated uncertainty and a safety gate drive closed-loop? |

All projects share **one** conda environment.

---

## Pipeline 1: BEV → VLM → DiffusionDrive

A perception-to-decision chain where the middle stage is a vision-language model,
built to test one claim: **does detection quality survive a natural-language
bottleneck?**

```
6 surround cameras
        │
        ▼
┌─────────────────┐   swap this stage, hold everything else fixed
│  3-D DETECTOR   │   BEVFormer-Tiny (cam) / BEVFusion-robust (L+C) / BEVFusion-MIT (L+C)
└─────────────────┘
        │  boxes + scores
        ▼
┌───────────────────────────────────────────────────────────┐
│  Qwen2.5-VL-7B  (local, via Ollama)  — FOUR channels:     │
│    1. BEV raster        the detector's whole output       │
│    2. forward camera    raw scene appearance              │
│    3. traffic-light crop, map-projected                   │
│    4. detections as text, capped at 5 rows                │
└───────────────────────────────────────────────────────────┘
        │  {light state, decision, one-sentence reasoning}
        │  decision ∈ PROCEED | SLOW_DOWN | YIELD | STOP
        ▼
┌─────────────────┐
│ DiffusionDrive  │  anchor vocabulary conditioned on the decision;
│ truncated diff. │  2 denoising steps from anchors, not 100 from noise
└─────────────────┘
        │
        ▼   planned trajectory (6 waypoints, 3 s)
```

**Step by step**

1. **Detect.** One of three detectors runs on the same frames. Only this stage
   changes between arms — same camera, same light state, same prompt, same
   decoding.
2. **Render four channels.** The BEV raster carries the detector's *full* output;
   the detection text is capped at 5 rows, so it cannot.
3. **Ask the VLM.** Ollama serves Qwen2.5-VL-7B locally; structured output
   constrains the reply to a schema, removing prose parsing entirely.
4. **Condition the planner.** The decision selects DiffusionDrive's anchor set
   and target speed; the denoiser refines from anchors rather than from noise.

**Result — detection quality does survive the bottleneck:**

| arm | mAP | agrees with GT decision | trajectory endpoint vs GT |
|---|---|---|---|
| BEVFormer-Tiny (camera) | 0.163 | 54.3% | 2.92 m |
| BEVFusion-robust (L+C) | 0.468 | 58.0% | 2.60 m |
| **BEVFusion-MIT (L+C)** | **0.578** | **69.1%** | **1.95 m** |

Pearson **r = +0.856** between mAP and decision agreement.

**And the finding depends on the BEV raster.** An earlier run sending camera +
light + detection text alone measured **r = +0.07** and concluded detector
quality did not matter. That conclusion is retracted: with the text capped at 5
rows, two detectors differing mainly in the other 40+ boxes produce nearly
identical payloads. The raster has no such cap.

---

## Pipeline 2: e2e_pipeline

A modular closed-loop stack — [`e2e_pipeline/`](e2e_pipeline) — where every port
sits behind a `Protocol` and everything downstream reads **one** scene
representation.

```
6 surround cameras ──┬──► object branch    (Sparse4D v3 / BEVFormer / BEVFusion)
                     └──► dense branch     (FlashOcc, 18-class Occ3D)
                                │
                                ▼
                    SceneRepresentation  (+x fwd, +y left, metres)
              agents + Kalman covariance │ free space + ESDF │ ego
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   risk model            candidate planner        safety filter
  calibrated             DiffusionDrive          1 drivable area
  P(collision)           anchors, command-       1b unknown, graded
                         conditioned             2 swept footprint
                                                 3 KBM feasibility
                                                 4 calibrated risk
                                │
                                ▼   best feasible, else emergency brake
                        independent verifier (8 rules, default-off)
                                ▼
              pure-pursuit controller → kinematic bicycle → ego state ──┐
                                ▲                                       │
                                └───────────── closed loop ─────────────┘
```

**Step by step**

1. **Perceive** — object and dense branches run in parallel off the same
   cameras. Neither is redundant: the detector reports 10 scored classes above
   threshold, occupancy marks a voxel occupied without needing a name for it.
2. **Fuse** — both converge on `SceneRepresentation`. Everything downstream
   reads only that, which is what makes the backbones swappable.
3. **Quantify uncertainty** — a constant-velocity Kalman filter supplies the
   covariance detectors do not, with the score→σ mapping **fitted** to 11,730
   detector-to-annotation matches rather than assumed.
4. **Propose** — DiffusionDrive anchors, conditioned on a drive command derived
   per step and on VLM intent where available.
5. **Gate** — four hard gates plus a graded treatment of unobserved space
   (observed obstacle rejects, observed free passes, unknown is priced under a
   stopping constraint).
6. **Verify** — an independent re-check using *different algorithms* for the
   same properties (finite-difference dynamics vs reachability, polygon SAT vs
   occupancy-raster sweep), default-off.
7. **Control and close the loop** — pure pursuit into a kinematic bicycle model;
   the ego's own errors compound, which is the point.

**Results** — 10 nuScenes-mini scenes × 20 steps, derived commands:

| perception | mode | ego-fault | other | brakes | clearance | completion | latency p50/p95 |
|---|---|---|---|---|---|---|---|
| GT oracle | free | 0 | 0 | 78 | 2.05 m | 49.2% | 23 / 68 ms |
| BEVFormer-tiny | free | 0 | 20 | 94 | 1.83 m | 39.8% | 27 / 69 ms |
| **BEVFusion-MIT** | free | 2 | **1** | **77** | 1.52 m | **48.3%** | — |

A real detector brings other-fault collisions **20 → 1** and lands at
oracle-level braking and completion. Full method, ablations and **13 retracted
causal claims** in [`e2e_pipeline/EXPERIMENT.md`](e2e_pipeline/EXPERIMENT.md).

**What this stack does not do:** recovery rate from a divergence excursion is
**0%**. Once the ego falls behind the logged trajectory it never returns.

---

## Requirements

- macOS on **Apple Silicon** (M1/M2/M3…). Tested on an M3 Max.
- [Miniforge / Miniconda](https://github.com/conda-forge/miniforge) (`conda`, or `mamba`).
- Python 3.12.

## Set up the environment

```bash
# from the repo root
conda env create -f environment.yml      # or: mamba env create -f environment.yml
conda activate vldrive
python check_mps.py                       # expect "MPS available: True"
```

Or run the helper script, which does the same thing:

```bash
./setup_env.sh
```

Prefer a plain virtualenv instead of conda?

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python check_mps.py
```

Exact pinned versions live in [`environment.yml`](environment.yml) and
[`requirements.txt`](requirements.txt). The Argoverse 2 stack (`av2`, `pandas`,
`pyarrow`) is only needed for QCNet motion forecasting; everything else is the
shared nuScenes / imaging / PyTorch stack.

## macOS gotcha: duplicate OpenMP runtime

You may hit:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

PyTorch and conda each ship their own `libomp.dylib`. Two fixes:

```bash
# (a) quick — set before running any script
export KMP_DUPLICATE_LIB_OK=TRUE

# (b) clean — point conda's copy at PyTorch's (run once, after install)
cd "$CONDA_PREFIX/lib"
mv libomp.dylib libomp.dylib.bak
ln -s python3.12/site-packages/torch/lib/libomp.dylib libomp.dylib
```


## Data & model weights

Model checkpoints (`*.pt` / `*.pth` / `*.ckpt`) and the nuScenes / Argoverse
datasets are **not** included in this repo (too large for GitHub). Download
official weights and datasets per each project's notes and place them under the
relevant `checkpoints/` and `data/` directories.

Cached **detector output** is checked in, under
[`e2e_pipeline/data/`](e2e_pipeline/data) — BEVFormer-tiny, Sparse4D tracking and
BEVFusion submissions over nuScenes mini (~53 MB). These are the inputs the
closed-loop comparisons replay, so keeping them tracked means the pipeline-2
results reproduce without a 577 MB checkpoint and a 20-minute MPS inference run.

## Running the pipelines

```bash
conda activate simple_bev_vldrive
export PYTORCH_ENABLE_MPS_FALLBACK=1
export NUSCENES_DATAROOT=~/Downloads/nuScenes_miniV1.0

# Pipeline 1 — BEV -> VLM -> DiffusionDrive  (needs Ollama for the VLM stage)
cd bevformer_vldrive
python tools/make_composite_gif.py --scenes 5 --vl      # --no-vl reuses cached decisions
python tools/compare_detectors.py                       # the three-arm table

# Pipeline 2 — e2e_pipeline
cd /Users/trish/VLMProjects
PYTHONPATH=. python -m e2e_pipeline.tools.final_baseline   # the results table
PYTHONPATH=. python -m e2e_pipeline.tools.visualize --scene 6
PYTHONPATH=. python -m pytest e2e_pipeline/tests -q        # 216 tests, ~2 s
```
