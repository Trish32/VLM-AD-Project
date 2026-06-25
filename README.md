# VLMProjects

Pure-PyTorch ports of several autonomous-driving perception, prediction, and
planning models, all re-implemented to run on **Apple Silicon (MPS)** without
CUDA, `mmcv`, `mmdet3d`, or `spconv`.

| Project | What it is |
|---------|------------|
| [`bevformer_vldrive`](bevformer_vldrive) | BEVFormer-Tiny (camera BEV detection), nuScenes mini |
| [`bevfusion_vldrive`](bevfusion_vldrive) | BEVFusion (PointPillars + MIT det/seg), custom sparse conv |
| [`Occupancy/FlashOcc`](Occupancy) | FlashOcc BEVDetOCC 3D occupancy |
| [`simple_bev_vldrive`](simple_bev_vldrive) | Simple-BEV lift-splat BEV segmentation |
| [`sparse4d_vldrive`](sparse4d_vldrive) | Sparse4D v2/v3 detection + tracking + motion + ego-planning |
| [`motionForecasting/QCNet_vldrive`](motionForecasting) | QCNet motion forecasting (Argoverse 2) |
| [`simulator`](simulator) | Kinematic Bicycle Model closed-loop sim |

All projects share **one** conda environment.

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

## Not available on macOS

The Bench2Drive / CARLA closed-loop interface in
`sparse4d_vldrive/.../bench2drive` imports `carla` and `leaderboard`, which are
**Linux/Windows-only** and are intentionally left out of the environment. Every
other code path — including the local KBM `simulator` — runs fully on MPS.

## Data & model weights

Model checkpoints (`*.pt` / `*.pth` / `*.ckpt`) and the nuScenes / Argoverse
datasets are **not** included in this repo (too large for GitHub). Download
official weights and datasets per each project's notes and place them under the
relevant `checkpoints/` and `data/` directories.

## Notes

- Always send tensors to `torch.device("mps")`. Each project defaults to MPS and
  falls back to CPU when MPS is unavailable.
- Some ops still lack MPS kernels; run with
  `PYTORCH_ENABLE_MPS_FALLBACK=1` to let them fall back to CPU automatically.
