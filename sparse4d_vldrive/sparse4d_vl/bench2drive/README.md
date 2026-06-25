# Bench2Drive interface for the Sparse4D-v3 driving stack

End-to-end vision pipeline: **6-camera multi-view → detect → track → motion → plan → control**.

```
controller.py          trajectory → steer/throttle/brake (pure-pursuit + PID), no carla dep
agent.py               CARLA-Leaderboard AutonomousAgent (sensors() + run_step() → VehicleControl)
validate_openloop.py   local end-to-end validation on nuScenes (no CARLA needed)
```

## Validate locally (this machine, no CARLA)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
  python sparse4d_vl/bench2drive/validate_openloop.py \
  --checkpoint checkpoints/train_v3_plan3/epoch_05.pt --scene 1 --max-frames 8
```
Runs the same `plan_to_control` path as the real agent and prints per-frame
#tracks, #motion agents, command, ego-plan endpoint, the emitted control, and
open-loop planning L2/collision vs GT ego.

## Run on real Bench2Drive (CARLA, Linux + GPU)

This **cannot run on macOS / Apple Silicon** — CARLA is a Linux/Windows GPU
simulator. On a proper box:

1. Install CARLA 0.9.15 + the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) repo (Leaderboard 2.0 + scenario_runner).
2. Copy this project so `sparse4d_vl` is importable, with a trained checkpoint.
3. Point the leaderboard at this agent:
   ```bash
   export TEAM_AGENT=sparse4d_vl/bench2drive/agent.py
   export TEAM_CONFIG=/path/to/checkpoint.pt      # passed to setup()
   bash leaderboard/scripts/run_evaluation.sh     # Bench2Drive entry
   ```
   `get_entry_point()` returns `Sparse4DB2DAgent`.

## ⚠️ Caveats (read before trusting Driving Scores)

- **Domain gap**: the checkpoint is trained on nuScenes (real). Zero-shot CARLA
  transfer is weak (sim-vs-real + different sensor rig). For real Bench2Drive
  scores, **retrain on Bench2Drive's CARLA data** (the loader/heads are reusable;
  only the dataset wrapper changes). The agent reads calibration at runtime so it
  is rig-agnostic.
- **Frames**: CARLA ego is x-fwd/y-RIGHT (left-handed); the model is x-fwd/y-LEFT
  (nuScenes). The agent prepends a y-flip to every camera projection and flips
  the planned-trajectory y / steer sign on output. Verify on the actual rig.
- **Camera rig** (`CAM_RIG` in agent.py): poses/FOV are nuScenes-like placeholders;
  set them to the exact Bench2Drive sensor extrinsics/intrinsics.
- **Controller** (`controller.py`): pure-pursuit + PID with default gains
  (wheelbase 2.85 m, dt 0.5 s); tune for the CARLA vehicle. The model→CARLA steer
  sign flip lives in `agent.run_step`.
- **No online map in CARLA path**: `agent.py` currently passes no `map_pts`
  (agent–map attention is skipped). To use it in CARLA, query the CARLA map for
  nearby lane/road markings and fill `img_metas['map_pts']/['map_mask']` in the
  model frame, mirroring `finetune_loader._load_map`.
