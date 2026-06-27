# KBM closed-loop simulator for Sparse4D-v3

A **Kinematic Bicycle Model** driving simulator that exercises the full Sparse4D-v3
end-to-end vision pipeline:

```
6-camera multi-view → DETECT → TRACK → MOTION → PLAN → CONTROL → KBM (ego dynamics)
```

The perception/prediction/planning stack runs on the **logged** nuScenes 6-camera
streams (sensors cannot be re-rendered on Apple Silicon). The **ego side is
genuinely closed-loop**: the KBM integrates the controller output, the controller
reads the model's planned trajectory *and the simulated ego speed*, and we measure
how far the closed-loop KBM path drifts from the logged human ego path
(= closed-loop tracking error).

![closed-loop simulator demo](sim_outputs/scene1_sparsedrive.gif)

*SparseDrive-style view: the 6 surround cameras with projected 3-D detections
(left) and the BEV panel (right) — HD-map lanes, tracked agents + motion
forecasts, the command-conditioned ego plan (green), the logged ego at the origin
(grey) and the closed-loop KBM ego (blue).*

## Files

| file | role |
|------|------|
| `kbm.py`        | Kinematic Bicycle Model — rear-axle, forward-Euler, world frame, torch/MPS |
| `controller.py` | planned waypoints + sim speed → (steering angle δ, acceleration a): pure-pursuit + PID |
| `bev.py`        | top-down BEV renderer (boxes, motion forecasts, ego plan, sim ego) → PNG + GIF |
| `simulator.py`  | closed-loop harness + per-frame log + aggregate metrics |

The pipeline itself lives at `/VLMProjects/sparse4d_vldrive/sparse4d_vl`
and is imported, not copied. The best planner checkpoint
(`checkpoints/train_v3_plan3/epoch_05.pt`, agent-map planner) is used by default.

## Run

```bash
cd /VLMProjects/simulator
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
  python simulator.py --scene 1 --max-frames 40 --bev
```

Flags: `--scene`, `--max-frames`, `--bev` (render PNGs + GIF to `--out`, default
`sim_outputs/`), `--checkpoint`, `--dataroot`.

Output per frame: #boxes, #tracks, command, plan endpoint, sim-vs-GT speed,
steering/accel, ego divergence, collision hits. Summary: mean/max/final ego
divergence vs the GT log, mean speed, collision frames, and a BEV GIF.

## Kinematic Bicycle Model

Rear-axle reference, control held constant over each frame interval (sub-stepped):

```
x   += v·cos(yaw)·dt
y   += v·sin(yaw)·dt
yaw += (v/L)·tan(δ)·dt          L = wheelbase = 2.85 m
v    = clamp(v + a·dt, 0, v_max)
```

The controller maps the ego plan to physical inputs: target speed from the first
planned step → longitudinal PID → acceleration `a`; pure-pursuit on a lookahead
waypoint → steering angle `δ`.

## What the simulator reveals (scene 1, mini)

The closed-loop ego drifts from the logged path (mean ≈ 13 m, final ≈ 25 m over
20 s) because the **planner under-predicts speed**: the human log drives ~8.5 m/s
while the planner targets ~5–6 m/s, so the KBM ego falls progressively behind.
This is the documented "lean planner" behaviour, and the simulator surfaces it
quantitatively. The drift is dominated by longitudinal (speed) error, not lateral.

### Honest limitations
- **Sensor replay**: cameras are the fixed nuScenes log, so once the ego diverges
  the perceived scene is no longer consistent with the ego's true location. This
  is the fundamental limit of sensor-replay closed-loop on a machine that can't
  re-render. Collisions are evaluated by transforming the logged obstacles into
  the simulated ego frame (so they reflect the diverged pose), but they inherit
  this inconsistency once divergence is large.
- The controller gains and KBM parameters (wheelbase, accel/decel limits) are
  reasonable defaults, not vehicle-identified.
