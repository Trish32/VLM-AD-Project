# e2e_pipeline — modular end-to-end AD stack

The integration layer between the perception ports and the planner: dense + object
perception fused into one scene representation, uncertainty plumbed end to end, and a
safety gate that can veto the learned plan.

![closed-loop rollout](assets/closed_loop_scene-0796.gif)

*scene-0796, 24 closed-loop steps at 15.3 m/s. **Left:** the front camera with 3-D
agent boxes and the chosen plan laid on the road surface. **Right:** the occupancy
branch reduced to what the planner consumes — drivable / unknown / obstacle — with all
six DiffusionDrive candidates coloured by the filter's verdict. **Below:** the five
metric families, live. The run ends with 0 collisions, 3 emergency brakes in 24 steps
and 67.8% route completion.*

One thing in that picture is not literally true, and the overlay says so rather than
hiding it. The loop simulates ego motion, so the ego drifts off the logged trajectory
— and nuScenes has no camera frame from a pose the car never occupied. The overlay
projects world-frame geometry through the **logged** camera: the projection is exact,
the viewpoint is logged, and the gap between the two poses is printed in the panel
header (`sim ego 19.2 m away`). Watch it grow across the clip. That number is also
why the teal corridor slides off-centre in the occupancy panel late in the run.


## Layout

```
e2e_pipeline/
├── scene.py          # unified ego-frame scene representation: Agent (+ Kalman
│                     #   covariance, + QCNet forecast), EgoState, footprint
│                     #   geometry. Frame contract lives here.
├── freespace.py      # FlashOcc (200,200,16,18) -> traversable / obstacle /
│                     #   unknown rasters + ESDF. Height band, Occ3D semantics,
│                     #   unknown-is-not-free.
├── uncertainty.py    # CV Kalman over Sparse4D track ids (detector score ->
│                     #   measurement noise) + QCNet Laplace scale -> collision
│                     #   probability, closed form (non-central chi-squared).
├── safety_filter.py  # four gates: drivable area, swept-footprint collision,
│                     #   KBM dynamic feasibility, probabilistic risk.
│                     #   Emergency brake when nothing survives.
├── vlm_planner.py    # the bridge: Qwen2.5-VL -> DrivingIntent -> DiffusionDrive
│                     #   anchors. Validation, IntentCache (rate decoupling),
│                     #   per-step speed conditioning.
├── pipeline.py       # per-frame orchestration behind Protocols for the four
│                     #   networks; frame conversion; occupancy rate decoupling.
├── closed_loop.py    # ClosedLoopRunner (pipeline -> controller -> KBM),
│                     #   GTWorldModel oracle, planner stand-ins.
├── metrics.py        # safety (SAT on oriented boxes) / route completion /
│                     #   comfort / prediction ADE-FDE / latency p50-p95.
├── visualize.py      # closed-loop rollout -> animated GIF
└── tests/            # 95 tests; each gate isolated by a test that fails it
```

## End-to-end architecture

```
                         6 x surround-view RGB
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
        Sparse4D v3                              FlashOcc
   3-D boxes + track ids                  (200,200,16,18) occupancy
        (LiDAR frame)                              │
                │  rotate -pi/2                    ▼  freespace.py
                │                        traversable / obstacle /
                │                          unknown + ESDF
                └──────────────────┬──────────────────┘
                                   ▼
                          scene.py  —  SceneRepresentation
                     one ego frame: +x fwd, +y left, metres
                   agents (+Kalman cov, +forecast) │ free space │ ego
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   uncertainty.py            vlm_planner.py            safety_filter.py
  Kalman cov  + QCNet     Qwen2.5-VL -> DrivingIntent   drivable area
  Laplace loc/scale/pi    {command, target_speed,       collision (swept)
        │                  light, hazard, conf}         dynamics (KBM)
        │                          │ validated, cached  risk  ◄──┐
        │                          ▼                       ▲     │
        │            DiffusionDrive anchors ──────────────►│     │
        │              K candidate trajectories            │     │
        └──────────── collision probability ───────────────┴─────┘
                                   │
                                   ▼  best feasible, else emergency brake
                        controller (pure pursuit)
                                   │
                                   ▼
                     kinematic bicycle model ──┐
                                   ▲           │
                                   └─ closed loop, ego state feeds back
```

Both perception branches run in parallel off the same cameras and converge on
`SceneRepresentation`. **Everything downstream reads only that** — the planner,
the risk model and the safety filter never touch a detector or an occupancy
tensor directly, which is what makes the four networks swappable behind
`Protocol`s.


Three properties the arrangement buys, none of which is free:

- **The VLM cannot cause a collision.** It sits upstream of the safety filter,
  so it narrows a candidate set the filter already vetted. Worst case it picks a
  worse feasible plan, or is overruled into a brake.
- **Latency stops being a defect.** Intent is slowly-varying, so `IntentCache`
  holds it while the planner and filter run at 20-55 Hz. ~7 s per VLM call is
  the cadence intent actually changes at.
- **Two perception branches, neither redundant.** Sparse4D reports only the 10
  scored nuScenes classes above threshold; FlashOcc marks a voxel occupied
  without needing a name for it.

## Frame contract

Everything in a `SceneRepresentation` is **ego frame at the current keyframe**:
`+x forward, +y left, +z up`, yaw CCW from `+x`, metres.

Neither branch produces this natively:

| Branch | Native frame | Conversion |
|---|---|---|
| Sparse4D | LiDAR (`+x` right, `+y` forward) | rotate by `−π/2` |
| FlashOcc | key-ego (camera-0's ego pose) | already ego-aligned |

The `−π/2` matches the offset recorded in the bevformer / sparse4d / bevfusion logs.
Getting it wrong is **silent** — a 90° BEV rotation raises nothing and simply makes
every clearance query answer about the wrong direction. It is pinned by
`tests/test_pipeline.py::test_lidar_forward_maps_to_ego_forward` rather than left to
inspection, and that test caught a sign error during development.

In production, pass the transform derived from the sample's `calibrated_sensor` record
via `meta["lidar_to_ego_yaw"]`; the module constant is a nominal default.

## Usage

```python
from e2e_pipeline import E2EPipeline, EgoState, FreeSpaceExtractor, GridConfig

pipe = E2EPipeline(
    detector=my_sparse4d_adapter,     # -> (boxes (N,9) lidar, track_ids, scores, labels)
    occupancy=my_flashocc_adapter,    # -> (semantics (200,200,16), mask_camera|None)
    planner=my_diffusiondrive_adapter,# -> (candidates (K,T,2) ego, scores (K,))
    forecaster=my_qcnet_adapter,      # -> {track_id: TrajectoryDistribution}  (optional)
    occupancy_every=2,                # run the dense branch every Nth frame
)

out = pipe.step(images, meta, EgoState(speed=8.0), command=1)
print(out.summary())
trajectory = out.trajectory           # (T, 2) — feed to the controller
```

The four networks sit behind `Protocol`s. They are separately-trained ports that share
no backbone or checkpoint and cost ~2–3 s/frame serially on an M3 Max, so a live
single-process loop is not the useful artifact. Supply live adapters when the
environments are up, or cached per-frame tensors when iterating on planning logic —
the integration logic under test is identical either way.

`forecaster` may be omitted, in which case every agent falls back to a
constant-velocity rollout with honestly growing covariance.

## Tests

```bash
conda run -n simple_bev_vldrive python -m pytest e2e_pipeline/tests/ -q
```

95 tests, ~0.7 s. Each gate has a test that isolates it: a candidate fine on every axis
except one, which must be rejected for that one reason.

---

## Results

Planning stack runs at **20-55 Hz** against a GT world model. 95 tests.

Four findings the closed loop surfaced, all invisible to unit tests:

- **Risk-gate behaviour is set by noise calibration, not geometry.** The same
  31 parked cars gave risk 0.475 with a detector-grade prior and 0.039 with a
  GT-grade one.
- **Collision geometry, not the compounding formula, was inflating risk.** A
  circumscribed disc over-reports broadside separation by 3.14 m; the
  rectangle support function took scene-0061 from `risk 1.000` to `0.43`.
- **The rollout was seeded at a constant 5 m/s regardless of the scene.** Logged
  frame-0 speeds in nuScenes-mini span 0 to 15.3 m/s, so the ego began every
  rollout at the wrong speed and diverged from the drivable corridor on step one,
  which then rejected every candidate. Seeding from the log instead:

  | scene | brakes/24 | route | collisions | min clearance |
  |---|---|---|---|---|
  | 0796 | 12 → **3** | 7.3% → **67.8%** | 2 → **0** | 0.00 → **3.88 m** |
  | 0061 | 24 → **12** | 7.4% → **55.5%** | 0 → **0** | 3.17 → 1.94 m |
  | 0655 | 11 → **11** | 17.8% → **39.5%** | 2 → **0** | 0.00 → **2.60 m** |

  Collisions went to zero on every scene where the ego actually drives. The
  "conservative filter" reading of the old numbers was wrong: the filter was
  reacting correctly to a badly initialised ego.
- **Route completion was scoring parked cars at 94%.** scene-0553's logged route
  is 4 cm of GPS jitter, and a `total > 0` guard divided by it happily — so a run
  that emergency-braked all 24 steps scored 94.3% completion. Routes under
  `MIN_ROUTE_M = 5.0` now report `completion: None`, not a number.

Method, ablations, retractions and caveats: **[RESULT.md](RESULT.md)**.

Before/after numbers for every change made to this pipeline, including the ones
that made it worse: **[EXPERIMENT.md](EXPERIMENT.md)**.

```bash
python -m e2e_pipeline.visualize                  # scene-0796, the GIF above
python -m pytest e2e_pipeline/tests/ -q
```
