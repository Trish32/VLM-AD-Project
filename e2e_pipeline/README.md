# e2e_pipeline — modular end-to-end AD stack

The integration layer between the perception ports and the planner: dense + object
perception fused into one scene representation, uncertainty plumbed end to end, and a
safety gate that can veto the learned plan.

```
multi-camera RGB
  ├── Sparse4D v3   (3D detection + tracking)  ──┐
  │                                              ├── unified scene representation
  └── FlashOcc      (3D occupancy → free space) ─┘
                                 │
                    QCNet / motion + planning heads
                                 │
                        DiffusionDrive  (K candidate plans)
                                 │
                       safety / feasibility filter
                                 │
                   controller → kinematic bicycle model
```

## What this adds

### 1. Dense geometry alongside object detection

`freespace.py` reduces FlashOcc's `(200, 200, 16, 18)` logit volume — 640k voxels — to
three `(200, 200)` rasters plus a distance field. That is ~0.5% of the voxel volume
and is directly consumable by a cost function.

The reduction is not "is this column occupied anywhere":

- **Height band.** Only the slab the vehicle body sweeps counts. Gantries and bridge
  decks are occupied voxels the ego drives under; the road surface is an occupied
  voxel it drives over. Default band is z ∈ [0.2, 2.2] m, derived from the grid config
  rather than hard-coded.
- **Semantics over geometry.** This is where Occ3D's 17 classes beat a binary grid:
  geometry alone cannot separate road from curb, since both are "stuff near the
  ground". `driveable_surface` is traversable, `sidewalk`/`terrain` are neither
  drivable nor walls, and a `pedestrian` voxel is an obstacle at any height.
- **Unknown ≠ free.** Occ3D ships `mask_camera` because unobserved voxels carry no
  information. A grid reporting "free" where no camera looked invites the planner into
  occlusion shadows, which is the exact failure occupancy was meant to fix.

Why it matters: Sparse4D only reports objects in the 10 scored nuScenes classes that
clear a score threshold. Debris, a jersey barrier at an odd angle, unclassified
construction equipment — none of those can enter an object-centric plan. Occupancy
does not need a name to mark a voxel occupied.

### 2. Safety / feasibility filter

`safety_filter.py` gates DiffusionDrive's candidates on four independent axes:

| Gate | Source | Catches |
|---|---|---|
| Drivable area | FlashOcc | plans that leave the road onto empty sidewalk |
| Collision | FlashOcc | swept footprint vs occupied space, class-agnostic |
| Dynamics | KBM limits | curvature, accel, lateral load the vehicle cannot deliver |
| Risk | Sparse4D + QCNet | probabilistic collision with uncertain agents |

Gates 1–2 are class-agnostic and close the taxonomy hole; gate 4 is identity-aware.
Neither is redundant.

**Re-ranking is legitimate here, and this is easy to get wrong.** In the SparseDrive
`EgoPlanner`, `ego_fut_mode=3` and the three modes *are* the driving commands — picking
among them would override the navigation intent and turn a commanded left turn into a
straight-ahead. DiffusionDrive is different: it has `3 commands × 6 anchors = 18` plan
queries and the command *selects* which set of 6 to use. The candidates reaching this
filter are genuine alternatives under one fixed intent, so choosing among them is a
safety decision, not a routing decision.

If every candidate fails, the filter emits an explicit emergency-brake profile and
flags the frame rather than returning the least-bad plan. `CandidateVerdict` records
*all* violated gates, not just the first — when everything is rejected you need to know
whether the scene is geometrically impossible or the planner is proposing undriveable
curvature.

### 3. Uncertainty-aware prediction and risk

`uncertainty.py` carries four sources, two of which already existed and were never
read:

- **Position / velocity** — Sparse4D emits boxes and stable track ids but no
  covariance; it is a detector with an instance bank, not a filter. A constant-velocity
  Kalman filter over the association it already provides yields a real 4×4 `P` for
  free. Detection score feeds the measurement noise, so a 0.3-score box enters as a
  *vague* observation rather than a confident one.
- **Prediction spread** — QCNet's decoder already regresses a Laplace scale per mode,
  per timestep, per dim (`to_scale_refine_pos`). The checkpoint predicts it and the
  loss trains it; nothing downstream looked at it.
- **Mode probability** — an agent that might turn left or go straight is genuinely
  bimodal. Collapsing to the argmax discards exactly the branch that hits you.
- **Plan multiplicity** — DiffusionDrive's 6 anchors give a candidate set to select
  among.

Combined as `σ²_total(t) = σ²_pred(a,k,t) + σ²_track(a,t)` — one is "where will it
choose to go", the other "where is it now and how fast", and neither informs the other.
Collision probability against a deterministic ego waypoint is then the probability that
a 2D Gaussian lands inside a disc, which is a non-central chi-squared CDF — closed form,
no sampling.

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

74 tests, ~0.5 s. Each gate has a test that isolates it: a candidate fine on every axis
except one, which must be rejected for that one reason.

## Known limits and tuning surfaces

- **Risk defaults are conservative.** `accel_noise=2.0 m²/s³` means a one-frame-old
  track carries roughly 5 m of positional uncertainty at a 3 s horizon. That is
  defensible physics for an agent of unknown intent, but it makes the risk gate fire
  readily. Calibrate against real tracks before trusting `max_risk=0.05` — the right
  way to set it is the collision-rate metric below, not intuition.
- **Disc-disc collision approximation.** `RiskModel` circumscribes both footprints with
  discs, which is conservative and cheap. The geometric gate uses the true swept
  rectangle; only the probabilistic gate approximates.
- **Rate decoupling does not ego-motion compensate.** `occupancy_every > 1` reuses the
  cached grid as-is. Under a metre of drift at N=2 and 2 Hz keyframes; if you raise N,
  warp it first with the same shift-and-rotate used for BEVFormer's `prev_bev`.
- **No live adapters ship here.** The `Protocol`s define the contract; the four
  adapters are the remaining integration work. The closed loop therefore
  runs against `GTWorldModel`, an oracle, which isolates the planning stack.

## Closed-loop evaluation

```python
from e2e_pipeline import ClosedLoopRunner, GTWorldModel, LoopConfig, format_report
runner = ClosedLoopRunner(GTWorldModel(nusc, scene_idx=0), planner, LoopConfig())
records, metrics = runner.run(command=2)
print(format_report(metrics))
```

The ego pose comes **only** from integrating the kinematic bicycle model under the
controls its own planner produced. Nothing snaps it back to the log — that is the
property open-loop replay cannot test, since open loop teleports the ego onto the
recorded path every frame and hides compounding error.

Five metric families ([metrics.py](metrics.py)), definitions pinned by tests:

| family | definition |
|---|---|
| safety | collision by **separating-axis test on oriented boxes** — two cars 2.2 m apart laterally do not touch, but circumscribed discs (r≈2.47 m) do, so a disc test inflates every number. Plus min clearance and TTC violations. |
| route | arc-length projection onto the reference polyline; furthest point reached, so stopping does not surrender ground covered. Lateral deviation reported separately. |
| comfort | longitudinal/lateral accel and jerk, RMS **and** violation counts — RMS alone hides one violent manoeuvre. Lateral accel from the realised path, not the steering command. |
| prediction | ADE/FDE against what agents actually did; agents leaving the scene are `unscoreable`, not wrong. |
| latency | per-stage p50/p95 — the tail breaks a control loop, the mean does not. |

### Results, 20 steps/scene, GT world model

| planner | brakes | collisions | mean completion |
|---|---|---|---|
| speed-aware stand-in | 55 | 21 | 28.5% |
| DiffusionDrive anchors (raw) | 93 | 20 | 28.3% |
| DiffusionDrive anchors (speed-conditioned) | 93 | 20 | 28.3% |

Planning stack runs at **20–55 Hz**, dominated by the safety filter (~25 ms).

### What the closed loop found

**Risk-gate behaviour is set by noise calibration, not geometry.** The first
rollout emergency-braked on all 20 steps at 2.4–5.6 m of clearance. Decomposed:
top single agent 0.124, median 0.0000, but 31 parked cars compounding through
`1 − Π(1−pₐ)` reached 0.475 against a 0.05 threshold. Cause: `t²·P_vv` dominates
σ, and a detector-grade `vel_noise=1.0 m/s` prior was applied to a GT oracle whose
velocities are exact. Same scene, same cars: **risk 0.475 → 0.039** from the prior
alone. `WorldModel.measurement_noise()` now makes the source declare its fidelity.

**Anchors are not a planner.** DiffusionDrive's anchors imply 0.1–14.6 m/s; offered
unmodified to a car at 5 m/s, four of six fail the dynamics gate on acceleration,
and the filter brakes with nothing to choose from. That is what the denoiser is
for — it conditions on ego state and deforms the anchor into something reachable.
A shape prior is not a candidate set.

### Honest limits on these numbers

- **Collisions are contaminated.** Agents are non-reactive log replay: when the ego
  stalls, a logged car drives through it. Every scene with collisions also has a
  high brake count — that is the artifact, not necessarily planner failure.
- **Completion is measured against an unreachable target.** 20 steps at ~5 m/s
  covers ~50 m against routes up to 163 m. scene-1100's 98.2% means a short route.
- **This is not DiffusionDrive.** `DiffPlanner.forward` needs `feature_maps` and
  `agent_feature` from the Sparse4D image backbone; once the ego diverges, those
  images describe a scene the car is not in. Only the anchor vocabulary is used.
- **The speed-conditioning variant does not work yet.** It clips the horizon-average
  speed, but the binding constraint is the per-step accel limit (±1.5 m/s over
  0.5 s), so it never engages — hence identical numbers to raw. Fixing it means
  conditioning the first step, not the endpoint.

## Measuring whether this helps

The existing `PlanMeter` computes collision as `‖plan[t] − agent_future[t]‖ < 2.0` —
point-to-point against annotated agents, with the ego as a point rather than a
4.6 × 1.8 m footprint. It is structurally incapable of registering a collision with
anything that is not an annotated agent, which is the whole category this pipeline
adds.

Two metrics that isolate the contribution:

1. **Collisions with occupied space that had no detection box.** The set an
   object-centric planner provably could not have avoided — clean attribution for the
   occupancy branch.
2. **Override rate** — `PipelineOutput.chosen_is_planner_favourite` aggregated over a
   run. Zero means the filter is inert; very high means the planner and the constraints
   disagree systematically and one of them is miscalibrated.

One caveat worth stating before anyone quotes L2: on nuScenes open-loop, L2 measures
agreement with the logged human trajectory, so a genuinely safer plan can score
*worse*. Watch it for regressions, but collision rate against occupancy is the metric
that matches the intent.
