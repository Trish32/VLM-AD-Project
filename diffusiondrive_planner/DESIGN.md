# DiffusionDrive vs SparseDrive — structural deltas

Derived by diffing `motion_planning_head_v13.py` (1324 lines) against
`motion_planning_head.py` (483 lines) on upstream's `nusc` branch @ ae54fd8 — the two
heads live in the same codebase, so the diff is apples-to-apples. Config:
`projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py`.

Target for the port: our SparseDrive `EgoPlanner` in
`~/VLMProjects/sparse4d_vldrive/sparse4d_vl/model/motion_planning.py`.

## The one idea

A vanilla diffusion planner starts from pure Gaussian noise at t=999 and denoises ~100
steps. DiffusionDrive starts from **anchor trajectories plus a little noise** and
denoises **2 steps inside a truncated schedule**. That is the entire "10× fewer
denoising steps" claim — not distillation, just a much better starting point.

```
vanilla:   randn ──(t=999 → 0, ~100 steps)──────────────────────────► trajectory
truncated: anchor + noise@t=8 ──(t=20 → 0, 2 steps)──► trajectory
```

Everything else in the head is machinery to condition that 2-step denoiser well.

## Deltas that change the port

| | SparseDrive (our port) | DiffusionDrive v13 |
|---|---|---|
| Ego modes | `ego_fut_mode=3`, **modes ARE commands** (L/S/R) | `3 commands × 6 anchors = 18` plan queries; the command **selects** a 6-anchor set |
| Plan anchors | per-command, learned | `kmeans_plan_6.npy`, shape `(3, 6, ego_fut_ts, 2)` |
| Representation | absolute waypoints | **consecutive deltas** (zero waypoint prepended, then differenced) |
| Plan head | direct regression, anchor + offset | truncated diffusion, `prediction_type="sample"` |
| Query features | anchor embedding | sine positional embedding of the *noisy waypoints* → `plan_pos_encoder` |
| Interaction | single layer | `diff_operation_order` × **2** |
| Image features | none in planner | `traj_pooler` — deformable aggregation **along the trajectory** |
| Time conditioning | none | `time_mlp(t)` → `modulation` (FiLM-style) |

`ego_fut_ts=6` (3s @ 2Hz) and `fut_ts=12 / fut_mode=6` for agent motion are unchanged
from our port.

## The diffusion contract

Implemented and validated in [`truncated_diffusion.py`](truncated_diffusion.py);
`tests/test_truncated_diffusion.py` asserts agreement with the real
`diffusers.DDIMScheduler` to 1e-6.

```python
DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear",
              prediction_type="sample")   # + diffusers defaults clip_sample=True, range=1.0
```

| Phase | Behaviour |
|---|---|
| Train | `t ~ U[0, 40)` per **sample**, then `repeat_interleave(ego_fut_mode)` so all 6 modes of a sample share `t` |
| Train init | `add_noise(normalize(anchor_deltas), noise, t)`, clamp to [-1,1], denormalize |
| Infer init | `add_noise(normalize(anchor_deltas), noise, t=8)` — **not** `randn` |
| Infer loop | `roll_timesteps = [20, 0]` (`step_ratio = 40/2`), descending |
| Per step | clamp to [-1,1] → denormalize → sine-embed → head → normalize x0 → `scheduler.step` |

Upstream's own comment on the range: *"magic number 40 means that we add little noise
for each anchor"*.

Normalization is asymmetric per axis (`motion_planning_head_v13.py:407`):

```
x: x/3,               clamp[-1, 1]
y: (y+0.5)/8.1,       clamp[0, 1], then *2-1     # backward motion is clipped away
```

## `diff_operation_order`, repeated ×2

```
traj_pooler → self_attn → norm → agent_cross_gnn → norm
            → anchor_cross_gnn → norm → ffn → norm → modulation → diff_refine
```

- `traj_pooler` — DFA on keypoints sampled along the current noisy trajectory. This is
  what makes the denoiser image-aware rather than purely a trajectory prior.
- `agent_cross_gnn` — attends to selected detection instance features.
- `anchor_cross_gnn` — attends to `cmd_plan_nav_query`, the command-selected anchor queries.
- `modulation` — injects `time_mlp(t)`.
- `map_cross_gnn` exists in the code but is **not** in the stage-2 config.

## Four traps

1. **`prediction_type="sample"`.** The network predicts x0 directly, not epsilon. An
   epsilon head trains fine and converges to something plausible while being wrong.
2. **`clip_sample=True` is a diffusers default upstream never overrides**, and inside
   `step` the epsilon is derived from the **unclipped** x0 *before* x0 is clipped.
   Clipping first is the natural reading and gives a small systematic error. (Cost us a
   real debug cycle — see `bug_log.txt`.)
3. **`prev_timestep = t - 1`**, because `set_timesteps(1000)` makes the stride 1 even
   though only 2 steps are taken. Using the stride implied by 2 steps takes a far
   bigger hop.
4. **Deltas, not positions.** Diffusion runs on consecutive differences; recover
   waypoints with `.cumsum(dim=-2)`.

## Port order

1. ~~`truncated_diffusion.py` — schedule, normalization, delta conversion~~ **done, validated**
2. ~~`plan_anchor` — regenerated from nuScenes~~ **done**, `tools/gen_plan_anchors.py` →
   `data/kmeans/kmeans_plan_6.npy` `(3, 6, 6, 2)`. See "Anchors" below.
3. ~~Anchor query bank: 3 commands × 6 modes, command-selected~~ **done, byte-exact**
4. ~~Trajectory sine embedding + `plan_pos_encoder` + `time_mlp`~~ **done, byte-exact**
   → [`plan_query.py`](plan_query.py), validated against upstream's own functions
5. ~~`traj_pooler` — DFA along the trajectory~~ **done** → [`traj_pooler.py`](traj_pooler.py),
   byte-exact keypoints/projection vs upstream and **state_dict-compatible with
   `V1TrajPooler` at 0 missing / 0 unexpected**
6. `diff_refine` head + the ×2 operation loop
7. Load the official nusc checkpoint 0/0, then reproduce L2 / collision

Steps 1–4 are pure and testable on the Mac. Step 5 onward needs the checkpoint and,
for any trustworthy number, a GPU (see the DFA carry-over risk in `bug_log.txt`).

## Three embeddings, three different contracts

Ported in [`plan_query.py`](plan_query.py). The widths are not interchangeable:

| what | source | hidden_dim | encoder |
|---|---|---|---|
| plan mode query | anchor **endpoint** `[...,-1,:]` | 256 (default) | `plan_anchor_encoder` |
| trajectory feature | **all** noisy waypoints, flattened | **128** | `plan_pos_encoder` (in 768) |
| diffusion time embed | scalar timestep | 256 | `time_mlp` |

`hidden_dim=128` for trajectories exists precisely so `ego_fut_ts * 128 = 768` matches
`plan_pos_encoder`'s declared input. Passing the 256 default gives 1536 and fails loudly
— the good case. Three others fail silently:

- **`gen_sineembed_for_position` returns `cat([pos_y, pos_x])` — y first.** Swapping keeps
  every shape valid and every downstream layer happy.
- **Embeddings run on raw metres × 2π, unnormalized.** Anchor endpoints reach ~40m,
  trajectory deltas are 0–7m. Normalizing first changes the embedding scale entirely.
- **`plan_mode_query` is command-major**: `(bs,3,K,C).flatten(1,2)` → `(bs,3K,C)`, recovered
  with `.view(bs,3,K,-1)`. Mode-major round-trips through identical shapes while pairing
  each command with the wrong anchors.

`SinusoidalPosEmb` (timestep) and `gen_sineembed_for_position` (position) use different
frequency laws and are **not** interchangeable despite both being "sine embeddings".

## `traj_pooler` — what makes the denoiser image-aware

Ported in [`traj_pooler.py`](traj_pooler.py) on top of `dfa_torch.py`. Without it the
planner is a trajectory prior that never looks at the cameras; with it, every denoising
step samples multi-view features along the *current noisy trajectory*.

```
trajs (deltas) --cumsum--> absolute --kps--> (bs,modal,T*5,3) --project--> (bs,modal,T*5,6,2)
instance_feature --weights_fc--> (bs,modal,6,4,T*5,8)
                            --DFA--> (bs,modal,256) --output_proj + residual-->
```

Note `pool_feature_from_traj` also exists as a **method on the head** — that copy is dead
code, commented out at both call sites. The live path is the `V1TrajPooler` layer.

Five traps:

1. **`forward` cumsums first.** The diffusion state is deltas; the pooler needs world
   positions. Feeding deltas straight through projects points near the ego and samples
   the wrong pixels, with no shape error. Pinned by `test_forward_cumsums_deltas`.
2. **Group is the fastest-varying weight index** — `reshape(bs, A, -1, G).softmax(dim=-2)`
   normalises jointly across cam × level × point per group. Same layout that cost days in
   the Sparse4D port.
3. **The camera encoder takes 12 numbers**, `projection_mat[:, :, :3].reshape(bs, cams, -1)`
   — top 3 rows of the 4×4, not all 16.
4. **Keypoint z is `ground_height + fix_height`** with `ground_height=-1.84023` and 5 fixed
   heights → `T*5 = 30` points, laid out **waypoint-major** (all heights of waypoint 0,
   then waypoint 1, …).
5. **`project_points` does not mask points behind the camera** — `clamp(z, min=1e-5)` folds
   them to huge coordinates, and the DFA's in-view test drops them. Masking earlier would
   change which points contribute.

## Anchors

Regenerated from raw nuScenes rather than downloaded, via
[`tools/gen_plan_anchors.py`](tools/gen_plan_anchors.py), which collapses upstream's
two-stage pipeline (`nuscenes_converter.py` → infos pkl → `kmeans_plan.py`) into one pass.

That immediately surfaced two things a downloaded artifact would have hidden:

**Upstream's shipped `kmeans_plan.py` does not reproduce upstream's artifact.** As
shipped it pools all commands and writes `(K, 6, 2)` to `kmeans_plan_vocab_K.npy`, but
the head indexes `plan_anchor[bs, cmd]` and needs `(3, K, 6, 2)` from
`kmeans_plan_K.npy`. The per-command path is present but commented out. `--mode
per_command` restores it; `--mode vocab` reproduces the shipped behaviour.

**The frame convention is asserted, not assumed.** x is lateral (right positive), y is
forward — inferred from the command rule (`ego_fut_trajs[-1][0] >= 2` → Right) and
checked against the data on every run, because a swapped axis rotates every anchor 90°
while keeping all shapes valid.

Corroboration that the interpretation is right: the head's normalization constants are
tight against real data. Mini gives deltas `x ∈ [-3.05, 2.07]` against the `x/3` clamp
at ±3.0, and `y ∈ [0.03, 7.33]` against `(y+0.5)/8.1`'s range of `[-0.5, 7.6]`. Both
axes *just* fit — which only happens if frame and units are read correctly.

**Mini caveat.** 344 usable samples split right 22 / left 30 / straight 292, versus
upstream's ~28k from full trainval. These are for pipeline validation, not training.
Mitigating this: `self.plan_anchor` is an `nn.Parameter(requires_grad=False)`, so it is
in the state_dict and **the official checkpoint overwrites it** — for stage-1
reproduction only the shape matters. Verify the ckpt actually carries a `plan_anchor`
key when it lands.
