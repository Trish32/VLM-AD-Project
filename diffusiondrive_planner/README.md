# DiffusionDrive

Truncated diffusion policy for end-to-end planning. CVPR 2025 Highlight.
Reference: https://github.com/hustvl/DiffusionDrive — **two branches, two different systems**:

| Branch | Base system | Backbone | Reported |
|---|---|---|---|
| `nusc`  | SparseDrive | ResNet-50 | L2 0.27/0.54/0.90m (avg 0.57), collision 0.03/0.05/0.16% (avg 0.08) |
| `main`  | Transfuser/NAVSIM | ResNet-34, 60M | 88.1 PDMS on navtest |

A vanilla diffusion planner starts from Gaussian noise at t=999 and denoises ~100 steps.
DiffusionDrive starts from **anchor trajectories plus a little noise** and denoises **2 steps**
inside a truncated schedule. That is the whole "10× fewer denoising steps" claim — not
distillation, just a far better starting point.

## Visualization

![DiffusionDrive on nuScenes scene-0916](assets/demo_scene-0916.gif)

**Full-resolution video:** [scene-0916](assets/demo_scene-0916.mp4) (1920×1080, 20 s) ·
[scene-0103](assets/demo_scene-0103.mp4) · [full-res still](assets/demo_scene-0916_still.png)

`CAM_FRONT` carries the plan re-projected onto the road as the ego's swept corridor, alongside
the five remaining cameras and a BEV with the online map, tracked agents and their forecasts.
The plan is drawn in the three layers that make the truncated-diffusion story visible: the
command-selected kmeans **anchors** the denoiser is seeded from, the six **denoised modes**
shaded by confidence, and the **top-1** mode the controller follows. Colours follow the paper's
own legend (`autumn` top-1, `winter` other modes).

Three sources are kept visually distinct, because conflating them would claim more than the
model does:

| | shown as | what it is |
|---|---|---|
| **planned** | orange | derived from `final_planning` — DiffusionDrive's 3 s trajectory |
| **measured** | cyan | nuScenes CAN bus / IMU: what the human driver actually did |
| **nav command** | green | `gt_ego_fut_cmd`, the route command fed *into* the planner |

The model outputs a path, not actuator commands, so steering / throttle / brake come from a
pure-pursuit + PI controller closed around that path ([tools/demo_control.py](tools/demo_control.py),
unit-tested in [tests/test_demo_control.py](tests/test_demo_control.py)). The steering ratio
is not a spec-sheet number — it is fit per scene from the CAN yaw rate and speed, and comes
out at **15.5:1** on scene-0916, which is right for the Renault Zoe that nuScenes drives.
Action labels are the nav command plus a manoeuvre (lateral × longitudinal) read off the plan's
own geometry.

Gauges redraw every output frame against the ~100 Hz `steeranglefeedback` / `ms_imu`
channels, while the cameras and BEV step at the planner's 2 Hz — so the video shows the rate
gap rather than hiding it.

Known and expected: at low speed the 3 s plan is only ~6 m long, which is shorter than the
near-field the front camera can see, so the corridor shrinks to a sliver at the bottom of
`CAM_FRONT`. The BEV still shows the whole plan.

## Metric results

Open-loop planning on nuScenes **mini_val**, scored with **upstream's own `PlanningMetric`**
rather than a reimplementation, so the numbers are comparable to the paper by construction.

| L2 (m) ↓ | 1.0 s | 2.0 s | 3.0 s | avg |
|---|---|---|---|---|
| **ours** — mini_val, 69 scored samples | **0.2433** | **0.5490** | **0.9603** | **0.5842** |
| paper — full val | 0.27 | 0.54 | 0.90 | 0.57 |

**Within 2.5 % of the published average on ~10× fewer samples**, and 2 s is nearly exact.

Collision is 0.161 % vs the paper's 0.08 %, which is **not meaningful at n=69** — the entire
figure is one event in the 3 s bucket, where a single collision is worth ~0.5 pp. L2 is the
signal here; collision is noise at this sample size.

Per scene, the two mini_val scenes behave very differently:

| scene | L2 1 s / 2 s / 3 s | character |
|---|---|---|
| `scene-0916` | **0.145 / 0.323 / 0.560** | parking lot; full steering range (−273°→+251°), all three nav commands |
| `scene-0103` | 0.343 / 0.780 / 1.371 | decelerates 8.9→1.8 m/s for a turning car; carries nearly all the error |
| mini_val | 0.243 / 0.549 / 0.960 | the 69 fully-scored samples |

`scene-0103` is where the planner under-predicts a hard brake — visible as a spike in the
video's L2 trace.

L2 here is upstream's `PlanningMetric` definition — mean per-step displacement up to the
horizon, with samples whose 3 s future is unlogged dropped, **not** the endpoint error. The
number rendered on the video reproduces `eval_planning.py` to 1e-4, so it is the same number as
the table above.

## Fidelity chain

Each step is checked independently, so a failure localises instead of showing up as a bad metric:

| what | result |
|---|---|
| official `diffusiondrive_nusc_stage2.pth` → ported planner | **0 missing / 0 unexpected**, 128/128 tensors, no shape mismatches |
| our `_run_block` vs upstream's 22-slot loop, real sample, official weights | `plan_reg` **0.000e+00** · `plan_cls` **0.000e+00** — exact, not a tolerance |
| our `TruncatedDDIM` vs `diffusers.DDIMScheduler` | agree to **1e-6** (21 tests) |
| our `TruncatedDDIM` swapped into the real head, scored | L2 avg **0.5834** vs 0.5842 — below the run-to-run spread of the unseeded noise draw |
| `dfa_torch` vs the real CUDA kernel, Tesla T4 | max abs diff **4.578e-05** |

The mini_val run exercises `dfa_torch` and `attention_compat` throughout the detection, map and
motion heads, so the metric confirms both end to end — not just the planner.

## What is actually ported

The adaptation layer is deliberately thin:

| file | what it replaces |
|---|---|
| `truncated_diffusion.py` | the DDIM schedule, asymmetric normalisation, delta↔waypoint conversion |
| `plan_query.py` | anchor query bank (3 commands × 6 kmeans anchors), trajectory + time embeddings |
| `traj_pooler.py` | deformable aggregation **along the trajectory** — what makes the denoiser image-aware |
| `diff_head.py`, `ffn.py`, `diff_planner.py` | `diff_refine` + FiLM modulation + the ×2 `diff_operation_order` loop |
| `dfa_torch.py` | the `deformable_aggregation_ext` CUDA kernel, in pure PyTorch |
| `attention_compat.py` | `MultiheadFlashAttention` → SDPA, state-dict compatible so the checkpoint still loads 0/0 |


## Local setup

Both branches cloned and pinned (`upstream` = main/NAVSIM 9b52ed0, `upstream-nusc` =
nusc ae54fd8, as a git worktree sharing objects). The nusc plugin **imports and unit-tests
on Mac** in the `uniad2.0` env. Three blockers cleared — full detail in `bug_log.txt`:

1. unguarded `flash_attn` import at module scope → guarded, raises at use site
2. `deformable_aggregation_ext` CUDA kernel → pure-PyTorch equivalent in `dfa_torch.py`
3. `diffusers`/`huggingface_hub` `cached_download` break → pinned 0.27.2 / 0.25.2

Upstream edits are captured in `patches/0001-*.patch` so upstream stays re-pullable.

```bash
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
  conda run -n uniad2.0 python -m pytest diffusiondrive_planner/tests -q   # 8 passed, 1 skipped
```

**Carry-over risk:** `dfa_torch.py` is validated against a literal transcription of the
CUDA kernel, not against the kernel itself. That catches misreadings of the kernel but
not a misreading shared by both, so it needs a real GPU before any metric is trusted.

No GPU provisioning required — [tools/make_cuda_validation_bundle.py](tools/make_cuda_validation_bundle.py)
emits a **single self-contained file** to paste into a free Colab or Kaggle GPU notebook:

```bash
python diffusiondrive_planner/tools/make_cuda_validation_bundle.py
# -> diffusiondrive_planner/tools/validate_dfa_cuda.py  (22 KB, no deps beyond torch+nvcc)
```

It inlines the CUDA sources and the pure-PyTorch implementation, JIT-compiles the kernel
with `cpp_extension.load` (no `setup.py`, nothing installed), and prints a PASS/FAIL table
over three tiny cases plus one realistic 900×13×256 case. It also patches
`#include <THC/THCAtomics.cuh>` → `<ATen/cuda/Atomic.cuh>`, since THC was removed in
PyTorch 1.11 and upstream's source predates that.

## Stage 1 — nuScenes

Port target: `EgoPlanner` -> truncated-diffusion planner. Full spec in **[DESIGN.md](DESIGN.md)**.

- [x] Read `motion_planning_head_v13.py` against our `motion_planning.py`; write down every
      structural delta. → [DESIGN.md](DESIGN.md). Headline: the plan head goes from 3 modes
      (== driving commands) to 3 commands × 6 kmeans anchors, regresses **waypoint deltas**
      not positions, and denoises 2 steps from an anchor seeded at t=8.
- [x] Truncated diffusion schedule + normalization + delta conversion →
      [truncated_diffusion.py](truncated_diffusion.py), validated to 1e-6 against the real
      `diffusers.DDIMScheduler` (21 tests). Two traps caught, see `bug_log.txt`.
- [x] `plan_anchor` `(3, 6, 6, 2)` **regenerated from raw nuScenes** →
      [tools/gen_plan_anchors.py](tools/gen_plan_anchors.py). Surfaced that upstream's
      shipped `kmeans_plan.py` emits the wrong shape (per-command path is commented out)
      and that the frame is x=lateral / y=forward — both in `bug_log.txt`.
- [x] Anchor query bank (3 commands × 6 modes, command-selected) + trajectory sine
      embedding + time embedding → [plan_query.py](plan_query.py), byte-exact against
      upstream's own functions (20 tests).
- [x] `traj_pooler` — deformable aggregation along the trajectory →
      [traj_pooler.py](traj_pooler.py). Keypoints and projection byte-exact vs upstream;
      **loads `V1TrajPooler`'s state_dict at 0 missing / 0 unexpected** (11 tests).
- [x] `diff_refine` + `modulation` → [diff_head.py](diff_head.py) (byte-exact, state_dict 0/0).
- [x] Attention without flash-attn → [attention_compat.py](attention_compat.py); SDPA on
      torch≥2, manual softmax on torch 1.12. State_dict keys match upstream exactly.
- [x] `AsymmetricFFN` → [ffn.py](ffn.py) (byte-exact, state_dict 0/0).
- [x] **The ×2 `diff_operation_order` loop** → [diff_planner.py](diff_planner.py).
      Wiring tests assert the output actually depends on image features, agents, anchor
      queries and the anchor itself — the failures component tests cannot see.
- [x] **Official nusc checkpoint loads 0 missing / 0 unexpected** into the ported planner
      (128/128 tensors, no shape mismatches) → [tests/test_checkpoint_load.py](tests/test_checkpoint_load.py).
      `plan_anchor` IS in the checkpoint at `(3,6,6,2)`, so the regenerated mini anchors
      are overwritten and cannot affect reproduction. Forward runs with trained weights.
- [x] **Detector connected** → [tools/run_detector.py](tools/run_detector.py). Upstream's
      full SparseDrive stack runs on nuScenes mini, on CPU, no flash-attn, no CUDA — using
      our `dfa_torch` and `MultiheadAttentionCompat` as drop-in substitutes.
- [x] **Wiring confirmed bit-identical** → [tools/compare_planner.py](tools/compare_planner.py).
      Our `_run_block` vs upstream's 22-slot loop on a real sample with official weights:
      `plan_reg` and `plan_cls` both **0.000e+00** max abs diff.
- [x] **Metric reproduced** on mini_val (81 samples), scored with upstream's own
      `PlanningMetric` → [tools/eval_planning.py](tools/eval_planning.py):

      | metric              | 1.0s   | 2.0s   | 3.0s   | avg    |
      |---------------------|--------|--------|--------|--------|
      | L2 (ours, mini_val) | 0.2433 | 0.5490 | 0.9603 | 0.5842 |
      | L2 (paper, full val)| 0.27   | 0.54   | 0.90   | 0.57   |

      **within 2.5% on ~10x fewer samples.** Collision 0.161% vs 0.08% is not meaningful
      at n=81 — the whole figure is one event in the 3s bucket.
      This run exercises `dfa_torch` and `attention_compat` throughout the det/map/motion
      heads, so the metric confirms both end to end.
- [x] **Our `TruncatedDDIM` scored inside the real head** (`--scheduler ours`, which swaps
      it in behind diffusers' interface): L2 avg **0.5834** vs **0.5842** with upstream's
      scheduler, collision identical. The 0.0008 gap is below the run-to-run spread from
      the unseeded noise draw (0.5816–0.5842 observed), so the outer denoising loop —
      the one part `compare_planner.py` deliberately excluded — is confirmed in situ.
      

Get the checkpoint with:
```bash
curl -L -o diffusiondrive_planner/checkpoints/diffusiondrive_nusc_stage2.pth \
  https://huggingface.co/hustvl/DiffusionDrive/resolve/main/diffusiondrive_nusc_stage2.pth
```
- [ ] Load the official nusc checkpoint 0/0; confirm it carries `plan_anchor`.
- [ ] Load the official `nusc` checkpoint, 0 missing / 0 unexpected. Perception weights should
      map straight onto our existing SparseDrive port; only the head is new.
- [ ] Reproduce L2 and collision on our eval split. Note our split is nuScenes **mini_val**
      (~81 planning samples) so absolute numbers will be noisy vs the paper's full val — compare
      against our own SparseDrive baseline (plan L2 1.50/2.69/4.03m) as the controlled A/B.
- [ ] Decide whether to pull full nuScenes val; mini is too small to trust a 0.57m claim.

## Demo video

A qualitative video in the shape of upstream's `final_github.mp4`, with control signals and
action labels, on one nuScenes mini_val scene →
[tools/make_demo_video.py](tools/make_demo_video.py).

```bash
cd diffusiondrive_planner/upstream-nusc
PYTHONPATH=.:../.. python ../tools/make_demo_video.py \
    --config projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
    --results ../data/planning_results_ours.pt \
    --scene scene-0916 --out ../data/demo_diffusiondrive_scene-0916.mp4
```

It **re-renders the cached eval outputs — it does not re-run the model.**
`eval_planning.py` already dumped every planner output for all 81 mini_val samples, so
this is a pure rendering pass: ~70 s on the Mac, CPU only, no checkpoint forward.
`--results planning_results_ours.pt` renders our `TruncatedDDIM`;
`planning_results.pt` renders upstream's scheduler.

What each panel shows, and where its numbers come from, is described under
[Visualization](#visualization). Four frame/format traps had to be settled before any of it
was trustworthy — including one that silently blanked the road overlay on scene-0103 — and
they are written up in `bug_log.txt`.

## Stage 2 — NAVSIM 

This is the free-GPU training target. 60M params on ResNet-34 is the only model in this repo
that trains end-to-end on a single 16GB T4.

- [ ] **Start the OpenScene/NAVSIM download on day 1, in parallel with stage 1.** navtrain +
      navtest is the gating item for the whole week — it is a data-transfer problem, not a
      modelling one, and it is the single most likely reason week 1 slips.
- [ ] Port the Transfuser-style NAVSIM backbone (ResNet-34 via `timm`) + the same
      truncated-diffusion head.
- [ ] Reproduce 88.1 PDMS on navtest from the official checkpoint.
- [ ] Then fine-tune from scratch via `common.trainer.ResumableTrainer` on Kaggle and see how
      close the free-tier budget gets. This is the honest end-to-end test of the compute setup.

## Open questions
- Exact denoising step count and anchor vocabulary size are not in the README; read them out of
  the config (`projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py`).
- `nusc` branch is stage-2 only — confirm whether stage-1 SparseDrive weights are a prerequisite
  or whether the released checkpoint is self-contained.
