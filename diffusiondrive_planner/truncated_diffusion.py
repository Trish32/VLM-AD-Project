"""Truncated diffusion planning — the mechanism that makes DiffusionDrive fast.

Extracted from `motion_planning_head_v13.py` (upstream nusc @ ae54fd8) as a standalone,
dependency-light module so it can be dropped onto our existing SparseDrive port without
dragging in mmdet3d or diffusers.

The idea
--------
A vanilla diffusion planner starts from pure Gaussian noise at t=999 and denoises for
~100 steps. DiffusionDrive instead starts from **anchor trajectories corrupted by a
little noise** and denoises for **2 steps inside a truncated schedule**. That is the
whole "10x fewer denoising steps" claim — not a distillation trick, just a much better
starting point.

Concretely, three deviations from textbook DDIM, all of which must be reproduced:

1. **Truncated training range.** Training samples t ~ U[0, 40), never U[0, 1000).
   Upstream's own comment: "magic number 40 means that we add little noise for each
   anchor". Sampling the full range trains the model for a regime inference never visits.
2. **Anchor-initialised inference.** Inference seeds with the anchor at a fixed t=8,
   `add_noise(anchor, noise, t=8)` — not `randn`. The anchors carry the mode structure;
   diffusion only refines them.
3. **prediction_type="sample".** The network predicts x0 directly, not epsilon. Wiring
   an epsilon-prediction head here trains and converges to something plausible-looking
   while being systematically wrong.

Also easy to miss: the diffusion runs on **consecutive waypoint deltas**, not absolute
positions, and in a normalized space with asymmetric per-axis constants (x and y are
scaled differently because forward and lateral motion have different ranges).

Verified against `diffusers.schedulers.DDIMScheduler` in `tests/test_truncated_diffusion.py`.
"""

from __future__ import annotations

import numpy as np
import torch

# Upstream normalization constants (motion_planning_head_v13.py:407-432). Asymmetric by
# axis: x is lateral in [-3, 3]; y is longitudinal, offset by 0.5 and spanning 8.1m.
_X_SCALE = 3.0
_Y_OFFSET = 0.5
_Y_SCALE = 8.1

# Upstream truncation constants.
TRAIN_TIMESTEP_MAX = 40   # training samples t ~ U[0, 40)
INFER_TRUNC_T = 8         # inference seeds the anchor at exactly this t
INFER_STEP_NUM = 2        # denoising steps at inference
INFER_STEP_RANGE = 40     # roll timesteps span [0, 40)


def normalize_traj(traj: torch.Tensor) -> torch.Tensor:
    """(..., 2) metres -> normalized ~[-1, 1]. Mirrors normalize_ego_fut_trajs.

    Note the asymmetry: x is clamped symmetrically, y is clamped to [0, 1] *before*
    being mapped to [-1, 1], so negative (backward) longitudinal motion is clipped away
    rather than represented.
    """
    x = (traj[..., 0:1] / _X_SCALE).clamp(-1, 1)
    y = ((traj[..., 1:2] + _Y_OFFSET) / _Y_SCALE).clamp(0, 1) * 2 - 1
    return torch.cat([x, y], dim=-1)


def denormalize_traj(traj: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize_traj. Deliberately unclamped, as upstream leaves it."""
    x = traj[..., 0:1] * _X_SCALE
    y = (traj[..., 1:2] + 1) / 2 * _Y_SCALE - _Y_OFFSET
    return torch.cat([x, y], dim=-1)


def anchor_to_deltas(anchor: torch.Tensor) -> torch.Tensor:
    """(..., T, 2) absolute waypoints -> (..., T, 2) consecutive deltas.

    Upstream prepends a zero waypoint (the ego's current position) and differences, so
    the first delta is measured from the ego rather than being dropped. The output keeps
    length T; recover positions with `.cumsum(dim=-2)`.
    """
    zeros = torch.zeros_like(anchor[..., :1, :])
    padded = torch.cat([zeros, anchor], dim=-2)
    return padded[..., 1:, :] - padded[..., :-1, :]


class TruncatedDDIM:
    """DDIM with `prediction_type="sample"`, restricted to the truncated regime.

    Reimplemented in plain torch (rather than depending on `diffusers`) so this drops
    into the SparseDrive port cleanly. `tests/test_truncated_diffusion.py` asserts
    agreement with the real `DDIMScheduler` to 1e-6, including the `set_timesteps(1000)`
    behaviour that makes `prev_timestep = t - 1`.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        # diffusers defaults that upstream never overrides. clip_sample is easy to miss
        # because it lives in the scheduler config rather than the call site, and it is
        # nearly a no-op in upstream's pipeline (they normalize x_start into [-1,1]
        # first) — right up until someone feeds it an unnormalized prediction.
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        # "scaled_linear": betas are linear in sqrt-space, then squared.
        # float32 deliberately, NOT float64: diffusers builds this in float32, and
        # computing it more accurately puts us ~2e-6 off the reference. Matching the
        # reference matters more here than being closer to the real cumprod.
        betas = torch.linspace(
            beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32
        ) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device=device, dtype=dtype)
        # DDIM's convention for the t=0 -> t=-1 boundary.
        self.final_alpha_cumprod = self.alphas_cumprod.new_tensor(1.0)

    def _alphas_on(self, device: torch.device) -> torch.Tensor:
        """alphas_cumprod on the caller's device.

        This class is deliberately NOT an nn.Module (it is usable standalone, and the
        schedule carries no learned state), so `.to(device)` on an enclosing module does
        not move this table. Resolve per call instead — otherwise the first MPS/CUDA
        forward dies with a device mismatch deep inside the denoising loop.
        """
        if self.alphas_cumprod.device != device:
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.final_alpha_cumprod = self.final_alpha_cumprod.to(device)
        return self.alphas_cumprod

    def add_noise(
        self, original: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alphas = self._alphas_on(original.device)
        a = alphas[timesteps.to(original.device)].to(original.dtype)
        while a.dim() < original.dim():
            a = a.unsqueeze(-1)
        return a.sqrt() * original + (1 - a).sqrt() * noise

    def step(
        self, model_output: torch.Tensor, timestep: int | torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        """One DDIM step with eta=0, where `model_output` IS the predicted x0.

        `prev_timestep = t - 1` because upstream calls `set_timesteps(1000)` — the full
        resolution — and then iterates over its own sparse `roll_timesteps`. Using the
        stride implied by 2 inference steps instead would take a far larger jump and
        change the result.
        """
        t = int(timestep)
        # diffusers: prev = t - num_train_timesteps // num_inference_steps. Upstream
        # calls set_timesteps(1000), so num_inference_steps == num_train_timesteps and
        # the stride is 1 — a single-timestep hop, despite only 2 steps being taken.
        prev_t = t - 1

        alphas = self._alphas_on(sample.device)
        alpha_t = alphas[t]
        alpha_prev = alphas[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

        pred_x0 = model_output
        # ORDER MATTERS: epsilon is derived from the UNCLIPPED prediction, and only then
        # is x0 clipped. diffusers does it in this order (with use_clipped_model_output
        # defaulting to False, so epsilon is never recomputed from the clipped value).
        # Clipping first is the natural reading and gives a small systematic error.
        pred_eps = (sample - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt()
        if self.clip_sample:
            pred_x0 = pred_x0.clamp(-self.clip_sample_range, self.clip_sample_range)
        return alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * pred_eps

    # --------------------------------------------------------------- schedules

    @staticmethod
    def train_timesteps(batch_size: int, device="cpu", generator=None) -> torch.Tensor:
        """t ~ U[0, 40) per sample — NOT U[0, num_train_timesteps)."""
        return torch.randint(
            0, TRAIN_TIMESTEP_MAX, (batch_size,), device=device, generator=generator
        )

    @staticmethod
    def inference_timesteps(step_num: int = INFER_STEP_NUM) -> torch.Tensor:
        """Descending roll timesteps: [20, 0] for the default 2 steps."""
        step_ratio = INFER_STEP_RANGE / step_num
        ts = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        return torch.from_numpy(ts)

    def seed_from_anchor(
        self, anchor_deltas: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Inference initialisation: normalized anchor + noise at t=INFER_TRUNC_T.

        This is the truncated-diffusion trick. Replacing it with `randn` still runs and
        still produces trajectories, just much worse ones — so it will not fail loudly.
        """
        normed = normalize_traj(anchor_deltas)
        if noise is None:
            noise = torch.randn_like(normed)
        t = torch.full(
            (normed.shape[0],), INFER_TRUNC_T, device=normed.device, dtype=torch.long
        )
        return self.add_noise(normed, noise, t)
