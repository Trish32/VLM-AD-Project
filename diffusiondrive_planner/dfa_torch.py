"""Pure-PyTorch Deformable Aggregation — CPU/MPS stand-in for `deformable_aggregation_ext`.

SparseDrive (and therefore DiffusionDrive's nusc branch) aggregates multi-view,
multi-scale features through a hand-written CUDA kernel. That kernel cannot build on
Apple Silicon, which blocks every local import and unit test in the mmdet3d plugin.

This is NOT the kind of stub this repo forbids. A forbidden stub returns zeros and
silently poisons results; this is a numerically-equivalent reimplementation, derived
line by line from `ops/src/deformable_aggregation_cuda.cu`, and checked against the
real kernel by `tests/test_dfa_matches_cuda.py` when a GPU is present.

Semantics recovered from the kernel
-----------------------------------
Thread index decomposes as ``[batch, anchor, pts, cam, scale, embed]`` with embed
fastest, and the output is accumulated with atomicAdd over ``(pts, cam, scale)``:

    out[b, a, :] = sum_{p,cam,s} w[b,a,p,cam,s, group(:)] * bilinear(feat[b,cam,s], loc[b,a,p,cam])

Four details that are easy to get wrong, all load-bearing:

1. ``sampling_location`` has **no scale axis** — the same (w, h) is reused at every
   scale — and it is stored **(w, h)**, not (h, w).
2. Points are **dropped**, not clamped: ``if (loc_w <= 0 || loc_w >= 1) return;``.
   Strict inequality, so a point exactly on the border contributes nothing.
3. ``h_im = loc_h * H - 0.5`` is exactly ``grid_sample`` with ``align_corners=False``
   under ``grid = 2 * loc - 1``, and the kernel's per-corner bounds checks are exactly
   ``padding_mode='zeros'``.
4. Groups are **contiguous channel blocks**: ``group = channel // (num_embeds //
   num_groups)``, so the weight expansion is ``repeat_interleave``, not ``repeat``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def deformable_aggregation_torch(
    mc_ms_feat: torch.Tensor,      # (B, num_feat, C) column-major multi-cam multi-scale
    spatial_shape: torch.Tensor,   # (num_cams, num_scale, 2) as [H, W]
    scale_start_index: torch.Tensor,  # (num_cams, num_scale) offsets into num_feat
    sampling_location: torch.Tensor,  # (B, num_anchors, num_pts, num_cams, 2) as [w, h]
    weights: torch.Tensor,         # (B, num_anchors, num_pts, num_cams, num_scale, num_groups)
) -> torch.Tensor:
    """Returns (B, num_anchors, C). Differentiable w.r.t. feat, location and weights."""
    bs, _, num_embeds = mc_ms_feat.shape
    num_cams, num_scale = spatial_shape.shape[:2]
    num_anchors, num_pts = sampling_location.shape[1:3]
    num_groups = weights.shape[-1]

    if num_embeds % num_groups:
        raise ValueError(f"num_embeds {num_embeds} not divisible by num_groups {num_groups}")
    group_size = num_embeds // num_groups

    shapes = spatial_shape.long().tolist()
    starts = scale_start_index.long().tolist()

    # Validity is a property of the location alone, so it is computed once and reused
    # across scales — matching the kernel, where the early return precedes any scale
    # dependent work. Strict inequality on both ends is intentional.
    loc_w = sampling_location[..., 0]
    loc_h = sampling_location[..., 1]
    valid = (loc_w > 0) & (loc_w < 1) & (loc_h > 0) & (loc_h < 1)  # (B, A, P, cams)

    # grid_sample wants [-1, 1] with align_corners=False; see docstring point 3.
    grid = sampling_location * 2.0 - 1.0

    out = mc_ms_feat.new_zeros(bs, num_anchors, num_embeds)

    for cam in range(num_cams):
        cam_valid = valid[..., cam]                       # (B, A, P)
        if not bool(cam_valid.any()):
            continue
        # Zero the grid where invalid so a NaN/inf location cannot leak through
        # grid_sample; the mask below is what actually removes the contribution.
        # zeros_like rather than a scalar/1-element tensor: torch 1.12 (the version in
        # the uniad2.0 env) requires matching rank in torch.where.
        cam_grid_raw = grid[:, :, :, cam, :]
        cam_grid = torch.where(
            cam_valid.unsqueeze(-1), cam_grid_raw, torch.zeros_like(cam_grid_raw)
        )                                                  # (B, A, P, 2)

        for s in range(num_scale):
            h, w = shapes[cam][s]
            start = starts[cam][s]
            # (B, H*W, C) -> (B, C, H, W)
            feat = mc_ms_feat[:, start : start + h * w, :]
            feat = feat.reshape(bs, h, w, num_embeds).permute(0, 3, 1, 2)

            sampled = F.grid_sample(
                feat,
                cam_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )                                              # (B, C, A, P)
            sampled = sampled.permute(0, 2, 3, 1)          # (B, A, P, C)

            w_cs = weights[:, :, :, cam, s, :]             # (B, A, P, G)
            w_cs = w_cs.repeat_interleave(group_size, dim=-1)  # contiguous groups
            out = out + (sampled * w_cs * cam_valid.unsqueeze(-1)).sum(dim=2)

    return out


class _DeformableAggregationTorchShim:
    """Adapter matching the `deformable_aggregation_ext` module surface.

    Only `deformable_aggregation_forward` is provided. The backward entry point is
    deliberately absent: upstream's autograd.Function is bypassed entirely by the
    fallback in `ops/deformable_aggregation.py`, which calls this function directly
    and lets autograd differentiate through grid_sample.
    """

    @staticmethod
    def deformable_aggregation_forward(
        mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights
    ):
        return deformable_aggregation_torch(
            mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights
        )
