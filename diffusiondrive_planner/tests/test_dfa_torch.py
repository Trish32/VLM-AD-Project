"""Validate the pure-PyTorch DFA against a literal transcription of the CUDA kernel.

There is no GPU on this machine, so the vectorized `grid_sample` implementation cannot
be diffed against `deformable_aggregation_ext` directly. Instead it is diffed against
`dfa_naive` below — an independent, deliberately slow transcription of
`ops/src/deformable_aggregation_cuda.cu`, written from the kernel's own arithmetic
(explicit floor, explicit corner bounds checks, explicit index decomposition).

The two implementations share no code and no reasoning: one reasons in grid_sample's
normalized [-1, 1] space with align_corners=False, the other in raw pixel offsets. If
the mapping between those is wrong, they disagree. That is exactly the class of bug
that would otherwise survive until a GPU eval produced quietly-wrong mAP.

`test_matches_cuda_extension` runs the real kernel and is skipped off-GPU — run it on
the GCP box before trusting any number produced with the fallback.

Run: PYTHONPATH=. conda run -n uniad2.0 python -m pytest diffusiondrive_planner/tests -q
"""

import math

import pytest
import torch

from diffusiondrive_planner.dfa_torch import deformable_aggregation_torch

B, A, P = 2, 3, 4
NUM_CAMS, NUM_SCALE = 2, 3
C, G = 8, 2
SHAPES = [[(6, 8), (3, 4), (2, 2)], [(5, 7), (3, 3), (2, 3)]]  # per cam, per scale


def build_inputs(seed=0, dtype=torch.float64, edge_cases=True):
    """Random but reproducible inputs, with border/outside locations mixed in."""
    g = torch.Generator().manual_seed(seed)
    sizes = [h * w for cam in SHAPES for (h, w) in cam]
    num_feat = sum(sizes)

    feat = torch.randn(B, num_feat, C, generator=g, dtype=dtype)
    spatial_shape = torch.tensor(SHAPES, dtype=torch.int64)
    starts = torch.tensor(sizes, dtype=torch.int64).cumsum(0) - torch.tensor(
        sizes, dtype=torch.int64
    )
    scale_start_index = starts.reshape(NUM_CAMS, NUM_SCALE)

    loc = torch.rand(B, A, P, NUM_CAMS, 2, generator=g, dtype=dtype)
    if edge_cases:
        # Exercise the kernel's strict early-return: exactly 0, exactly 1, and outside.
        loc[0, 0, 0, 0] = torch.tensor([0.0, 0.5], dtype=dtype)
        loc[0, 0, 1, 0] = torch.tensor([1.0, 0.5], dtype=dtype)
        loc[0, 1, 0, 0] = torch.tensor([-0.3, 0.5], dtype=dtype)
        loc[0, 1, 1, 0] = torch.tensor([0.5, 1.7], dtype=dtype)
        # Just inside the border, where the -0.5 offset makes corners fall out of range
        loc[1, 0, 0, 1] = torch.tensor([1e-6, 1e-6], dtype=dtype)
        loc[1, 0, 1, 1] = torch.tensor([1 - 1e-6, 1 - 1e-6], dtype=dtype)

    weights = torch.randn(B, A, P, NUM_CAMS, NUM_SCALE, G, generator=g, dtype=dtype)
    return feat, spatial_shape, scale_start_index, loc, weights


def _bilinear(feat_bcam, h, w, h_im, w_im, num_embeds):
    """Transcribed from bilinear_sampling() in the .cu, including its bounds checks."""
    h_low, w_low = math.floor(h_im), math.floor(w_im)
    h_high, w_high = h_low + 1, w_low + 1
    lh, lw = h_im - h_low, w_im - w_low
    hh, hw = 1 - lh, 1 - lw

    def get(hh_idx, ww_idx):
        return feat_bcam[hh_idx * w + ww_idx]  # (C,)

    zero = torch.zeros(num_embeds, dtype=feat_bcam.dtype)
    v1 = get(h_low, w_low) if (h_low >= 0 and w_low >= 0) else zero
    v2 = get(h_low, w_high) if (h_low >= 0 and w_high <= w - 1) else zero
    v3 = get(h_high, w_low) if (h_high <= h - 1 and w_low >= 0) else zero
    v4 = get(h_high, w_high) if (h_high <= h - 1 and w_high <= w - 1) else zero

    return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4


def dfa_naive(feat, spatial_shape, scale_start_index, loc, weights):
    """Literal per-thread transcription of deformable_aggregation_kernel."""
    bs, _, num_embeds = feat.shape
    num_cams, num_scale = spatial_shape.shape[:2]
    num_anchors, num_pts = loc.shape[1:3]
    num_groups = weights.shape[-1]
    group_size = num_embeds // num_groups

    out = torch.zeros(bs, num_anchors, num_embeds, dtype=feat.dtype)
    for b in range(bs):
        for a in range(num_anchors):
            for p in range(num_pts):
                for cam in range(num_cams):
                    loc_w = float(loc[b, a, p, cam, 0])
                    loc_h = float(loc[b, a, p, cam, 1])
                    if loc_w <= 0 or loc_w >= 1:      # strict, per the kernel
                        continue
                    if loc_h <= 0 or loc_h >= 1:
                        continue
                    for s in range(num_scale):
                        h = int(spatial_shape[cam, s, 0])
                        w = int(spatial_shape[cam, s, 1])
                        start = int(scale_start_index[cam, s])
                        block = feat[b, start : start + h * w, :]
                        val = _bilinear(block, h, w, loc_h * h - 0.5, loc_w * w - 0.5, num_embeds)
                        for c in range(num_embeds):
                            out[b, a, c] += val[c] * weights[b, a, p, cam, s, c // group_size]
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_naive_kernel_transcription(seed):
    args = build_inputs(seed)
    got = deformable_aggregation_torch(*args)
    want = dfa_naive(*args)
    torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)


def test_border_and_outside_points_are_dropped_not_clamped():
    """A location exactly at 0 or 1 must contribute nothing.

    Clamping instead of dropping is the natural mistake, and it changes results only
    slightly — which is what makes it dangerous.
    """
    feat, shape, starts, loc, weights = build_inputs(0, edge_cases=False)
    loc = loc.clone()
    loc[:, :, :, :, :] = 0.5
    baseline = deformable_aggregation_torch(feat, shape, starts, loc, weights)

    for bad in (0.0, 1.0, -0.2, 1.4):
        probe = loc.clone()
        probe[0, 0, 0, 0, 0] = bad
        out = deformable_aggregation_torch(feat, shape, starts, probe, weights)
        # Only anchor 0 of batch 0 loses a contribution; everything else is untouched.
        assert not torch.allclose(out[0, 0], baseline[0, 0]), f"loc_w={bad} was not dropped"
        torch.testing.assert_close(out[0, 1:], baseline[0, 1:])
        torch.testing.assert_close(out[1], baseline[1])


def test_groups_are_contiguous_channel_blocks():
    """group = channel // (C // G). Using repeat() instead of repeat_interleave()
    would scramble which weight hits which channel while keeping every shape valid."""
    feat, shape, starts, loc, weights = build_inputs(0, edge_cases=False)
    group_size = C // G

    w_zeroed = weights.clone()
    w_zeroed[..., 0] = 0.0  # kill group 0 only
    out = deformable_aggregation_torch(feat, shape, starts, loc, w_zeroed)

    assert torch.allclose(out[..., :group_size], torch.zeros_like(out[..., :group_size])), \
        "zeroing group 0 should zero exactly the first channel block"
    assert not torch.allclose(out[..., group_size:], torch.zeros_like(out[..., group_size:]))


def test_gradients_flow_to_all_three_inputs():
    """Upstream's autograd.Function has its backward in the extension too, so the
    fallback must supply gradients by being differentiable end to end."""
    feat, shape, starts, loc, weights = build_inputs(0)
    feat = feat.clone().requires_grad_(True)
    loc = loc.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)

    deformable_aggregation_torch(feat, shape, starts, loc, weights).pow(2).sum().backward()

    for name, t in (("feat", feat), ("loc", loc), ("weights", weights)):
        assert t.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite gradient for {name}"
    assert feat.grad.abs().sum() > 0
    assert weights.grad.abs().sum() > 0


def test_zero_weights_give_zero_output():
    feat, shape, starts, loc, weights = build_inputs(0)
    out = deformable_aggregation_torch(feat, shape, starts, loc, torch.zeros_like(weights))
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_runs_on_mps_and_agrees_with_cpu():
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    args = build_inputs(0, dtype=torch.float32)
    cpu_out = deformable_aggregation_torch(*args)
    mps_args = [a.to("mps") if a.is_floating_point() else a.to("mps") for a in args]
    mps_out = deformable_aggregation_torch(*mps_args).cpu()
    torch.testing.assert_close(cpu_out, mps_out, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA extension")
def test_matches_cuda_extension():
    """Run this on the GCP box. It is the authoritative check."""
    from projects.mmdet3d_plugin.ops.deformable_aggregation import (
        DeformableAggregationFunction,
    )

    args = build_inputs(0, dtype=torch.float32)
    cuda_args = [a.cuda() for a in args]
    want = DeformableAggregationFunction.apply(*cuda_args)
    got = deformable_aggregation_torch(*cuda_args)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
