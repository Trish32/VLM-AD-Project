"""Validate our truncated-diffusion port against the real diffusers DDIMScheduler.

`truncated_diffusion.py` reimplements the scheduler in plain torch so the planner can
drop onto our SparseDrive port without a diffusers dependency. That is only safe if it
matches, so every numeric path is diffed against `diffusers.schedulers.DDIMScheduler`
configured exactly as upstream does it.

Also pins the schedule constants themselves — the truncation to [0, 40), the t=8 seed,
and the [20, 0] roll — because those are the paper's contribution and a "cleanup" that
restores the textbook full-range schedule would silently gut the method.

Run: PYTHONPATH=. conda run -n uniad2.0 python -m pytest \
       diffusiondrive_planner/tests/test_truncated_diffusion.py -q
"""

import numpy as np
import pytest
import torch

from diffusiondrive_planner.truncated_diffusion import (
    INFER_TRUNC_T,
    TRAIN_TIMESTEP_MAX,
    TruncatedDDIM,
    anchor_to_deltas,
    denormalize_traj,
    normalize_traj,
)

DDIMScheduler = pytest.importorskip("diffusers.schedulers").DDIMScheduler

BS, EGO_FUT_MODE, EGO_FUT_TS = 2, 6, 6


def upstream_scheduler():
    """Exactly the construction at motion_planning_head_v13.py:290."""
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )


def test_alphas_cumprod_match():
    ours = TruncatedDDIM()
    theirs = upstream_scheduler()
    torch.testing.assert_close(
        ours.alphas_cumprod, theirs.alphas_cumprod.float(), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("t", [0, 1, 8, 20, 39, 100, 999])
def test_add_noise_matches(t):
    ours, theirs = TruncatedDDIM(), upstream_scheduler()
    torch.manual_seed(0)
    x0 = torch.randn(BS * EGO_FUT_MODE, EGO_FUT_TS, 2)
    noise = torch.randn_like(x0)
    ts = torch.full((x0.shape[0],), t, dtype=torch.long)

    torch.testing.assert_close(
        ours.add_noise(x0, noise, ts),
        theirs.add_noise(original_samples=x0, noise=noise, timesteps=ts),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("t", [0, 1, 8, 20, 39])
def test_ddim_step_matches(t):
    """The step is where prediction_type and the prev_timestep stride both bite."""
    ours = TruncatedDDIM()
    theirs = upstream_scheduler()
    theirs.set_timesteps(1000, "cpu")  # upstream does this; makes prev_timestep = t-1

    torch.manual_seed(1)
    sample = torch.randn(BS * EGO_FUT_MODE, EGO_FUT_TS, 2)
    x_start = torch.randn_like(sample)  # the network's predicted x0

    want = theirs.step(model_output=x_start, timestep=t, sample=sample).prev_sample
    got = ours.step(model_output=x_start, timestep=t, sample=sample)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_full_inference_loop_matches():
    """End-to-end: seed from anchor at t=8, then denoise over [20, 0]."""
    ours = TruncatedDDIM()
    theirs = upstream_scheduler()
    theirs.set_timesteps(1000, "cpu")

    torch.manual_seed(2)
    anchor = torch.randn(BS * EGO_FUT_MODE, EGO_FUT_TS, 2) * 0.5
    noise = torch.randn_like(anchor)
    trunc = torch.full((anchor.shape[0],), INFER_TRUNC_T, dtype=torch.long)

    normed = normalize_traj(anchor)
    img_ours = ours.add_noise(normed, noise, trunc)
    img_theirs = theirs.add_noise(original_samples=normed, noise=noise, timesteps=trunc)
    torch.testing.assert_close(img_ours, img_theirs, atol=1e-6, rtol=1e-6)

    # A fixed pseudo-network so both loops see identical model outputs.
    torch.manual_seed(3)
    preds = [torch.randn_like(anchor) * 0.3 for _ in range(2)]

    for k, pred in zip(TruncatedDDIM.inference_timesteps(), preds):
        img_ours = ours.step(pred, int(k), img_ours)
        img_theirs = theirs.step(model_output=pred, timestep=int(k), sample=img_theirs).prev_sample

    torch.testing.assert_close(img_ours, img_theirs, atol=1e-6, rtol=1e-6)


def test_seed_from_anchor_uses_t8_not_pure_noise():
    """The truncated seed must stay close to the anchor.

    If someone 'fixes' this to start from randn, the planner still runs and still
    produces trajectories — just much worse ones. So pin the property.
    """
    ours = TruncatedDDIM()
    torch.manual_seed(0)
    anchor = torch.rand(BS * EGO_FUT_MODE, EGO_FUT_TS, 2) * 2 - 1
    normed = normalize_traj(anchor)
    seeded = ours.seed_from_anchor(anchor)

    anchor_err = (seeded - normed).abs().mean()
    pure_noise_err = (torch.randn_like(normed) - normed).abs().mean()
    assert anchor_err < pure_noise_err / 2, (
        f"seed is not anchor-dominated: {anchor_err:.3f} vs {pure_noise_err:.3f}"
    )
    # alphas_cumprod[8] is very close to 1, so the anchor should barely move.
    assert float(ours.alphas_cumprod[INFER_TRUNC_T]) > 0.99


def test_schedule_constants_are_the_truncated_ones():
    ts = TruncatedDDIM.inference_timesteps()
    assert ts.tolist() == [20, 0], "default roll timesteps must be [20, 0]"
    assert TruncatedDDIM.inference_timesteps(4).tolist() == [30, 20, 10, 0]

    train = TruncatedDDIM.train_timesteps(4096)
    assert int(train.max()) < TRAIN_TIMESTEP_MAX
    assert int(train.min()) >= 0
    # Sanity that it is the truncated range, not the full one.
    assert int(train.max()) < 40 <= 1000


def test_normalize_roundtrip_inside_range():
    """Round-trip is exact only inside the clamp range; that asymmetry is upstream's."""
    x = torch.linspace(-2.9, 2.9, 32).unsqueeze(-1)
    y = torch.linspace(-0.4, 7.5, 32).unsqueeze(-1)
    traj = torch.cat([x, y], dim=-1)
    torch.testing.assert_close(denormalize_traj(normalize_traj(traj)), traj, atol=1e-5, rtol=1e-5)


def test_normalize_clamps_asymmetrically():
    """x clamps symmetrically; y clamps to [0,1] pre-rescale, so backward motion clips."""
    traj = torch.tensor([[[10.0, 100.0], [-10.0, -5.0]]])
    out = normalize_traj(traj)
    assert float(out[0, 0, 0]) == pytest.approx(1.0)
    assert float(out[0, 0, 1]) == pytest.approx(1.0)
    assert float(out[0, 1, 0]) == pytest.approx(-1.0)
    assert float(out[0, 1, 1]) == pytest.approx(-1.0)


def test_anchor_to_deltas_and_back():
    """Deltas must cumsum back to the original absolute waypoints."""
    torch.manual_seed(0)
    anchor = torch.randn(BS, EGO_FUT_MODE, EGO_FUT_TS, 2).cumsum(dim=-2)
    deltas = anchor_to_deltas(anchor)
    assert deltas.shape == anchor.shape, "length must be preserved (zero waypoint prepended)"
    torch.testing.assert_close(deltas.cumsum(dim=-2), anchor, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(deltas[..., 0, :], anchor[..., 0, :], atol=1e-6, rtol=1e-6)


def test_matches_upstream_anchor_delta_expression():
    """Byte-for-byte the expression at motion_planning_head_v13.py:718-720."""
    torch.manual_seed(0)
    plan_anchor = torch.randn(BS, EGO_FUT_MODE, EGO_FUT_TS, 2)
    zeros_cat = torch.zeros(BS, 6, 1, 2)
    cmd_plan_anchor = torch.cat([zeros_cat, plan_anchor], dim=2)
    want = cmd_plan_anchor[:, :, 1:, :] - cmd_plan_anchor[:, :, :-1, :]
    torch.testing.assert_close(anchor_to_deltas(plan_anchor), want, atol=0, rtol=0)


def test_train_timesteps_repeat_interleave_layout():
    """Upstream draws bs timesteps then repeat_interleave's by ego_fut_mode, so all
    modes of a sample share a timestep. repeat() instead would misalign them."""
    torch.manual_seed(0)
    t = TruncatedDDIM.train_timesteps(BS)
    repeated = t.repeat_interleave(EGO_FUT_MODE)
    assert repeated.shape == (BS * EGO_FUT_MODE,)
    for b in range(BS):
        block = repeated[b * EGO_FUT_MODE : (b + 1) * EGO_FUT_MODE]
        assert (block == t[b]).all(), "all modes of one sample must share its timestep"
