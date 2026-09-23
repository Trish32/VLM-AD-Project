"""Validate the plan-query port against upstream's own functions.

`plan_query.py` restates upstream's embedding code in plain torch. The `uniad2.0` env
can import the real plugin, so every ported function is diffed against its original
rather than merely being shape-checked — the same approach used for the ROS wire format
and the DDIM scheduler.

Run: PYTHONPATH=.:diffusiondrive_planner/upstream-nusc PYTORCH_ENABLE_MPS_FALLBACK=1 \
       conda run -n uniad2.0 python -m pytest diffusiondrive_planner/tests/test_plan_query.py -q
"""

import numpy as np
import pytest
import torch

from diffusiondrive_planner.plan_query import (
    EGO_FUT_MODE,
    EGO_FUT_TS,
    NUM_COMMANDS,
    PlanAnchorQueryBank,
    SinusoidalPosEmb,
    TimeEmbedder,
    TrajEmbedder,
    gen_sineembed_for_position,
    linear_relu_ln,
    select_anchor_by_command,
    select_by_command,
)

BS, EMBED = 2, 256
CMD_RIGHT, CMD_LEFT, CMD_STRAIGHT = 0, 1, 2


def upstream_sineembed():
    mod = pytest.importorskip(
        "projects.mmdet3d_plugin.models.attention",
        reason="run with PYTHONPATH including diffusiondrive_planner/upstream-nusc",
    )
    return mod.gen_sineembed_for_position


def upstream_sinusoidal_pos_emb():
    mod = pytest.importorskip(
        "projects.mmdet3d_plugin.models.motion.modules.conditional_unet1d",
        reason="run with PYTHONPATH including diffusiondrive_planner/upstream-nusc",
    )
    return mod.SinusoidalPosEmb


# ------------------------------------------------------------------ port fidelity

@pytest.mark.parametrize("hidden_dim", [128, 256])
@pytest.mark.parametrize("shape", [(4, 2), (BS, 3, EGO_FUT_MODE, 2), (BS * EGO_FUT_MODE, EGO_FUT_TS, 2)])
def test_sineembed_matches_upstream(hidden_dim, shape):
    ref = upstream_sineembed()
    torch.manual_seed(0)
    pos = torch.randn(*shape) * 5  # raw metres; embeddings are NOT normalized first
    torch.testing.assert_close(
        gen_sineembed_for_position(pos, hidden_dim), ref(pos, hidden_dim), atol=0, rtol=0
    )


def test_sineembed_is_y_first():
    """Output is cat([pos_y, pos_x]). Swapping keeps every shape valid."""
    pos_a = torch.tensor([[1.0, 0.0]])
    pos_b = torch.tensor([[0.0, 1.0]])
    ea, eb = gen_sineembed_for_position(pos_a, 128), gen_sineembed_for_position(pos_b, 128)
    half = 64
    # x=1,y=0 must put its signal in the SECOND half; y=1,x=0 in the first.
    assert ea[0, half:].abs().sum() > ea[0, :half].abs().sum()
    assert eb[0, :half].abs().sum() > eb[0, half:].abs().sum()


def test_sinusoidal_pos_emb_matches_upstream():
    ref_cls = upstream_sinusoidal_pos_emb()
    ours, ref = SinusoidalPosEmb(EMBED), ref_cls(EMBED)
    t = torch.arange(0, 40, dtype=torch.float32)
    torch.testing.assert_close(ours(t), ref(t), atol=0, rtol=0)


def test_sinusoidal_pos_emb_differs_from_position_embed():
    """The two sine embeddings are not interchangeable — different frequency law."""
    t = torch.tensor([8.0, 20.0])
    scalar = SinusoidalPosEmb(128)(t)
    positional = gen_sineembed_for_position(torch.stack([t, t], dim=-1), 128)
    assert not torch.allclose(scalar, positional)


def test_linear_relu_ln_structure_matches_upstream():
    mod = pytest.importorskip("projects.mmdet3d_plugin.models.blocks")
    ours = linear_relu_ln(EMBED, 1, 1, 768)
    ref = mod.linear_relu_ln(EMBED, 1, 1, 768)
    assert [type(l).__name__ for l in ours] == [type(l).__name__ for l in ref]
    assert ours[0].in_features == ref[0].in_features == 768
    assert ours[0].out_features == ref[0].out_features == EMBED


# ---------------------------------------------------------------------- semantics

@pytest.fixture
def anchors():
    torch.manual_seed(0)
    a = torch.randn(NUM_COMMANDS, EGO_FUT_MODE, EGO_FUT_TS, 2)
    # Make the commands distinguishable so selection errors are detectable.
    a[CMD_RIGHT, ..., 0] += 5.0
    a[CMD_LEFT, ..., 0] -= 5.0
    a[CMD_STRAIGHT, ..., 0] *= 0.01
    return a


def test_bank_shapes(anchors):
    bank = PlanAnchorQueryBank(anchors, EMBED)
    plan_anchor, plan_mode_query = bank(BS)
    assert plan_anchor.shape == (BS, NUM_COMMANDS, EGO_FUT_MODE, EGO_FUT_TS, 2)
    assert plan_mode_query.shape == (BS, 1, NUM_COMMANDS * EGO_FUT_MODE, EMBED)


def test_bank_rejects_upstreams_shipped_shape():
    """kmeans_plan.py as shipped emits (K, T, 2); that must fail loudly, not broadcast."""
    with pytest.raises(ValueError, match=r"must be \(3, K, T, 2\)"):
        PlanAnchorQueryBank(torch.randn(EGO_FUT_MODE, EGO_FUT_TS, 2), EMBED)


def test_anchor_is_frozen_parameter(anchors):
    """Must be a Parameter (in the state_dict, overwritten by the ckpt) but not trained."""
    bank = PlanAnchorQueryBank(anchors, EMBED)
    assert isinstance(bank.plan_anchor, torch.nn.Parameter)
    assert not bank.plan_anchor.requires_grad
    assert "plan_anchor" in bank.state_dict()


def test_command_selection_picks_the_right_anchors(anchors):
    bank = PlanAnchorQueryBank(anchors, EMBED)
    plan_anchor, _ = bank(BS)
    cmd = torch.tensor([CMD_RIGHT, CMD_LEFT])
    picked = select_anchor_by_command(plan_anchor, cmd)
    assert picked.shape == (BS, EGO_FUT_MODE, EGO_FUT_TS, 2)
    assert picked[0, ..., 0].mean() > 2.0, "sample 0 asked for RIGHT"
    assert picked[1, ..., 0].mean() < -2.0, "sample 1 asked for LEFT"


def test_query_layout_is_command_major(anchors):
    """flatten(1,2) then view(bs,3,K,-1) must round-trip to the same grouping.

    Building the query mode-major passes every shape check while pairing each command
    with the wrong anchor set — this is the test that catches it.
    """
    bank = PlanAnchorQueryBank(anchors, EMBED)
    plan_anchor, plan_mode_query = bank(BS)

    for cmd_idx in range(NUM_COMMANDS):
        cmd = torch.full((BS,), cmd_idx, dtype=torch.long)
        selected = select_by_command(plan_mode_query, cmd, EGO_FUT_MODE)
        # Recompute the query for just this command's anchors and compare.
        pos = gen_sineembed_for_position(plan_anchor[:, cmd_idx, :, -1, :])
        expected = bank.plan_anchor_encoder(pos)
        torch.testing.assert_close(selected, expected, atol=1e-5, rtol=1e-5)


def test_traj_embedder_width_matches_encoder():
    """hidden_dim=128 exists so ego_fut_ts*128 == 768 == plan_pos_encoder input."""
    emb = TrajEmbedder(EMBED, EGO_FUT_TS)
    assert emb.input_dims == 768
    assert emb.plan_pos_encoder[0].in_features == 768

    traj = torch.randn(BS * EGO_FUT_MODE, EGO_FUT_TS, 2)
    out = emb(traj, BS, EGO_FUT_MODE)
    assert out.shape == (BS, EGO_FUT_MODE, EMBED)
    assert torch.isfinite(out).all()


def test_traj_embedder_reproduces_upstream_pipeline():
    """The exact three-line sequence at motion_planning_head_v13.py:1040-1044."""
    ref = upstream_sineembed()
    emb = TrajEmbedder(EMBED, EGO_FUT_TS)
    torch.manual_seed(0)
    traj = torch.randn(BS * EGO_FUT_MODE, EGO_FUT_TS, 2)

    want = emb.plan_pos_encoder(ref(traj, 128).flatten(-2)).view(BS, EGO_FUT_MODE, -1)
    torch.testing.assert_close(emb(traj, BS, EGO_FUT_MODE), want, atol=0, rtol=0)


def test_time_embedder():
    te = TimeEmbedder(EMBED)
    out = te(torch.tensor([0.0, 8.0, 20.0]))
    assert out.shape == (3, EMBED)
    assert torch.isfinite(out).all()
    # Different timesteps must produce different conditioning.
    assert not torch.allclose(out[0], out[2])


def test_gradients_reach_encoders_but_not_anchors(anchors):
    bank = PlanAnchorQueryBank(anchors, EMBED)
    _, query = bank(BS)
    query.pow(2).mean().backward()
    assert bank.plan_anchor.grad is None, "anchors are frozen"
    assert bank.plan_anchor_encoder[0].weight.grad is not None
    assert torch.isfinite(bank.plan_anchor_encoder[0].weight.grad).all()


def test_works_with_the_regenerated_anchors():
    """End-to-end with the real regenerated file, if present."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "kmeans" / "kmeans_plan_6.npy"
    if not path.exists():
        pytest.skip("anchors not generated")

    bank = PlanAnchorQueryBank(torch.from_numpy(np.load(path)), EMBED)
    plan_anchor, query = bank(BS)
    cmd = torch.tensor([CMD_RIGHT, CMD_LEFT])

    picked = select_anchor_by_command(plan_anchor, cmd)
    assert picked[0, :, -1, 0].mean() > 0, "right-turn anchors should end at +x"
    assert picked[1, :, -1, 0].mean() < 0, "left-turn anchors should end at -x"
    assert select_by_command(query, cmd, EGO_FUT_MODE).shape == (BS, EGO_FUT_MODE, EMBED)
