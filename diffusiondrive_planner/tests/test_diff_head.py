"""Validate the diffusion refinement head and modulation layer against upstream.

Run: PYTHONPATH=.:diffusiondrive_planner/upstream-nusc PYTORCH_ENABLE_MPS_FALLBACK=1 \
       conda run -n uniad2.0 python -m pytest diffusiondrive_planner/tests/test_diff_head.py -q
"""

import pytest
import torch

from diffusiondrive_planner.diff_head import (
    NUM_COMMANDS,
    DiffRefine,
    ModulationLayer,
    bias_init_with_prob,
    undo_command_replication,
)

BS, K, T, EMBED = 2, 6, 6, 256


def upstream():
    return pytest.importorskip(
        "projects.mmdet3d_plugin.models.motion.diff_motion_blocks",
        reason="run with PYTHONPATH including diffusiondrive_planner/upstream-nusc",
    )


# ------------------------------------------------------------------ DiffRefine

def build_refine_pair():
    ref_cls = upstream().V4DiffMotionPlanningRefinementModule
    torch.manual_seed(0)
    ref = ref_cls(embed_dims=EMBED, ego_fut_ts=T, ego_fut_mode=K, if_zeroinit_reg=False)
    ours = DiffRefine(embed_dims=EMBED, ego_fut_ts=T, ego_fut_mode=K, if_zeroinit_reg=False)
    ours.load_state_dict(ref.state_dict(), strict=True)
    return ours.eval(), ref.eval()


def test_refine_state_dict_compatible():
    ref_cls = upstream().V4DiffMotionPlanningRefinementModule
    ref = ref_cls(embed_dims=EMBED, ego_fut_ts=T, ego_fut_mode=K)
    ours = DiffRefine(embed_dims=EMBED, ego_fut_ts=T, ego_fut_mode=K)
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=False)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"


def test_refine_matches_upstream():
    ours, ref = build_refine_pair()
    torch.manual_seed(1)
    feat = torch.randn(BS, K, EMBED)
    with torch.no_grad():
        got_reg, got_cls = ours(feat)
        want_reg, want_cls = ref(feat)
    torch.testing.assert_close(got_reg, want_reg, atol=0, rtol=0)
    torch.testing.assert_close(got_cls, want_cls, atol=0, rtol=0)


def test_refine_output_shapes():
    ours, _ = build_refine_pair()
    with torch.no_grad():
        reg, cls = ours(torch.randn(BS, K, EMBED))
    assert reg.shape == (BS, 1, NUM_COMMANDS * K, T, 2)
    assert cls.shape == (BS, 1, NUM_COMMANDS * K)


def test_output_is_a_plain_3x_replication():
    """Only K distinct trajectories exist; the 3K output repeats them.

    Treating 3K as genuine modes (e.g. taking an argmax over all 3K) would silently
    triple-count and bias mode selection.
    """
    ours, _ = build_refine_pair()
    torch.manual_seed(2)
    with torch.no_grad():
        reg, cls = ours(torch.randn(BS, K, EMBED))

    for c in range(1, NUM_COMMANDS):
        torch.testing.assert_close(reg[:, 0, :K], reg[:, 0, c * K : (c + 1) * K], atol=0, rtol=0)
        torch.testing.assert_close(cls[:, 0, :K], cls[:, 0, c * K : (c + 1) * K], atol=0, rtol=0)


def test_undo_replication_round_trips():
    """The loop feeds the prediction back via [..., -K:]; that slice must recover K modes."""
    ours, _ = build_refine_pair()
    torch.manual_seed(3)
    with torch.no_grad():
        reg, _ = ours(torch.randn(BS, K, EMBED))

    back = undo_command_replication(reg, K)
    assert back.shape == (BS * K, T, 2)
    # Because the block is a plain replication, the last K must equal the first K.
    torch.testing.assert_close(back, reg[:, :, :K].flatten(0, 2), atol=0, rtol=0)


def test_undo_replication_matches_upstream_expression():
    """Byte-for-byte the head's `diff_plan_reg[:, :, -ego_fut_mode:, ].flatten(0, 2)`."""
    torch.manual_seed(4)
    reg = torch.randn(BS, 1, NUM_COMMANDS * K, T, 2)
    want = reg[:, :, -K:, ].flatten(0, 2)
    torch.testing.assert_close(undo_command_replication(reg, K), want, atol=0, rtol=0)


def test_cls_and_reg_branches_have_different_structure():
    """cls uses linear_relu_ln (has LayerNorm); reg is a plain MLP. Not interchangeable."""
    ours, _ = build_refine_pair()
    assert any(isinstance(m, torch.nn.LayerNorm) for m in ours.plan_cls_branch)
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in ours.plan_reg_branch)
    assert ours.plan_reg_branch[-1].out_features == T * 2
    assert ours.plan_cls_branch[-1].out_features == 1


def test_zeroinit_reg_flag():
    """Stage-2 sets if_zeroinit_reg=False, so init_weight must NOT zero the reg branch."""
    off = DiffRefine(EMBED, T, K, if_zeroinit_reg=False)
    torch.nn.init.normal_(off.plan_reg_branch[-1].weight, std=0.1)
    off.init_weight()
    assert off.plan_reg_branch[-1].weight.abs().sum() > 0, "should not have been zeroed"

    on = DiffRefine(EMBED, T, K, if_zeroinit_reg=True)
    on.init_weight()
    assert on.plan_reg_branch[-1].weight.abs().sum() == 0

    # Focal-loss bias is applied regardless of the flag.
    assert torch.allclose(
        off.plan_cls_branch[-1].bias,
        torch.full_like(off.plan_cls_branch[-1].bias, bias_init_with_prob(0.01)),
    )


def test_bias_init_with_prob_matches_mmdet():
    mmdet_util = pytest.importorskip("mmdet.models.utils")
    fn = getattr(mmdet_util, "bias_init_with_prob", None)
    if fn is None:
        pytest.skip("mmdet helper not exposed here")
    for p in (0.01, 0.1, 0.5):
        assert bias_init_with_prob(p) == pytest.approx(fn(p))


# -------------------------------------------------------------- Modulation

def build_modulation_pair(global_cond=False):
    ref_cls = upstream().V1ModulationLayer
    torch.manual_seed(0)
    ref = ref_cls(embed_dims=EMBED, if_global_cond=global_cond, if_zeroinit_scale=False)
    ours = ModulationLayer(embed_dims=EMBED, if_global_cond=global_cond, if_zeroinit_scale=False)
    ours.load_state_dict(ref.state_dict(), strict=True)
    return ours.eval(), ref.eval()


def test_modulation_state_dict_compatible():
    ours, _ = build_modulation_pair()
    assert ours.scale_shift_mlp[-1].in_features == EMBED
    assert ours.scale_shift_mlp[-1].out_features == EMBED * 2


def test_modulation_matches_upstream():
    ours, ref = build_modulation_pair()
    torch.manual_seed(5)
    feat = torch.randn(BS, K, EMBED)
    time_embed = torch.randn(BS, K, EMBED)
    with torch.no_grad():
        torch.testing.assert_close(
            ours(feat, time_embed), ref(feat, time_embed), atol=0, rtol=0
        )


def test_modulation_with_global_cond_matches_upstream():
    ours, ref = build_modulation_pair(global_cond=True)
    torch.manual_seed(6)
    feat = torch.randn(BS, K, EMBED)
    time_embed = torch.randn(BS, K, EMBED)
    gc = torch.randn(BS, K, EMBED)
    with torch.no_grad():
        torch.testing.assert_close(
            ours(feat, time_embed, global_cond=gc),
            ref(feat, time_embed, global_cond=gc),
            atol=0, rtol=0,
        )


def test_modulation_is_film():
    """feature * (1 + scale) + shift — with scale=shift=0 it must be the identity."""
    ours, _ = build_modulation_pair()
    torch.nn.init.constant_(ours.scale_shift_mlp[-1].weight, 0)
    torch.nn.init.constant_(ours.scale_shift_mlp[-1].bias, 0)
    feat = torch.randn(BS, K, EMBED)
    with torch.no_grad():
        out = ours(feat, torch.randn(BS, K, EMBED))
    torch.testing.assert_close(out, feat, atol=1e-6, rtol=1e-6)


def test_modulation_actually_uses_the_timestep():
    ours, _ = build_modulation_pair()
    feat = torch.randn(BS, K, EMBED)
    with torch.no_grad():
        a = ours(feat, torch.zeros(BS, K, EMBED))
        b = ours(feat, torch.full((BS, K, EMBED), 5.0))
    assert not torch.allclose(a, b), "output ignores the time embedding"


def test_gradients_flow():
    ours, _ = build_refine_pair()
    ours.train()
    feat = torch.randn(BS, K, EMBED, requires_grad=True)
    reg, cls = ours(feat)
    (reg.pow(2).mean() + cls.pow(2).mean()).backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    assert ours.plan_reg_branch[-1].weight.grad is not None
    assert ours.plan_cls_branch[-1].weight.grad is not None
