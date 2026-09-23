"""Wiring tests for the truncated-diffusion planner and AsymmetricFFN.

The individual components are already diffed against upstream elsewhere. What is tested
here is that they are *connected correctly* — the failure mode the component tests
cannot see. Run with synthetic detector output so it works on a Mac with no checkpoint.

Run: PYTHONPATH=.:diffusiondrive_planner/upstream-nusc PYTORCH_ENABLE_MPS_FALLBACK=1 \
       conda run -n uniad2.0 python -m pytest diffusiondrive_planner/tests/test_diff_planner.py -q
"""

import pytest
import torch

from diffusiondrive_planner.diff_planner import DIFF_OPERATION_ORDER, DiffPlanner
from diffusiondrive_planner.ffn import AsymmetricFFN

BS, K, T, EMBED, N_AGENTS = 2, 6, 6, 64, 10
CAMS, LEVELS, HEADS = 6, 4, 8
FEAT_SHAPES = [(8, 12), (4, 6), (2, 3), (1, 2)]


# ---------------------------------------------------------------- AsymmetricFFN

def test_ffn_matches_upstream():
    mod = pytest.importorskip(
        "projects.mmdet3d_plugin.models.blocks",
        reason="run with PYTHONPATH including diffusiondrive_planner/upstream-nusc",
    )
    torch.manual_seed(0)
    ref = mod.AsymmetricFFN(
        in_channels=EMBED, pre_norm=dict(type="LN"), embed_dims=EMBED,
        feedforward_channels=EMBED * 2, num_fcs=2, ffn_drop=0.0,
        act_cfg=dict(type="ReLU", inplace=True),
    )
    ours = AsymmetricFFN(
        in_channels=EMBED, pre_norm=True, embed_dims=EMBED,
        feedforward_channels=EMBED * 2, num_fcs=2,
    )
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=False)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"

    x = torch.randn(BS, K, EMBED)
    ours.eval(); ref.eval()
    with torch.no_grad():
        torch.testing.assert_close(ours(x), ref(x), atol=1e-6, rtol=1e-6)


def test_ffn_identity_fc_is_a_linear_not_identity():
    """The local-rebind quirk: identity_fc carries weights. See ffn.py docstring."""
    ffn = AsymmetricFFN(in_channels=EMBED, embed_dims=EMBED, feedforward_channels=EMBED * 2)
    assert isinstance(ffn.identity_fc, torch.nn.Linear), (
        "identity_fc must be a Linear — upstream's ternary compares the REBOUND "
        "in_channels (=feedforward_channels) against embed_dims"
    )
    assert "identity_fc.weight" in ffn.state_dict()


def test_ffn_residual_uses_pre_normed_input():
    """identity = x happens AFTER pre_norm, so the skip carries the normalised tensor."""
    torch.manual_seed(1)
    ffn = AsymmetricFFN(in_channels=EMBED, embed_dims=EMBED, feedforward_channels=EMBED * 2).eval()
    x = torch.randn(BS, K, EMBED) * 5 + 3  # far from zero-mean so norm matters

    with torch.no_grad():
        out = ffn(x)
        normed = ffn.pre_norm(x)
        expect = ffn.identity_fc(normed) + ffn.layers(normed)
        wrong = ffn.identity_fc(x) + ffn.layers(normed)  # the natural misreading
    torch.testing.assert_close(out, expect, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out, wrong, atol=1e-4), "test cannot distinguish the two paths"


# ------------------------------------------------------------------- fixtures

def make_planner():
    torch.manual_seed(0)
    return DiffPlanner(
        embed_dims=EMBED, ego_fut_ts=T, ego_fut_mode=K,
        num_heads=HEADS, ffn_channels=EMBED * 2, num_repeats=2,
    ).eval()


def make_inputs():
    torch.manual_seed(1)
    proj = torch.eye(4).repeat(BS, CAMS, 1, 1)
    proj[:, :, 2, 3] = 6.0
    metas = {
        "projection_mat": proj,
        "image_wh": torch.tensor([[[64.0, 32.0]] * CAMS] * BS),
    }
    sizes = [h * w for _ in range(CAMS) for (h, w) in FEAT_SHAPES]
    col = torch.randn(BS, sum(sizes), EMBED)
    spatial_shape = torch.tensor([list(FEAT_SHAPES)] * CAMS, dtype=torch.int64)
    starts = (torch.tensor(sizes, dtype=torch.int64).cumsum(0)
              - torch.tensor(sizes, dtype=torch.int64))
    feature_maps = [col, spatial_shape, starts.reshape(CAMS, LEVELS)]

    return dict(
        plan_anchor=torch.randn(BS, K, T, 2).cumsum(dim=-2) * 0.3,
        anchor_query=torch.randn(BS, K, EMBED),
        agent_feature=torch.randn(BS, N_AGENTS, EMBED),
        agent_pos=torch.randn(BS, N_AGENTS, EMBED),
        ego_pos=torch.randn(BS, K, EMBED),
        metas=metas,
        feature_maps=feature_maps,
    )


# --------------------------------------------------------------------- wiring

def test_operation_order_is_repeated_twice():
    p = make_planner()
    assert len(p.operation_order) == len(DIFF_OPERATION_ORDER) * 2
    assert len(p.diff_layers) == len(p.operation_order), "one layer slot per op, by index"
    assert p.operation_order[: len(DIFF_OPERATION_ORDER)] == DIFF_OPERATION_ORDER


def test_forward_runs_and_shapes():
    p, kw = make_planner(), make_inputs()
    reg, cls = p(**kw)
    assert reg.shape == (BS, 1, 3 * K, T, 2)
    assert cls.shape == (BS, 1, 3 * K)
    assert torch.isfinite(reg).all() and torch.isfinite(cls).all()


def test_two_denoising_steps_by_default():
    """Default schedule is [20, 0]; changing step_num must change the trajectory."""
    p, kw = make_planner(), make_inputs()
    torch.manual_seed(7)
    two = p(step_num=2, **kw)[0]
    torch.manual_seed(7)
    four = p(step_num=4, **kw)[0]
    assert not torch.allclose(two, four, atol=1e-6), "step_num had no effect"


def test_prediction_depends_on_image_features():
    """traj_pooler must actually couple image evidence into the output.

    If the pooler were mis-wired (or its residual swallowed the pooled term), the
    planner would still produce plausible trajectories from the anchors alone.
    """
    p, kw = make_planner(), make_inputs()
    torch.manual_seed(7)
    a = p(**kw)[0]

    kw2 = dict(kw)
    kw2["feature_maps"] = [kw["feature_maps"][0] * -3.0, *kw["feature_maps"][1:]]
    torch.manual_seed(7)
    b = p(**kw2)[0]
    assert not torch.allclose(a, b, atol=1e-6), "output ignores the feature maps"


def test_prediction_depends_on_agents():
    """agent_cross_gnn must couple detections into the plan."""
    p, kw = make_planner(), make_inputs()
    torch.manual_seed(7)
    a = p(**kw)[0]

    kw2 = dict(kw, agent_feature=kw["agent_feature"] * -3.0)
    torch.manual_seed(7)
    b = p(**kw2)[0]
    assert not torch.allclose(a, b, atol=1e-6), "output ignores agent features"


def test_prediction_depends_on_anchor_query():
    p, kw = make_planner(), make_inputs()
    torch.manual_seed(7)
    a = p(**kw)[0]
    kw2 = dict(kw, anchor_query=kw["anchor_query"] * -3.0)
    torch.manual_seed(7)
    b = p(**kw2)[0]
    assert not torch.allclose(a, b, atol=1e-6), "output ignores the anchor queries"


def test_prediction_depends_on_the_anchor_itself():
    """The seed is the anchor; a different anchor must give a different plan."""
    p, kw = make_planner(), make_inputs()
    torch.manual_seed(7)
    a = p(**kw)[0]
    kw2 = dict(kw, plan_anchor=kw["plan_anchor"] + 2.0)
    torch.manual_seed(7)
    b = p(**kw2)[0]
    assert not torch.allclose(a, b, atol=1e-6), "output ignores the plan anchor"


def test_decoupled_attention_widths():
    """agent_cross_gnn runs at 2*embed_dims; self/anchor attention do not."""
    p = make_planner()
    assert p.fc_before.out_features == EMBED * 2
    assert p.fc_after.in_features == EMBED * 2
    for op, layer in zip(p.operation_order, p.diff_layers):
        if op == "agent_cross_gnn":
            assert layer.embed_dims == EMBED * 2
        elif op in ("self_attn", "anchor_cross_gnn"):
            assert layer.embed_dims == EMBED


def test_no_decouple_variant_still_runs():
    torch.manual_seed(0)
    p = DiffPlanner(
        embed_dims=EMBED, ego_fut_ts=T, ego_fut_mode=K, num_heads=HEADS,
        ffn_channels=EMBED * 2, decouple_attn=False,
    ).eval()
    reg, _ = p(**make_inputs())
    assert reg.shape == (BS, 1, 3 * K, T, 2) and torch.isfinite(reg).all()


def test_runs_on_mps():
    """End-to-end on MPS — the local dev path.

    Gated on torch >= 2.0: the mmdet-2.x env pins torch 1.12, whose 2022-era MPS backend
    cannot run this graph and ABORTS the process (SIGABRT) instead of raising, taking the
    whole pytest session with it. That env exists only to diff against upstream on CPU;
    the MPS path is validated in the torch 2.x env, which is what local dev actually uses.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    if torch.__version__ < "2":
        pytest.skip(f"torch {torch.__version__} MPS backend aborts on this graph")
    p = make_planner().to("mps")
    kw = make_inputs()
    kw["plan_anchor"] = kw["plan_anchor"].to("mps")
    kw["anchor_query"] = kw["anchor_query"].to("mps")
    kw["agent_feature"] = kw["agent_feature"].to("mps")
    kw["agent_pos"] = kw["agent_pos"].to("mps")
    kw["ego_pos"] = kw["ego_pos"].to("mps")
    kw["metas"] = {k: v.to("mps") for k, v in kw["metas"].items()}
    kw["feature_maps"] = [t.to("mps") for t in kw["feature_maps"]]
    reg, _ = p(**kw)
    assert reg.shape == (BS, 1, 3 * K, T, 2) and torch.isfinite(reg).all()
