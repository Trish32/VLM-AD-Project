"""Load the official DiffusionDrive nuScenes checkpoint into the ported planner.

This is the fidelity bar for every port in this repo: the official weights must load with 0 missing
and 0 unexpected keys. It is also the strongest available check on every structural
decision made while porting — layer order, parameter naming, the AsymmetricFFN
`identity_fc` quirk, the decoupled-attention widths. A wrong call in any of those shows
up here as a missing or mis-shaped key.

Skips cleanly when the checkpoint is absent:
    curl -L -o diffusiondrive_planner/checkpoints/diffusiondrive_nusc_stage2.pth \\
      https://huggingface.co/hustvl/DiffusionDrive/resolve/main/diffusiondrive_nusc_stage2.pth

Run: PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \\
       conda run -n uniad2.0 python -m pytest diffusiondrive_planner/tests/test_checkpoint_load.py -q
"""

from pathlib import Path

import pytest
import torch

from diffusiondrive_planner.diff_planner import DIFF_OPERATION_ORDER, DiffPlanner
from diffusiondrive_planner.plan_query import (
    PlanAnchorQueryBank,
    select_anchor_by_command,
)

CKPT = (Path(__file__).resolve().parents[1] / "checkpoints"
        / "diffusiondrive_nusc_stage2.pth")
PREFIX = "head.motion_plan_head."
EMBED, T, K, HEADS, FFN = 256, 6, 6, 8, 512
CMD_RIGHT, CMD_LEFT, CMD_STRAIGHT = 0, 1, 2


@pytest.fixture(scope="module")
def head_sd():
    if not CKPT.exists():
        pytest.skip(f"{CKPT.name} not downloaded; see module docstring")
    try:
        blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13
        blob = torch.load(CKPT, map_location="cpu")
    sd = blob["state_dict"]
    return {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}


def remap(our_keys, sub):
    """Our parameter names -> checkpoint names (only two wrappers differ)."""
    out = {}
    for k in our_keys:
        ck = k.replace("time_mlp.time_mlp.", "time_mlp.").replace("traj_embedder.", "")
        out[k] = sub[ck]
    return out


@pytest.fixture(scope="module")
def planner(head_sd):
    p = DiffPlanner(EMBED, T, K, num_heads=HEADS, ffn_channels=FFN, num_repeats=2)
    p.load_state_dict(remap(p.state_dict().keys(), head_sd), strict=True)
    return p.eval()


# ------------------------------------------------------------------- the bar

def test_loads_0_missing_0_unexpected(head_sd):
    p = DiffPlanner(EMBED, T, K, num_heads=HEADS, ffn_channels=FFN, num_repeats=2)
    built = remap(p.state_dict().keys(), head_sd)
    incompatible = p.load_state_dict(built, strict=False)
    assert not incompatible.missing_keys, f"missing: {incompatible.missing_keys[:8]}"
    assert not incompatible.unexpected_keys, f"unexpected: {incompatible.unexpected_keys[:8]}"


def test_no_shape_mismatches(head_sd):
    p = DiffPlanner(EMBED, T, K, num_heads=HEADS, ffn_channels=FFN, num_repeats=2)
    ours = p.state_dict()
    built = remap(ours.keys(), head_sd)
    bad = [(k, tuple(ours[k].shape), tuple(built[k].shape))
           for k in ours if ours[k].shape != built[k].shape]
    assert not bad, f"shape mismatches: {bad[:5]}"


def test_checkpoint_has_22_diff_layer_slots(head_sd):
    """11 operations x 2 repeats. Confirms DIFF_OPERATION_ORDER is the right length."""
    slots = {int(k.split(".")[1]) for k in head_sd if k.startswith("diff_layers.")}
    assert slots == set(range(len(DIFF_OPERATION_ORDER) * 2)) == set(range(22))


def test_ffn_identity_fc_exists_in_checkpoint(head_sd):
    """The quirk is real: identity_fc carries trained weights, it is not an Identity.

    Slots 7 and 18 are the two `ffn` positions.
    """
    for slot in (7, 18):
        assert f"diff_layers.{slot}.identity_fc.weight" in head_sd
        assert head_sd[f"diff_layers.{slot}.identity_fc.weight"].shape == (EMBED, EMBED)


def test_decoupled_attention_width_in_checkpoint(head_sd):
    """agent_cross_gnn runs at 2*embed_dims, self/anchor attention at embed_dims."""
    assert head_sd["fc_before.weight"].shape == (EMBED * 2, EMBED)
    assert head_sd["fc_after.weight"].shape == (EMBED, EMBED * 2)
    # slot 1 = self_attn, slot 3 = agent_cross_gnn, slot 5 = anchor_cross_gnn
    assert head_sd["diff_layers.1.attn.in_proj_weight"].shape == (3 * EMBED, EMBED)
    assert head_sd["diff_layers.3.attn.in_proj_weight"].shape == (6 * EMBED, 2 * EMBED)
    assert head_sd["diff_layers.5.attn.in_proj_weight"].shape == (3 * EMBED, EMBED)


# ------------------------------------------------------------------- anchors

def test_plan_anchor_is_in_the_checkpoint(head_sd):
    """So the regenerated mini anchors are overwritten and cannot affect reproduction."""
    assert "plan_anchor" in head_sd
    assert head_sd["plan_anchor"].shape == (3, K, T, 2)


def test_official_anchors_confirm_command_order(head_sd):
    """[right, left, straight], with x = lateral and right positive.

    Both of those had to be INFERRED from nuscenes_converter.py while regenerating
    anchors. The trained weights are the independent confirmation.
    """
    x_end = head_sd["plan_anchor"][:, :, -1, 0]
    assert x_end[CMD_RIGHT].mean() > 1.0
    assert x_end[CMD_LEFT].mean() < -1.0
    assert abs(float(x_end[CMD_STRAIGHT].mean())) < 1.0

    y_end = head_sd["plan_anchor"][:, :, -1, 1]
    assert (y_end > 0).all(), "y is forward; every anchor should advance"
    assert y_end.abs().mean() > x_end.abs().mean(), "forward motion should dominate"


def test_our_bank_accepts_the_official_anchors(head_sd):
    bank = PlanAnchorQueryBank(head_sd["plan_anchor"], EMBED)
    bank.plan_anchor_encoder.load_state_dict(
        {k[len("plan_anchor_encoder."):]: v
         for k, v in head_sd.items() if k.startswith("plan_anchor_encoder.")}
    )
    plan_anchor, query = bank(2)
    assert plan_anchor.shape == (2, 3, K, T, 2)
    assert query.shape == (2, 1, 3 * K, EMBED)
    assert torch.isfinite(query).all()


# ------------------------------------------------------------------- forward

def synthetic_scene(bs=1, embed=EMBED):
    """Real feature-map geometry (704x256 input, ResNet-50 + FPN strides), random values.

    Detector outputs are random, so this exercises the pipeline rather than the metric.
    """
    torch.manual_seed(0)
    cams, levels = 6, 4
    shapes = [(32, 88), (16, 44), (8, 22), (4, 11)]
    sizes = [h * w for _ in range(cams) for (h, w) in shapes]
    starts = torch.tensor(sizes).cumsum(0) - torch.tensor(sizes)
    feature_maps = [
        torch.randn(bs, sum(sizes), embed) * 0.1,
        torch.tensor([list(shapes)] * cams, dtype=torch.int64),
        starts.reshape(cams, levels),
    ]
    proj = torch.eye(4).repeat(bs, cams, 1, 1)
    proj[:, :, 2, 3] = 8.0
    metas = {
        "projection_mat": proj,
        "image_wh": torch.tensor([[[704.0, 256.0]] * cams] * bs),
    }
    return metas, feature_maps


@pytest.mark.parametrize("cmd", [CMD_RIGHT, CMD_LEFT, CMD_STRAIGHT])
def test_forward_with_trained_weights(planner, head_sd, cmd):
    metas, feature_maps = synthetic_scene()
    anchors = head_sd["plan_anchor"][None]
    torch.manual_seed(1)

    reg, cls = planner(
        plan_anchor=select_anchor_by_command(anchors, torch.tensor([cmd])),
        anchor_query=torch.randn(1, K, EMBED) * 0.1,
        agent_feature=torch.randn(1, 20, EMBED) * 0.1,
        agent_pos=torch.randn(1, 20, EMBED) * 0.1,
        ego_pos=torch.randn(1, K, EMBED) * 0.1,
        metas=metas,
        feature_maps=feature_maps,
    )

    assert reg.shape == (1, 1, 3 * K, T, 2) and cls.shape == (1, 1, 3 * K)
    assert torch.isfinite(reg).all() and torch.isfinite(cls).all()

    best = int(cls[0, 0, :K].argmax())
    endpoint = reg[0, 0, best].cumsum(0)[-1]
    # 3s of ego motion: forward, and within a sane speed envelope.
    assert 1.0 < float(endpoint[1]) < 60.0, f"implausible forward distance {endpoint[1]:.2f}m"
    assert abs(float(endpoint[0])) < 25.0, f"implausible lateral offset {endpoint[0]:.2f}m"
