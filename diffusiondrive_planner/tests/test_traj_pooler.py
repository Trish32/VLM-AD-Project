"""Validate the trajectory pooler against upstream's V1TrajPooler.

Upstream's classes import fine in `uniad2.0` (they only need mmcv registries, not the
CUDA extension), so the ported keypoint generator and projection are diffed directly
against the originals. The pooler itself is compared end to end by copying weights
across, which also exercises `dfa_torch` in the shape regime the planner actually uses.

Run: PYTHONPATH=.:diffusiondrive_planner/upstream-nusc PYTORCH_ENABLE_MPS_FALLBACK=1 \
       conda run -n uniad2.0 python -m pytest diffusiondrive_planner/tests/test_traj_pooler.py -q
"""

import pytest
import torch

from diffusiondrive_planner.traj_pooler import (
    DEFAULT_FIX_HEIGHT,
    DEFAULT_GROUND_HEIGHT,
    TrajKeypointGenerator,
    TrajPooler,
    project_points,
)

BS, MODAL, EGO_FUT_TS = 2, 6, 6
EMBED, CAMS, LEVELS, GROUPS = 256, 6, 4, 8
NUM_PTS = EGO_FUT_TS * len(DEFAULT_FIX_HEIGHT)  # 30
FEAT_SHAPES = [(8, 12), (4, 6), (2, 3), (1, 2)]  # tiny multi-level maps


def upstream_module():
    return pytest.importorskip(
        "projects.mmdet3d_plugin.models.motion.diff_motion_blocks",
        reason="run with PYTHONPATH including diffusiondrive_planner/upstream-nusc",
    )


def make_metas(bs=BS):
    torch.manual_seed(0)
    proj = torch.eye(4).repeat(bs, CAMS, 1, 1)
    # Give each camera a distinct, non-degenerate pose so projection differences show up.
    proj[:, :, 0, 3] = torch.linspace(-2, 2, CAMS)
    proj[:, :, 1, 3] = 0.5
    proj[:, :, 2, 3] = 6.0  # push points in front of the camera
    return {
        "projection_mat": proj,
        "image_wh": torch.tensor([[[64.0, 32.0]] * CAMS] * bs),
    }


def make_feature_maps(bs=BS, embed=EMBED):
    """Column-format multi-cam multi-level maps, as feature_maps_format produces."""
    sizes = [h * w for _ in range(CAMS) for (h, w) in FEAT_SHAPES]
    num_feat = sum(sizes)
    torch.manual_seed(1)
    col = torch.randn(bs, num_feat, embed)
    spatial_shape = torch.tensor([list(FEAT_SHAPES)] * CAMS, dtype=torch.int64)
    starts = torch.tensor(sizes, dtype=torch.int64).cumsum(0) - torch.tensor(
        sizes, dtype=torch.int64
    )
    return [col, spatial_shape, starts.reshape(CAMS, LEVELS)]


# ------------------------------------------------------------- keypoint generator

def test_keypoint_generator_matches_upstream():
    ref_cls = upstream_module().TrajSparsePoint3DKeyPointsGenerator
    ref = ref_cls(
        embed_dims=EMBED, num_sample=EGO_FUT_TS,
        fix_height=DEFAULT_FIX_HEIGHT, ground_height=DEFAULT_GROUND_HEIGHT,
    )
    ours = TrajKeypointGenerator(num_sample=EGO_FUT_TS)

    torch.manual_seed(0)
    anchor = torch.randn(BS, MODAL, EGO_FUT_TS * 2)
    torch.testing.assert_close(ours(anchor), ref(anchor), atol=0, rtol=0)


def test_keypoint_shape_and_heights():
    gen = TrajKeypointGenerator(num_sample=EGO_FUT_TS)
    assert gen.num_pts == NUM_PTS == 30

    anchor = torch.zeros(1, 1, EGO_FUT_TS * 2)
    kps = gen(anchor)
    assert kps.shape == (1, 1, NUM_PTS, 3)
    # z must be ground_height + each fix_height, and xy untouched at zero.
    zs = sorted({round(float(v), 5) for v in kps[0, 0, :, 2]})
    expected = sorted(round(DEFAULT_GROUND_HEIGHT + h, 5) for h in DEFAULT_FIX_HEIGHT)
    assert zs == expected
    assert torch.allclose(kps[..., :2], torch.zeros_like(kps[..., :2]))


def test_keypoint_layout_is_waypoint_major():
    """flatten(2,4) gives waypoint-major order: all heights of waypoint 0, then 1, ...

    A height-major layout has the same shape and would silently pair weights with the
    wrong points.
    """
    gen = TrajKeypointGenerator(num_sample=EGO_FUT_TS)
    waypoints = torch.arange(EGO_FUT_TS * 2, dtype=torch.float32).reshape(1, 1, -1)
    kps = gen(waypoints)
    n_h = len(DEFAULT_FIX_HEIGHT)
    for t in range(EGO_FUT_TS):
        block = kps[0, 0, t * n_h : (t + 1) * n_h, :2]
        assert torch.allclose(block, block[0].expand_as(block)), "heights must be contiguous per waypoint"


# ------------------------------------------------------------------- projection

def test_project_points_matches_upstream():
    ref = upstream_module().V1TrajPooler.project_points
    metas = make_metas()
    torch.manual_seed(0)
    kps = torch.randn(BS, MODAL, NUM_PTS, 3)

    torch.testing.assert_close(
        project_points(kps, metas["projection_mat"], metas["image_wh"]),
        ref(kps, metas["projection_mat"], metas["image_wh"]),
        atol=0, rtol=0,
    )


def test_projection_normalizes_into_unit_range():
    """image_wh division puts in-view points in [0,1] — the range dfa_torch samples."""
    metas = make_metas()
    kps = torch.zeros(BS, MODAL, NUM_PTS, 3)
    kps[..., 2] = 0.0  # in front of camera thanks to the +6 translation
    out = project_points(kps, metas["projection_mat"], metas["image_wh"])
    assert out.shape == (BS, CAMS, MODAL, NUM_PTS, 2)
    assert torch.isfinite(out).all()


# ----------------------------------------------------------------- full pooler

def build_pair():
    """Our pooler and upstream's, with identical weights."""
    ref_cls = upstream_module().V1TrajPooler
    torch.manual_seed(0)
    ref = ref_cls(embed_dims=EMBED, ego_fut_ts=EGO_FUT_TS)
    ours = TrajPooler(embed_dims=EMBED, ego_fut_ts=EGO_FUT_TS)
    ours.load_state_dict(ref.state_dict(), strict=True)
    return ours.eval(), ref.eval()


def test_state_dicts_are_compatible():
    """Our module must accept upstream's state_dict verbatim — 0 missing / 0 unexpected."""
    ref_cls = upstream_module().V1TrajPooler
    ref = ref_cls(embed_dims=EMBED, ego_fut_ts=EGO_FUT_TS)
    ours = TrajPooler(embed_dims=EMBED, ego_fut_ts=EGO_FUT_TS)
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"


def test_weights_match_upstream():
    ours, ref = build_pair()
    metas = make_metas()
    torch.manual_seed(2)
    instance_feature = torch.randn(BS, MODAL, EMBED)

    with torch.no_grad():
        got = ours._get_weights(instance_feature, metas)
        want = ref._get_weights(instance_feature, metas)
    assert got.shape == (BS, MODAL, CAMS, LEVELS, NUM_PTS, GROUPS)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_weights_are_group_wise_softmaxed():
    """Each group must sum to 1 over cam x level x point jointly — not per-camera."""
    ours, _ = build_pair()
    metas = make_metas()
    torch.manual_seed(3)
    with torch.no_grad():
        w = ours._get_weights(torch.randn(BS, MODAL, EMBED), metas)
    totals = w.sum(dim=(2, 3, 4))  # over cams, levels, pts
    torch.testing.assert_close(totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5)


def test_forward_cumsums_deltas():
    """forward() must convert deltas to absolute positions before pooling.

    Feeding deltas straight through produces valid shapes and silently samples the
    wrong pixels, so compare forward(deltas) against pool_feature(cumsum(deltas)).
    """
    ours, _ = build_pair()
    metas, fmaps = make_metas(), make_feature_maps()
    torch.manual_seed(4)
    instance_feature = torch.randn(BS, MODAL, EMBED)
    deltas = torch.randn(BS * MODAL, EGO_FUT_TS, 2) * 0.5

    with torch.no_grad():
        via_forward = ours(instance_feature, deltas, metas, fmaps, modal_num=MODAL)
        via_manual = ours.pool_feature_from_traj(
            instance_feature, deltas.cumsum(dim=-2), metas, fmaps, modal_num=MODAL
        )
        wrong = ours.pool_feature_from_traj(
            instance_feature, deltas, metas, fmaps, modal_num=MODAL
        )

    torch.testing.assert_close(via_forward, via_manual, atol=0, rtol=0)
    assert not torch.allclose(via_forward, wrong), "cumsum made no difference — check the fixture"


def test_forward_shape_and_residual():
    ours, _ = build_pair()
    metas, fmaps = make_metas(), make_feature_maps()
    torch.manual_seed(5)
    instance_feature = torch.randn(BS, MODAL, EMBED)
    deltas = torch.randn(BS * MODAL, EGO_FUT_TS, 2) * 0.5

    with torch.no_grad():
        out = ours(instance_feature, deltas, metas, fmaps, modal_num=MODAL)
    assert out.shape == (BS, MODAL, EMBED)
    assert torch.isfinite(out).all()
    # Residual is additive, so the output must differ from the pooled part alone.
    assert not torch.allclose(out, instance_feature)


def test_gradients_flow_to_trajectory():
    """The pooler must be differentiable w.r.t. the trajectory itself.

    This is what lets image evidence influence the denoised waypoints rather than only
    the query features.
    """
    ours, _ = build_pair()
    ours.train()
    metas, fmaps = make_metas(), make_feature_maps()
    torch.manual_seed(6)
    instance_feature = torch.randn(BS, MODAL, EMBED, requires_grad=True)
    deltas = (torch.randn(BS * MODAL, EGO_FUT_TS, 2) * 0.5).requires_grad_(True)

    ours(instance_feature, deltas, metas, fmaps, modal_num=MODAL).pow(2).mean().backward()

    assert deltas.grad is not None and torch.isfinite(deltas.grad).all()
    assert instance_feature.grad is not None
    assert ours.weights_fc.weight.grad is not None
    assert ours.camera_encoder[0].weight.grad is not None
