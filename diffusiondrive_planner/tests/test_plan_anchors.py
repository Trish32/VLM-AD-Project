"""Checks on the regenerated plan anchors.

These guard the contract between `tools/gen_plan_anchors.py` and the v13 head: shape,
command ordering, and compatibility with the head's normalization. Semantic checks, not
quality checks — mini-derived anchors are not expected to be good, only well-formed.

Run: PYTHONPATH=. conda run -n uniad2.0 python -m pytest \
       diffusiondrive_planner/tests/test_plan_anchors.py -q
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from diffusiondrive_planner.truncated_diffusion import (
    anchor_to_deltas,
    denormalize_traj,
    normalize_traj,
)

ANCHORS = Path(__file__).resolve().parents[1] / "data" / "kmeans" / "kmeans_plan_6.npy"
EGO_FUT_MODE, EGO_FUT_TS = 6, 6
CMD_RIGHT, CMD_LEFT, CMD_STRAIGHT = 0, 1, 2


@pytest.fixture(scope="module")
def anchors():
    if not ANCHORS.exists():
        pytest.skip(f"{ANCHORS} not generated yet; run tools/gen_plan_anchors.py")
    return np.load(ANCHORS)


def test_shape_is_what_the_head_indexes(anchors):
    """The head does plan_anchor[None].tile(bs,1,1,1,1) then plan_anchor[bs, cmd].

    That requires (3, ego_fut_mode, ego_fut_ts, 2). Upstream's SHIPPED kmeans_plan.py
    emits (K, 6, 2) instead — the per-command path is commented out there — so this is
    exactly the mismatch that would surface as a confusing index error at model build.
    """
    assert anchors.shape == (3, EGO_FUT_MODE, EGO_FUT_TS, 2)
    assert anchors.dtype == np.float32

    a = torch.from_numpy(anchors)
    tiled = a[None].tile(2, 1, 1, 1, 1)
    assert tiled.shape == (2, 3, EGO_FUT_MODE, EGO_FUT_TS, 2)
    selected = tiled[torch.arange(2), torch.tensor([CMD_RIGHT, CMD_LEFT])]
    assert selected.shape == (2, EGO_FUT_MODE, EGO_FUT_TS, 2)


def test_command_ordering_is_right_left_straight(anchors):
    """Index order must match the converter: [right, left, straight].

    The converter emits one-hot [1,0,0]=Right, [0,1,0]=Left, [0,0,1]=Straight, and the
    head indexes with cmd.argmax(). Anchors stored in a different order would steer the
    ego the wrong way while every shape stayed valid.
    """
    x_end = anchors[:, :, -1, 0]  # x is lateral, right positive
    assert x_end[CMD_RIGHT].mean() > 1.0, "right-turn anchors must go +x"
    assert x_end[CMD_LEFT].mean() < -1.0, "left-turn anchors must go -x"
    assert abs(float(x_end[CMD_STRAIGHT].mean())) < 1.0, "straight anchors must stay near x=0"
    assert x_end[CMD_RIGHT].mean() > x_end[CMD_STRAIGHT].mean() > x_end[CMD_LEFT].mean()


def test_forward_axis_dominates(anchors):
    """y is longitudinal. If x and y were swapped every shape would still be valid."""
    y_end = anchors[:, :, -1, 1]
    x_end = anchors[:, :, -1, 0]
    assert np.abs(y_end).mean() > np.abs(x_end).mean() * 2
    assert (y_end > 0).mean() > 0.8, "anchors should mostly move forward"


def test_deltas_fit_the_heads_normalization(anchors):
    """Normalization applies to per-step DELTAS, not endpoints.

    Endpoints reach ~40m, which is fine; a single 0.5s delta above 7.6m is not, because
    it would be silently clipped when the diffusion seeds from the anchor.
    """
    deltas = anchor_to_deltas(torch.from_numpy(anchors))
    err = (denormalize_traj(normalize_traj(deltas)) - deltas).abs()
    # Allow a little clipping at the extremes (mini is a small, skewed sample), but the
    # bulk must be representable or the anchor set is in the wrong units/frame entirely.
    assert float(err.max()) < 0.5, f"anchors badly outside normalization range: {err.max():.3f}m"
    assert float((err > 1e-4).float().mean()) < 0.05, "more than 5% of deltas clipped"


def test_delta_cumsum_round_trip(anchors):
    """anchor_to_deltas must be exactly invertible by cumsum on real anchors."""
    a = torch.from_numpy(anchors)
    torch.testing.assert_close(anchor_to_deltas(a).cumsum(dim=-2), a, atol=1e-5, rtol=1e-5)


def test_anchors_are_distinct(anchors):
    """K-means centres should be distinct; duplicates mean k exceeded the sample count."""
    for cmd in range(3):
        endpoints = anchors[cmd, :, -1, :]
        dists = np.linalg.norm(endpoints[:, None] - endpoints[None, :], axis=-1)
        off_diag = dists[~np.eye(EGO_FUT_MODE, dtype=bool)]
        assert off_diag.min() > 1e-3, f"duplicate anchors for command {cmd}"


def test_trajectories_start_near_ego(anchors):
    """First waypoint is one 0.5s step from the ego, so it must be close to the origin."""
    first = anchors[:, :, 0, :]
    assert np.abs(first).max() < 10.0, "first waypoint implausibly far from ego"
