"""Trajectory feature pooling — what makes the DiffusionDrive denoiser image-aware.

Port of `V1TrajPooler` (diff_motion_blocks.py:467) plus its keypoint generator, in plain
torch on top of our `dfa_torch.py`. Without this the planner is a trajectory prior that
never looks at the cameras; with it, each denoising step samples multi-view image
features along the *current noisy trajectory* and feeds them back into the next step.

Pipeline
--------
    trajs (deltas)  --cumsum-->  absolute waypoints
                    --kps_generator-->  (bs, modal, T*5, 3)   5 heights per waypoint
                    --project_points-->  (bs, modal, T*5, 6, 2) normalized image coords
    instance_feature --weights_fc-->     (bs, modal, 6, 4, T*5, 8)
                    --DFA-->             (bs, modal, 256)
                    --output_proj + residual-->  updated traj feature

Traps
-----
1. **`forward` cumsums first.** The diffusion state is consecutive deltas; the pooler
   needs world positions. Feeding deltas straight in projects points near the ego and
   samples the wrong pixels — with no shape error anywhere.
2. **Weight layout: group is the fastest-varying index.** `weights_fc` output is
   reshaped `(bs, num_anchor, -1, num_groups)` and softmaxed over `dim=-2`, i.e. jointly
   across cam x level x point per group. This is the same layout that cost days in the
   Sparse4D port; see that project's bug log.
3. **The camera encoder eats 12 numbers**: `projection_mat[:, :, :3].reshape(bs, cams, -1)`
   — the top 3 rows of the 4x4, flattened. Not the full 16.
4. **Keypoint z is `ground_height + fix_height`**, with `ground_height=-1.84023` (the
   nuScenes lidar-to-ground offset) and 5 fixed heights, giving `T*5 = 30` points.
5. `project_points` divides by `image_wh`, so coordinates arrive at the DFA already
   normalized to [0, 1] — which is the range `dfa_torch` treats as in-view.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dfa_torch import deformable_aggregation_torch
from .plan_query import linear_relu_ln

DEFAULT_FIX_HEIGHT = (0, 0.5, -0.5, 1, -1)
DEFAULT_GROUND_HEIGHT = -1.84023


class TrajKeypointGenerator(nn.Module):
    """Trajectory waypoints -> 3D keypoints at several fixed heights.

    Port of `TrajSparsePoint3DKeyPointsGenerator`, single-frame path only (the temporal
    branch needs `T_cur2temp_list`, which the planner never passes).

    `num_learnable_pts` is 0 in the shipped config and the learnable projection is
    commented out upstream, so the per-point offset is an explicit zero — kept here so
    the structure stays recognisable against the reference.
    """

    def __init__(
        self,
        num_sample: int = 6,
        fix_height: tuple = DEFAULT_FIX_HEIGHT,
        ground_height: float = DEFAULT_GROUND_HEIGHT,
    ) -> None:
        super().__init__()
        self.num_sample = num_sample
        self.fix_height = np.array(fix_height)
        self.ground_height = ground_height

    @property
    def num_pts(self) -> int:
        return self.num_sample * len(self.fix_height)

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        """(bs, num_anchor, num_sample*2) -> (bs, num_anchor, num_sample*len(h), 3)."""
        bs, num_anchor, _ = anchor.shape
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)

        offset = torch.zeros(
            [bs, num_anchor, self.num_sample, len(self.fix_height), 1, 2],
            device=anchor.device,
            dtype=anchor.dtype,
        )
        key_points = offset + key_points[..., None, None, :]

        # Append z = ground_height, then add the per-height offset.
        key_points = torch.cat(
            [
                key_points,
                key_points.new_full(key_points.shape[:-1] + (1,), self.ground_height),
            ],
            dim=-1,
        )
        fix_height = key_points.new_tensor(self.fix_height)
        height_offset = key_points.new_zeros([len(fix_height), 2])
        height_offset = torch.cat([height_offset, fix_height[:, None]], dim=-1)
        key_points = key_points + height_offset[None, None, None, :, None]

        return key_points.flatten(2, 4)


def project_points(
    key_points: torch.Tensor, projection_mat: torch.Tensor, image_wh: torch.Tensor | None = None
) -> torch.Tensor:
    """(bs, A, P, 3) + (bs, cams, 4, 4) -> (bs, cams, A, P, 2).

    Perspective divide with `clamp(z, min=1e-5)`: points behind the camera are NOT
    masked here, they fold to huge coordinates and are dropped later by the DFA's
    in-view test. That is upstream's behaviour and it matters — masking here instead
    would change which points contribute.
    """
    pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
    points_2d = torch.matmul(
        projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
    ).squeeze(-1)
    points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
    if image_wh is not None:
        points_2d = points_2d / image_wh[:, :, None, None]
    return points_2d


class TrajPooler(nn.Module):
    """Port of `V1TrajPooler`. Samples image features along the noisy trajectory."""

    def __init__(
        self,
        embed_dims: int = 256,
        ego_fut_ts: int = 6,
        num_cams: int = 6,
        num_levels: int = 4,
        num_groups: int = 8,
        attn_drop: float = 0.0,
        residual_mode: str = "add",
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode

        self.kps_generator = TrajKeypointGenerator(num_sample=ego_fut_ts)
        self.num_pts = self.kps_generator.num_pts  # ego_fut_ts * 5

        # 12 = the top 3 rows of the 4x4 projection matrix, flattened.
        self.camera_encoder = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, 12))
        self.weights_fc = nn.Linear(embed_dims, num_groups * num_levels * self.num_pts)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(0.0)

    def _get_weights(self, instance_feature: torch.Tensor, metas: dict) -> torch.Tensor:
        """(bs, A, C) -> (bs, A, cams, levels, pts, groups), softmaxed group-wise."""
        bs, num_anchor = instance_feature.shape[:2]

        camera_embed = self.camera_encoder(
            metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1)
        )
        feature = instance_feature[:, :, None] + camera_embed[:, None]

        # Group is the FASTEST-VARYING index: reshape to (..., -1, G) and softmax over
        # dim=-2 normalises jointly across cam x level x point within each group.
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs, num_anchor, self.num_cams, self.num_levels, self.num_pts, self.num_groups
            )
        )

        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1,
                device=weights.device, dtype=weights.dtype,
            )
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)
        return weights

    def pool_feature_from_traj(
        self,
        instance_feature: torch.Tensor,
        traj_points: torch.Tensor,
        metas: dict,
        feature_maps,
        modal_num: int = 1,
    ) -> torch.Tensor:
        """`traj_points` must already be ABSOLUTE waypoints (see forward)."""
        bs_modal = traj_points.shape[0]
        bs = bs_modal // modal_num
        plan_reg = traj_points.view(bs, modal_num, self.ego_fut_ts * 2)

        key_points = self.kps_generator(plan_reg)
        weights = self._get_weights(instance_feature, metas)

        points_2d = (
            project_points(key_points, metas["projection_mat"], metas.get("image_wh"))
            .permute(0, 2, 3, 1, 4)
            .reshape(bs, modal_num, self.num_pts, self.num_cams, 2)
        )
        weights = (
            weights.permute(0, 1, 4, 2, 3, 5)
            .contiguous()
            .reshape(bs, modal_num, self.num_pts, self.num_cams, self.num_levels, self.num_groups)
        )

        features = deformable_aggregation_torch(*feature_maps, points_2d, weights).reshape(
            bs, modal_num, self.embed_dims
        )

        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            return output + instance_feature
        if self.residual_mode == "cat":
            return torch.cat([output, instance_feature], dim=-1)
        raise ValueError(f"unknown residual_mode {self.residual_mode!r}")

    def forward(self, instance_feature, trajs, metas, feature_maps, modal_num: int = 1):
        """`trajs` are DELTAS; cumsum to absolute before pooling. See trap 1."""
        trajs_cum = trajs.cumsum(dim=-2)
        return self.pool_feature_from_traj(
            instance_feature, trajs_cum, metas, feature_maps, modal_num=modal_num
        )
