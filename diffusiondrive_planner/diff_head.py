"""Diffusion refinement head and time modulation for the DiffusionDrive planner.

Ports `V4DiffMotionPlanningRefinementModule` and `V1ModulationLayer`
(diff_motion_blocks.py) — the two leaf modules of the `diff_operation_order` loop that
carry learned weights and produce the actual output.

The x3 replication
------------------
`DiffRefine` predicts for the SELECTED command's `K` modes only, then replicates the
result across all 3 commands so the output shape matches what the loss and the nuScenes
planning eval expect:

    plan_reg: (bs, 1, 3K, T, 2)   but only K distinct trajectories
    plan_cls: (bs, 1, 3K)         likewise

That is genuinely redundant — upstream computes `K` and repeats. It matters for two
reasons: the checkpoint's branch widths are sized for `K`, not `3K`; and when the loop
feeds the prediction back for the next denoising step it slices `[..., -K:, :]` to undo
the replication. Slicing a different block, or treating `3K` as real modes, both keep
every shape valid.

Modulation is FiLM
------------------
`traj_feature * (1 + scale) + shift`, with `(scale, shift)` chunked from a Mish+Linear on
the timestep embedding. The stage-2 config sets `if_zeroinit_scale=False`, so it is NOT
zero-initialised despite that being the usual convention for FiLM in diffusion models —
reproduce the config, not the convention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .plan_query import linear_relu_ln

NUM_COMMANDS = 3


def bias_init_with_prob(prior_prob: float) -> float:
    """Focal-loss bias init. Port of mmdet's helper, inlined to avoid the dependency."""
    import math

    return float(-math.log((1 - prior_prob) / prior_prob))


class ModulationLayer(nn.Module):
    """FiLM conditioning on the diffusion timestep. Port of `V1ModulationLayer`."""

    def __init__(
        self,
        embed_dims: int = 256,
        if_global_cond: bool = False,
        if_zeroinit_scale: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.if_global_cond = if_global_cond
        self.if_zeroinit_scale = if_zeroinit_scale
        in_dims = embed_dims * 2 if if_global_cond else embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(in_dims, embed_dims * 2),
        )

    def init_weight(self) -> None:
        # Stage-2 config sets this False, so by default nothing happens here.
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature: torch.Tensor,
        time_embed: torch.Tensor,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if global_cond is not None:
            global_feature = torch.cat([global_cond, time_embed], dim=-1)
        else:
            global_feature = time_embed
        scale, shift = self.scale_shift_mlp(global_feature).chunk(2, dim=-1)
        return traj_feature * (1 + scale) + shift


class DiffRefine(nn.Module):
    """Trajectory feature -> (plan_reg, plan_cls).

    Port of `V4DiffMotionPlanningRefinementModule`. Note the two branches have different
    shapes: `plan_cls_branch` uses `linear_relu_ln` (with LayerNorm), `plan_reg_branch`
    is a plain 3-layer MLP with no norm. Not interchangeable.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        ego_fut_ts: int = 6,
        ego_fut_mode: int = 6,
        if_zeroinit_reg: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.if_zeroinit_reg = if_zeroinit_reg

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )

    def init_weight(self) -> None:
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init_with_prob(0.01))

    def forward(self, traj_feature: torch.Tensor):
        """(bs, K, C) -> plan_reg (bs, 1, 3K, T, 2), plan_cls (bs, 1, 3K)."""
        bs = traj_feature.shape[0]
        traj_feature = traj_feature.view(bs, 1, self.ego_fut_mode, -1)

        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        plan_cls = plan_cls.repeat(1, NUM_COMMANDS, 1).reshape(bs, 1, -1)

        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(
            bs, 1, self.ego_fut_mode, self.ego_fut_ts, 2
        ).repeat(1, NUM_COMMANDS, 1, 1, 1)
        plan_reg = plan_reg.view(bs, 1, NUM_COMMANDS * self.ego_fut_mode, self.ego_fut_ts, 2)

        return plan_reg, plan_cls


def undo_command_replication(plan_reg: torch.Tensor, ego_fut_mode: int) -> torch.Tensor:
    """(bs, 1, 3K, T, 2) -> (bs*K, T, 2) for the next denoising step.

    Mirrors the head's `diff_plan_reg[:, :, -ego_fut_mode:, ].flatten(0, 2)`. The last K
    of the 3K entries are taken; because the block is a plain 3x replication they are
    identical to the first K, but the slice must stay aligned with how DiffRefine builds
    the output.
    """
    return plan_reg[:, :, -ego_fut_mode:, ].flatten(0, 2)
