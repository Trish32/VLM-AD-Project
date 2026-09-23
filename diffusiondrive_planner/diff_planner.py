"""The truncated-diffusion planner: the x2 operation loop plus the denoising loop.

This is the wiring that turns the individually-validated components into a planner. Two
nested loops:

    for k in [20, 0]:                      # denoising steps (truncated schedule)
        for op in diff_operation_order:    # x2 repetition of the block below
            traj_pooler -> self_attn -> norm -> agent_cross_gnn -> norm
              -> anchor_cross_gnn -> norm -> ffn -> norm -> modulation -> diff_refine
        x = scheduler.step(normalize(x0_pred), k, x)

Everything upstream of the detector is injected rather than owned: this module takes
already-selected agent features and a plan-query bank, so it can be exercised on a Mac
with synthetic detector output and later bolted onto the real SparseDrive port unchanged.

Wiring details that are easy to get wrong
-----------------------------------------
* `agent_cross_gnn` goes through **decoupled attention** (`decouple_attn_motion=True`):
  query and key are concatenated with their positional embeddings to 2*embed_dims, value
  passes through a shared `fc_before`, and `fc_after` projects the result back down.
  Same pattern as Sparse4D v3. `anchor_cross_gnn` and `self_attn` do NOT decouple.
* `traj_pooler` receives the *current* prediction (deltas), which it cumsums internally.
  On the first pass that is the seeded noisy anchor; afterwards it is `diff_refine`'s
  output, un-replicated back to K modes.
* `modulation` receives the timestep embedding shaped `(bs, K, C)`.
* `diff_refine` returns `(plan_reg, plan_cls)`, replicated x3 across commands.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention_compat import MultiheadAttentionCompat
from .diff_head import DiffRefine, ModulationLayer, undo_command_replication
from .ffn import AsymmetricFFN
from .plan_query import TimeEmbedder, TrajEmbedder, gen_sineembed_for_position
from .traj_pooler import TrajPooler
from .truncated_diffusion import (
    TruncatedDDIM,
    anchor_to_deltas,
    denormalize_traj,
    normalize_traj,
)

DIFF_OPERATION_ORDER = (
    "traj_pooler",
    "self_attn",
    "norm",
    "agent_cross_gnn",
    "norm",
    "anchor_cross_gnn",
    "norm",
    "ffn",
    "norm",
    "modulation",
    "diff_refine",
)


class DiffPlanner(nn.Module):
    """Truncated-diffusion ego planner.

    Args:
        embed_dims, ego_fut_ts, ego_fut_mode: as in the stage-2 config.
        num_repeats: how many times `DIFF_OPERATION_ORDER` is repeated (2 upstream).
        decouple_attn: use the 2*embed_dims decoupled path for agent cross-attention.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        ego_fut_ts: int = 6,
        ego_fut_mode: int = 6,
        num_heads: int = 8,
        ffn_channels: int = 512,
        num_repeats: int = 2,
        decouple_attn: bool = True,
        operation_order: tuple[str, ...] = DIFF_OPERATION_ORDER,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.decouple_attn = decouple_attn
        self.operation_order = tuple(operation_order) * num_repeats

        attn_dims = embed_dims * 2 if decouple_attn else embed_dims
        if decouple_attn:
            self.fc_before = nn.Linear(embed_dims, attn_dims, bias=False)
            self.fc_after = nn.Linear(attn_dims, embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        # One module per operation slot, so indices line up with operation_order exactly
        # (upstream indexes self.diff_layers[i] by position).
        layers: list[nn.Module] = []
        for op in self.operation_order:
            if op == "traj_pooler":
                layers.append(TrajPooler(embed_dims, ego_fut_ts))
            elif op == "self_attn":
                layers.append(MultiheadAttentionCompat(embed_dims, num_heads))
            elif op == "agent_cross_gnn":
                layers.append(MultiheadAttentionCompat(attn_dims, num_heads))
            elif op == "anchor_cross_gnn":
                layers.append(MultiheadAttentionCompat(embed_dims, num_heads))
            elif op == "norm":
                layers.append(nn.LayerNorm(embed_dims))
            elif op == "ffn":
                layers.append(
                    AsymmetricFFN(
                        in_channels=embed_dims,
                        pre_norm=True,
                        embed_dims=embed_dims,
                        feedforward_channels=ffn_channels,
                        num_fcs=2,
                    )
                )
            elif op == "modulation":
                layers.append(ModulationLayer(embed_dims, if_global_cond=False))
            elif op == "diff_refine":
                layers.append(DiffRefine(embed_dims, ego_fut_ts, ego_fut_mode))
            else:
                raise ValueError(f"unknown operation {op!r}")
        self.diff_layers = nn.ModuleList(layers)

        self.traj_embedder = TrajEmbedder(embed_dims, ego_fut_ts)
        self.time_mlp = TimeEmbedder(embed_dims)
        self.scheduler = TruncatedDDIM()

    # ------------------------------------------------------------------ helpers

    def _agent_cross(self, index, query, key, value, query_pos, key_pos):
        """Decoupled cross-attention, mirroring the head's `diff_graph_model`."""
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            key = torch.cat([key, key_pos], dim=-1)
            query_pos = key_pos = None
        value = self.fc_before(value)
        out = self.diff_layers[index](
            query, key, value, query_pos=query_pos, key_pos=key_pos
        )
        return self.fc_after(out)

    def _run_block(
        self,
        traj_feature,
        diff_plan_reg,
        time_embed,
        agent_feature,
        agent_pos,
        ego_pos,
        anchor_query,
        metas,
        feature_maps,
    ):
        """One pass over `operation_order`, returning (traj_feature, reg, cls)."""
        bs = traj_feature.shape[0]
        plan_reg = plan_cls = None

        for i, op in enumerate(self.operation_order):
            layer = self.diff_layers[i]
            if op == "traj_pooler":
                if diff_plan_reg.dim() != 3:
                    diff_plan_reg = undo_command_replication(diff_plan_reg, self.ego_fut_mode)
                traj_feature = layer(
                    traj_feature, diff_plan_reg, metas, feature_maps,
                    modal_num=self.ego_fut_mode,
                )
            elif op == "self_attn":
                traj_feature = layer(traj_feature, traj_feature, traj_feature)
            elif op == "agent_cross_gnn":
                traj_feature = self._agent_cross(
                    i, traj_feature, agent_feature, agent_feature,
                    query_pos=ego_pos, key_pos=agent_pos,
                )
            elif op == "anchor_cross_gnn":
                traj_feature = layer(traj_feature, key=anchor_query, value=anchor_query)
            elif op == "modulation":
                traj_feature = layer(traj_feature, time_embed.view(bs, self.ego_fut_mode, -1))
            elif op in ("norm", "ffn"):
                traj_feature = layer(traj_feature)
            elif op == "diff_refine":
                plan_reg, plan_cls = layer(traj_feature)
                diff_plan_reg = plan_reg
        return traj_feature, plan_reg, plan_cls

    # ------------------------------------------------------------------ forward

    @torch.no_grad()
    def forward(
        self,
        plan_anchor: torch.Tensor,   # (bs, K, T, 2) command-selected ABSOLUTE waypoints
        anchor_query: torch.Tensor,  # (bs, K, C)   command-selected mode queries
        agent_feature: torch.Tensor, # (bs, N, C)   selected detection features
        agent_pos: torch.Tensor,     # (bs, N, C)   their anchor embeddings
        ego_pos: torch.Tensor,       # (bs, K, C)   ego anchor embedding, repeated per mode
        metas: dict,
        feature_maps,
        step_num: int = 2,
    ):
        """Run the truncated denoising loop. Returns (plan_reg, plan_cls)."""
        bs = plan_anchor.shape[0]
        device = plan_anchor.device

        # Seed from the anchor at t=8 rather than from pure noise — the whole trick.
        deltas = anchor_to_deltas(plan_anchor)
        img = self.scheduler.seed_from_anchor(
            deltas.reshape(bs * self.ego_fut_mode, self.ego_fut_ts, 2)
        )

        plan_reg = plan_cls = None
        for k in self.scheduler.inference_timesteps(step_num).tolist():
            x = img.clamp(-1, 1)
            noisy = denormalize_traj(x)

            traj_feature = self.traj_embedder(noisy, bs, self.ego_fut_mode)
            t = torch.full((bs * self.ego_fut_mode,), k, device=device, dtype=torch.float32)
            time_embed = self.time_mlp(t)

            traj_feature, plan_reg, plan_cls = self._run_block(
                traj_feature, noisy, time_embed,
                agent_feature, agent_pos, ego_pos, anchor_query,
                metas, feature_maps,
            )

            x0 = normalize_traj(undo_command_replication(plan_reg, self.ego_fut_mode))
            img = self.scheduler.step(x0, k, img)

        return plan_reg, plan_cls
