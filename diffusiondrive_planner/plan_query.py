"""Plan anchor queries and trajectory embeddings for the DiffusionDrive planner.

Ports the query-construction half of `V13MotionPlanningHead` — everything between the
anchor `.npy` and the `diff_operation_order` loop — in plain torch, so it drops onto our
SparseDrive port without mmdet3d. Validated against upstream's own functions in
`tests/test_plan_query.py` (they import fine in the `uniad2.0` env).

Three embeddings, each with a different contract, which is the main source of confusion:

| what                 | source                       | hidden_dim | encoder          |
|----------------------|------------------------------|-----------|------------------|
| plan mode query      | anchor **endpoint** `[...,-1,:]` | 256 (default) | `plan_anchor_encoder` |
| trajectory feature   | **all** noisy waypoints, flattened | **128**   | `plan_pos_encoder` (in 768) |
| diffusion time embed | scalar timestep              | 256       | `time_mlp`       |

The trajectory one uses `hidden_dim=128` precisely so `ego_fut_ts * 128 = 6 * 128 = 768`
matches `plan_pos_encoder`'s declared input width. Passing the 256 default there produces
1536 and fails loudly — which is the good case. The traps below fail silently.

Traps
-----
1. `gen_sineembed_for_position` returns `cat([pos_y, pos_x])` — **y first**. Swapping the
   order keeps every shape valid and every downstream layer happy.
2. Sine embeddings are computed on **raw metres × 2π**, with no normalization. Anchor
   endpoints can be ~40m; trajectory deltas are ~0-7m. Normalizing first would change
   the embedding scale entirely.
3. `plan_mode_query` is laid out **command-major**: `(bs, 3, K, C).flatten(1, 2)` gives
   `(bs, 3K, C)`, and the head later recovers it with `.view(bs, 3, K, -1)`. Building it
   mode-major round-trips through the same shapes while pairing each command with the
   wrong anchors.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

EGO_FUT_TS = 6
EGO_FUT_MODE = 6
NUM_COMMANDS = 3  # right, left, straight
TRAJ_HIDDEN_DIM = 128


def gen_sineembed_for_position(pos_tensor: torch.Tensor, hidden_dim: int = 256) -> torch.Tensor:
    """(..., 2) -> (..., hidden_dim). Exact port of attention.py:317 (DAB-DETR lineage).

    Note the output ordering: `cat([pos_y, pos_x])`, y first.
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    # Integer division pairs consecutive channels onto the same frequency, so that the
    # sin/cos interleave below forms proper (sin, cos) couples.
    # torch warns that `//` truncates rather than floors; harmless here because dim_t is
    # a non-negative arange, so trunc == floor. Kept as-is to stay byte-identical to
    # upstream, which emits the same warning.
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


class SinusoidalPosEmb(nn.Module):
    """Scalar -> (dim,) embedding for the diffusion timestep. Port of conditional_unet1d.py:44.

    Distinct from `gen_sineembed_for_position`: log-spaced frequencies, and `cat(sin, cos)`
    rather than an interleave. They are not interchangeable.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def linear_relu_ln(embed_dims: int, in_loops: int, out_loops: int, input_dims: int | None = None):
    """Port of blocks.py:32. Returns a list of layers for nn.Sequential."""
    if input_dims is None:
        input_dims = embed_dims
    layers: list[nn.Module] = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class PlanAnchorQueryBank(nn.Module):
    """Holds the `(3, K, T, 2)` anchors and turns them into command-selectable queries.

    The anchors are a frozen buffer, exactly as upstream registers them
    (`nn.Parameter(..., requires_grad=False)`), so they round-trip through the official
    checkpoint rather than being overwritten by our regenerated values.
    """

    def __init__(self, plan_anchor: torch.Tensor, embed_dims: int = 256) -> None:
        super().__init__()
        if plan_anchor.shape[0] != NUM_COMMANDS or plan_anchor.dim() != 4:
            raise ValueError(
                f"plan_anchor must be (3, K, T, 2), got {tuple(plan_anchor.shape)}. "
                "Upstream's shipped kmeans_plan.py emits (K, T, 2) — see DESIGN.md."
            )
        # nn.Parameter with requires_grad=False (not register_buffer) to match upstream's
        # state_dict key kind exactly; a buffer would still load but is a different type.
        self.plan_anchor = nn.Parameter(plan_anchor.float(), requires_grad=False)
        self.ego_fut_mode = plan_anchor.shape[1]
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (plan_anchor (bs,3,K,T,2), plan_mode_query (bs,1,3K,C))."""
        plan_anchor = torch.tile(self.plan_anchor[None], (batch_size, 1, 1, 1, 1))
        # Endpoint only, at the default hidden_dim=256.
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)
        return plan_anchor, plan_mode_query


def select_by_command(plan_query: torch.Tensor, cmd: torch.Tensor, ego_fut_mode: int) -> torch.Tensor:
    """(bs, 1, 3K, C) + (bs,) command index -> (bs, K, C).

    Mirrors motion_planning_head_v13.py:1018-1021. The `.view(bs, 3, K, -1)` here must
    match the command-major `.flatten(1, 2)` used to build the query.
    """
    bs = plan_query.shape[0]
    nav = plan_query.squeeze(1).view(bs, NUM_COMMANDS, ego_fut_mode, -1)
    return nav[torch.arange(bs, device=plan_query.device), cmd]


def select_anchor_by_command(plan_anchor: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    """(bs, 3, K, T, 2) + (bs,) -> (bs, K, T, 2)."""
    bs = plan_anchor.shape[0]
    return plan_anchor[torch.arange(bs, device=plan_anchor.device), cmd]


class TrajEmbedder(nn.Module):
    """Noisy trajectory waypoints -> per-mode query feature.

    Upstream (motion_planning_head_v13.py:1040-1044):
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=128)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature   = self.plan_pos_encoder(traj_pos_embed).view(bs, ego_fut_mode, -1)
    """

    def __init__(
        self, embed_dims: int = 256, ego_fut_ts: int = EGO_FUT_TS,
        hidden_dim: int = TRAJ_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dims = ego_fut_ts * hidden_dim  # 6 * 128 = 768
        self.plan_pos_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1, self.input_dims),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward(self, noisy_traj_points: torch.Tensor, batch_size: int, ego_fut_mode: int):
        """(bs*K, T, 2) -> (bs, K, C)."""
        emb = gen_sineembed_for_position(noisy_traj_points, hidden_dim=self.hidden_dim)
        emb = emb.flatten(-2)
        return self.plan_pos_encoder(emb).view(batch_size, ego_fut_mode, -1)


class TimeEmbedder(nn.Module):
    """Diffusion timestep -> conditioning vector. Port of the `time_mlp` Sequential."""

    def __init__(self, embed_dims: int = 256) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dims),
            nn.Linear(embed_dims, embed_dims * 4),
            nn.Mish(),
            nn.Linear(embed_dims * 4, embed_dims),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(timesteps)
