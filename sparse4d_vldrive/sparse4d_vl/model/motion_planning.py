"""
SparseDrive-style Motion Planner for Sparse4D-v3 — pure PyTorch, MPS.

End-to-end perception → prediction → planning on top of v3's sparse instance
features (the same recurrent queries that drive detection + tracking):

  AgentMotionHead  — agent–agent interaction (instance self-attention) then an
                     ANCHORED multi-modal motion head: K kmeans trajectory
                     anchors per agent, regress offset-to-anchor + mode class.
  EgoPlanner       — a learnable ego query cross-attends to agent features, then
                     a COMMAND-CONDITIONED anchored multi-modal head: per driving
                     command (right/straight/left) K ego-trajectory anchors,
                     regress offset + class. Trained with a collision-aware loss.

Trajectories are (x,y) DISPLACEMENTS from the current position, in the current
lidar (agents) / ego (planner) frame, over T future keyframes — matching the GT
produced by data/finetune_loader.py.

Simplification vs the paper: anchors are clustered in the scene frame (not each
agent's heading frame), interaction is a single attention layer, and online
mapping / agent–map attention is omitted (no map module in this port).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import GraphAttention


# ---------------------------------------------------------------------------
# Anchor clustering (pure-numpy kmeans over GT future trajectories)
# ---------------------------------------------------------------------------

def _kmeans(x: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    """x: (N, D) → centers (k, D). Plain Lloyd's algorithm."""
    rng = np.random.default_rng(seed)
    if x.shape[0] < k:
        # too few samples — pad by repeating
        reps = int(np.ceil(k / max(x.shape[0], 1)))
        x = np.tile(x, (reps, 1))[:max(k, x.shape[0])]
    centers = x[rng.choice(x.shape[0], k, replace=False)].copy()
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None]) ** 2).sum(-1)   # (N, k)
        assign = d.argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].mean(0)
    return centers


def build_anchors(loader, num_modes: int = 6, future_steps: int = 12,
                  ego_steps: int = 6, cache: str | None = None):
    """
    Cluster GT agent + ego future trajectories into anchor sets (SparseDrive
    fut_mode=6/fut_ts=12 for agents; ego_fut_mode=3 commands / ego_fut_ts=6).

    Returns
    -------
    agent_anchors : (K, T, 2)            K kmeans motion anchors
    ego_anchors   : (3, ego_steps, 2)    one anchor per driving command
    """
    import os
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return (torch.from_numpy(z['agent']).float(),
                torch.from_numpy(z['ego']).float())

    agent_traj, ego_by_cmd = [], {0: [], 1: [], 2: []}
    for frame in loader:
        gm = frame.get('gt_future_mask')
        if gm is not None and gm.numel() > 0:
            full = gm.all(dim=1)                  # only fully-observed futures
            fut = frame['gt_futures'][full].numpy()           # (n, T, 2) lidar frame
            box = frame['gt_boxes'][full].numpy()             # (n, 11) sin@6, cos@7
            for i in range(fut.shape[0]):
                # rotate the future into the AGENT heading frame so anchors are
                # canonical (e.g. "go straight" / "turn left" relative to heading)
                sin, cos = box[i, 6], box[i, 7]
                ax = cos * fut[i, :, 0] + sin * fut[i, :, 1]
                ay = -sin * fut[i, :, 0] + cos * fut[i, :, 1]
                agent_traj.append(np.stack([ax, ay], -1).reshape(-1))
        if 'ego_future' in frame and bool(frame['ego_future_mask'][:ego_steps].all()):
            ego_by_cmd[int(frame['command'])].append(
                frame['ego_future'][:ego_steps].numpy().reshape(-1))

    T = future_steps
    agent = _kmeans(np.stack(agent_traj), num_modes).reshape(num_modes, T, 2) \
        if agent_traj else np.zeros((num_modes, T, 2), np.float32)
    # ego: one anchor per command = mean ego trajectory of that command
    ego = np.zeros((3, ego_steps, 2), np.float32)
    for c in range(3):
        if ego_by_cmd[c]:
            ego[c] = np.stack(ego_by_cmd[c]).mean(0).reshape(ego_steps, 2)
    agent_t = torch.from_numpy(agent.astype(np.float32))
    ego_t = torch.from_numpy(ego.astype(np.float32))
    if cache:
        np.savez(cache, agent=agent_t.numpy(), ego=ego_t.numpy())
    return agent_t, ego_t


# ---------------------------------------------------------------------------
# Agent motion head (interaction + anchored multi-modal)
# ---------------------------------------------------------------------------

class MapEncoder(nn.Module):
    """PointNet-style encoder: each map polyline (P points) → one map token.
    Stands in for SparseDrive's online-mapping instances using the nuScenes HD
    map polylines (lane/road dividers, ped crossings) in the current lidar frame."""

    def __init__(self, embed_dims=256, hidden=128):
        super().__init__()
        self.pt = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(inplace=True),
            nn.Linear(64, hidden), nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(hidden, embed_dims)

    def forward(self, map_pts):           # (B, M, P, 2) → (B, M, D)
        f = self.pt(map_pts)              # (B, M, P, hidden)
        f = f.max(dim=2).values          # max-pool over polyline points
        return self.out(f)


class AgentMotionHead(nn.Module):
    """SparseDrive agent motion: agent–agent (+ optional agent–map) interaction,
    then ANCHORED multi-modal prediction in the AGENT heading frame, rotated back
    to the lidar frame via each agent's yaw."""

    def __init__(self, embed_dims=256, num_modes=6, future_steps=12,
                 agent_anchors: torch.Tensor | None = None, with_map: bool = False):
        super().__init__()
        self.K, self.T = num_modes, future_steps
        self.interact = GraphAttention(embed_dims, num_heads=8, dropout=0.1)
        self.norm = nn.LayerNorm(embed_dims)
        # agent–map cross-attention (optional)
        if with_map:
            self.map_cross = nn.MultiheadAttention(embed_dims, 8, dropout=0.1, batch_first=True)
            self.map_norm = nn.LayerNorm(embed_dims)
        else:
            self.map_cross = None
        self.reg = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_modes * future_steps * 2),
        )
        self.cls = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_modes),
        )
        if agent_anchors is None:
            agent_anchors = torch.zeros(num_modes, future_steps, 2)
        self.register_buffer('anchors', agent_anchors)     # (K, T, 2) AGENT frame

    def forward(self, instance_feature, anchor, map_feature=None, map_mask=None,
                anchor_embed=None):
        """
        instance_feature (B,N,D); anchor (B,N,11) for yaw → traj (B,N,K,T,2)
        in LIDAR frame, mode_logits (B,N,K).
        """
        B, N, D = instance_feature.shape
        # agent–agent interaction (residual self-attention over instances)
        x = instance_feature + self.interact(instance_feature, query_pos=anchor_embed)
        x = self.norm(x)
        # agent–map cross-attention (skip if no valid map elements)
        if self.map_cross is not None and map_feature is not None \
                and map_mask is not None and bool(map_mask.any()):
            kpm = ~map_mask                                # True = ignore
            attn, _ = self.map_cross(x, map_feature, map_feature,
                                     key_padding_mask=kpm, need_weights=False)
            x = self.map_norm(x + attn)

        offset = self.reg(x).view(B, N, self.K, self.T, 2)
        traj_a = self.anchors.view(1, 1, self.K, self.T, 2) + offset   # agent frame

        # rotate agent-frame → lidar frame using each agent's heading (sin@6,cos@7)
        sin = anchor[..., 6].view(B, N, 1, 1)
        cos = anchor[..., 7].view(B, N, 1, 1)
        lx = cos * traj_a[..., 0] - sin * traj_a[..., 1]
        ly = sin * traj_a[..., 0] + cos * traj_a[..., 1]
        traj = torch.stack([lx, ly], dim=-1)              # (B,N,K,T,2) lidar frame
        return traj, self.cls(x)


# ---------------------------------------------------------------------------
# Ego planner (command-conditioned anchored multi-modal)
# ---------------------------------------------------------------------------

class EgoPlanner(nn.Module):
    """SparseDrive ego planner: ego_fut_mode=3 modes (= driving commands),
    ego_fut_ts=6 (3 s). A learnable ego query cross-attends to agent features,
    then an anchored multi-modal head regresses an offset per command-mode."""

    def __init__(self, embed_dims=256, ego_modes=3, ego_steps=6,
                 ego_anchors: torch.Tensor | None = None, with_map: bool = False):
        super().__init__()
        self.M, self.T = ego_modes, ego_steps
        self.ego_query = nn.Parameter(torch.zeros(1, 1, embed_dims))
        nn.init.normal_(self.ego_query, std=0.02)
        self.cross = nn.MultiheadAttention(embed_dims, 8, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(embed_dims)
        # ego–map cross-attention (optional): the plan attends to road geometry
        if with_map:
            self.map_cross = nn.MultiheadAttention(embed_dims, 8, dropout=0.1, batch_first=True)
            self.map_norm = nn.LayerNorm(embed_dims)
        else:
            self.map_cross = None
        self.reg = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, ego_modes * ego_steps * 2),
        )
        self.cls = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, ego_modes),
        )
        if ego_anchors is None:
            ego_anchors = torch.zeros(ego_modes, ego_steps, 2)
        self.register_buffer('anchors', ego_anchors)       # (M, T, 2)

    def forward(self, agent_feature, map_feature=None, map_mask=None):
        """agent_feature (B,N,D) → ego_traj (B,M,T,2), ego_logits (B,M).
        Mode index m == driving command (0=right, 1=straight, 2=left)."""
        B = agent_feature.shape[0]
        q = self.ego_query.expand(B, -1, -1)               # (B,1,D)
        ego, _ = self.cross(q, agent_feature, agent_feature, need_weights=False)
        ego = self.norm(q + ego)                           # (B,1,D)
        # ego attends to map geometry (lanes/dividers) for road-compliant plans
        if self.map_cross is not None and map_feature is not None \
                and map_mask is not None and bool(map_mask.any()):
            mattn, _ = self.map_cross(ego, map_feature, map_feature,
                                      key_padding_mask=~map_mask, need_weights=False)
            ego = self.map_norm(ego + mattn)
        ego = ego.squeeze(1)                               # (B, D)
        offset = self.reg(ego).view(B, self.M, self.T, 2)
        traj = self.anchors.view(1, self.M, self.T, 2) + offset
        logits = self.cls(ego)                             # (B, M)
        return traj, logits


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class AgentMotionLoss(nn.Module):
    """Anchored winner-takes-all: best mode by masked ADE → L1 reg + CE class."""

    def __init__(self, weight_reg=1.0, weight_cls=0.5):
        super().__init__()
        self.wr, self.wc = weight_reg, weight_cls

    def forward(self, traj, mode_logits, gt_future, gt_mask):
        if traj.shape[0] == 0:
            return traj.sum() * 0.0
        has = gt_mask.any(1)
        if has.sum() == 0:
            return traj.sum() * 0.0
        traj, mode_logits = traj[has], mode_logits[has]
        gt, m = gt_future[has], gt_mask[has].float()
        P, K, T, _ = traj.shape
        l2 = (traj - gt.unsqueeze(1)).norm(dim=-1)                  # (P,K,T)
        ade = (l2 * m.unsqueeze(1)).sum(-1) / m.sum(-1, keepdim=True).clamp(min=1)
        best = ade.argmin(1)
        ar = torch.arange(P, device=traj.device)
        reg = (F.smooth_l1_loss(traj[ar, best], gt, reduction='none').sum(-1) * m).sum() \
            / m.sum().clamp(min=1)
        cls = F.cross_entropy(mode_logits, best)
        return self.wr * reg + self.wc * cls


class PlanLoss(nn.Module):
    """Ego planning where mode index == driving command. Regress the GT
    command's modal trajectory (L1), classify the command (CE), plus a soft
    collision penalty against agent GT future positions (paper-motivated extra;
    not in the released head, which instead regresses an ego-status vector)."""

    def __init__(self, weight_reg=1.0, weight_cls=0.5, weight_col=1.0,
                 col_margin=2.0):
        super().__init__()
        self.wr, self.wc, self.wcol, self.margin = weight_reg, weight_cls, weight_col, col_margin

    def forward(self, ego_traj, ego_logits, command, ego_future, ego_mask,
                agent_future=None, agent_mask=None):
        # ego_traj (B,M,Te,2); the GT command selects the supervised mode.
        B, M, Te, _ = ego_traj.shape
        ego_future = ego_future[:, :Te]                    # (B,Te,2)
        m = ego_mask[:, :Te].float()                       # (B,Te)
        if m.sum() == 0:
            return ego_traj.sum() * 0.0
        ar = torch.arange(B, device=ego_traj.device)
        plan = ego_traj[ar, command]                       # (B,Te,2) command's mode

        reg = (F.smooth_l1_loss(plan, ego_future, reduction='none').sum(-1) * m).sum() \
            / m.sum().clamp(min=1)
        cls = F.cross_entropy(ego_logits, command)         # learn to pick the command

        # Soft collision against agent GT futures (B=1 pipeline)
        col = ego_traj.sum() * 0.0
        if agent_future is not None and agent_future.shape[0] > 0:
            af = agent_future[:, :Te]                       # (A,Te,2)
            am = agent_mask[:, :Te].float()
            d = (plan[0].unsqueeze(0) - af).norm(dim=-1)    # (A,Te)
            valid = am * m[0].unsqueeze(0)
            col = (F.relu(self.margin - d) * valid).sum() / valid.sum().clamp(min=1)
        return self.wr * reg + self.wc * cls + self.wcol * col


# ---------------------------------------------------------------------------
# Planning metric: L2 @ {1,2,3s} + collision rate
# ---------------------------------------------------------------------------

class PlanMeter:
    """nuScenes-style open-loop planning: L2 to GT ego traj and collision rate
    at 1/2/3 s, for the GT-command best-mode ego trajectory."""

    def __init__(self, fps: int = 2, col_thresh: float = 2.0):
        self.h = [fps * 1, fps * 2, fps * 3]               # step indices for 1/2/3 s
        self.col_thresh = col_thresh
        self.l2 = {h: 0.0 for h in self.h}
        self.col = {h: 0.0 for h in self.h}
        self.n = 0

    @torch.no_grad()
    def update(self, ego_traj, ego_logits, command, ego_future, ego_mask,
               agent_future=None, agent_mask=None):
        B = ego_traj.shape[0]
        ar = torch.arange(B, device=ego_traj.device)
        plan = ego_traj[ar, command]                       # (B,Te,2) command's mode
        for b in range(B):
            self.n += 1
            for h in self.h:
                t = h - 1
                if t < plan.shape[1] and t < ego_mask.shape[1] and ego_mask[b, t]:
                    self.l2[h] += (plan[b, t] - ego_future[b, t]).norm().item()
                    if agent_future is not None and agent_future.shape[0] > 0:
                        d = (plan[b, t].unsqueeze(0) - agent_future[:, t]).norm(dim=-1)
                        hit = ((d < self.col_thresh) & agent_mask[:, t]).any().item()
                        self.col[h] += float(hit)

    def compute(self):
        n = max(self.n, 1)
        out = {}
        for h, s in zip(self.h, [1, 2, 3]):
            out[f'L2@{s}s'] = self.l2[h] / n
            out[f'col@{s}s'] = self.col[h] / n
        out['count'] = self.n
        return out
