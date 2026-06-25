"""
Trajectory prediction head for Sparse4D-v3 — multi-modal motion forecasting,
with QCNet-style probabilistic losses and metrics.

Sparse4D-v3 produces, per object, a temporally-fused instance feature (the
recurrent query that also drives tracking).  Motion forecasting is a small head
on top: from each instance feature predict K future trajectory modes over T
steps as Laplace distributions (loc + scale per (x,y) per step), plus a
probability per mode.  Mirrors the query-based forecasting in ViP3D / UniAD's
MotionFormer; the loss/metrics are ported from QCNet.

Head output is packed (..., K, T, 4) = [loc_x, loc_y, scale_x, scale_y], the
loc being a DISPLACEMENT from the current box centre in the current lidar frame.

Losses (QCNet):
  • regression    = Laplace-NLL on the single best (closest) mode, masked
  • classification = mixture-NLL over all modes (loc/scale detached), masked
Metrics (QCNet): minADE, minFDE, brier ((1-p_best)^2), miss-rate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryHead(nn.Module):
    """
    embed_dims   : instance-feature dim (256)
    num_modes    : K multi-modal hypotheses (6, nuScenes/AV2 default)
    future_steps : T future keyframes (12 = 6 s at 2 Hz)
    """

    def __init__(self, embed_dims: int = 256, num_modes: int = 6,
                 future_steps: int = 12, hidden: int = 256):
        super().__init__()
        self.K = num_modes
        self.T = future_steps
        self.loc = nn.Sequential(
            nn.Linear(embed_dims, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),     nn.ReLU(inplace=True),
            nn.Linear(hidden, num_modes * future_steps * 2),
        )
        self.scale = nn.Sequential(
            nn.Linear(embed_dims, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_modes * future_steps * 2),
        )
        self.cls = nn.Sequential(
            nn.Linear(embed_dims, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_modes),
        )

    def forward(self, instance_feature: torch.Tensor):
        """
        instance_feature : (B, N, D)
        Returns
        -------
        traj        : (B, N, K, T, 4)  [loc_x, loc_y, scale_x, scale_y]
        mode_logits : (B, N, K)
        """
        B, N, _ = instance_feature.shape
        loc = self.loc(instance_feature).view(B, N, self.K, self.T, 2)
        scale = F.softplus(self.scale(instance_feature)).view(B, N, self.K, self.T, 2) + 1e-3
        traj = torch.cat([loc, scale], dim=-1)            # (B, N, K, T, 4)
        mode_logits = self.cls(instance_feature)
        return traj, mode_logits


# ---------------------------------------------------------------------------
# QCNet losses (ported, adapted to a single-stage marginal head)
# ---------------------------------------------------------------------------

def _laplace_nll(loc, scale, target, eps: float = 1e-6):
    """Elementwise Laplace NLL = log(2*scale) + |target-loc|/scale."""
    scale = scale.clamp(min=eps)
    return torch.log(2 * scale) + (target - loc).abs() / scale


class MotionLoss(nn.Module):
    """
    QCNet-style trajectory loss = regression (Laplace-NLL of the best mode) +
    classification (mixture-NLL over modes).  Operates on matched queries.
    """

    def __init__(self, weight_reg: float = 1.0, weight_cls: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()
        self.weight_reg = weight_reg
        self.weight_cls = weight_cls
        self.eps = eps

    def forward(
        self,
        traj:        torch.Tensor,  # (P, K, T, 4)  matched [loc, scale]
        mode_logits: torch.Tensor,  # (P, K)
        gt_future:   torch.Tensor,  # (P, T, 2)
        gt_mask:     torch.Tensor,  # (P, T) bool
    ) -> torch.Tensor:
        if traj.shape[0] == 0:
            return traj.sum() * 0.0
        has_future = gt_mask.any(dim=1)
        if has_future.sum() == 0:
            return traj.sum() * 0.0
        traj, mode_logits = traj[has_future], mode_logits[has_future]
        gt_future, gt_mask = gt_future[has_future], gt_mask[has_future]

        P, K, T, _ = traj.shape
        loc, scale = traj[..., :2], traj[..., 2:]          # (P, K, T, 2)
        m = gt_mask.float()                                 # (P, T)

        # --- best mode by masked ADE on loc ---
        l2 = (loc - gt_future.unsqueeze(1)).norm(dim=-1)    # (P, K, T)
        ade = (l2 * m.unsqueeze(1)).sum(-1) / m.sum(-1, keepdim=True).clamp(min=1.0)
        best = ade.argmin(dim=1)                            # (P,)
        ar = torch.arange(P, device=traj.device)

        # --- regression: Laplace-NLL on the winning mode (masked) ---
        nll_best = _laplace_nll(loc[ar, best], scale[ar, best],
                                gt_future, self.eps).sum(-1)   # (P, T)
        reg = (nll_best * m).sum() / m.sum().clamp(min=1.0)

        # --- classification: mixture-NLL over all modes (loc/scale detached) ---
        nll = _laplace_nll(loc.detach(), scale.detach(),
                           gt_future.unsqueeze(1), self.eps)     # (P, K, T, 2)
        nll = (nll * m.view(P, 1, T, 1)).sum(dim=(-2, -1))       # (P, K)
        log_pi = F.log_softmax(mode_logits, dim=-1)             # (P, K)
        cls = -torch.logsumexp(log_pi - nll, dim=-1).mean()

        return self.weight_reg * reg + self.weight_cls * cls


# ---------------------------------------------------------------------------
# QCNet metrics: minADE / minFDE / brier / miss-rate
# ---------------------------------------------------------------------------

class MotionMeter:
    """Accumulate minADE / minFDE / brier / MR over batches (QCNet definitions)."""

    def __init__(self, miss_threshold: float = 2.0):
        self.miss_threshold = miss_threshold
        self.ade = self.fde = self.brier = self.mr = 0.0
        self.n = 0

    @torch.no_grad()
    def update(self, traj, mode_logits, gt_future, gt_mask):
        """traj (P,K,T,4 or P,K,T,2); gt_future (P,T,2); gt_mask (P,T)."""
        if traj.shape[0] == 0:
            return
        has = gt_mask.any(dim=1)
        if has.sum() == 0:
            return
        loc = traj[has][..., :2]
        gt, m = gt_future[has], gt_mask[has].float()
        prob = mode_logits[has].softmax(dim=-1)             # (P, K)
        P, K, T, _ = loc.shape
        ar = torch.arange(P, device=loc.device)

        l2 = (loc - gt.unsqueeze(1)).norm(dim=-1)           # (P, K, T)
        ade = (l2 * m.unsqueeze(1)).sum(-1) / m.sum(-1, keepdim=True).clamp(min=1.0)
        minade, best_ade = ade.min(dim=1)

        last = (m * torch.arange(1, T + 1, device=loc.device)).argmax(dim=-1)  # (P,)
        fde_k = l2[ar][:, :, :].gather(2, last.view(P, 1, 1).expand(P, K, 1)).squeeze(-1)  # (P, K)
        minfde, best_fde = fde_k.min(dim=1)

        brier = (1.0 - prob[ar, best_fde]).pow(2)
        miss = (minfde > self.miss_threshold).float()

        self.ade += minade.sum().item()
        self.fde += minfde.sum().item()
        self.brier += brier.sum().item()
        self.mr += miss.sum().item()
        self.n += P

    def compute(self) -> dict:
        n = max(self.n, 1)
        return {
            'minADE': self.ade / n,
            'minFDE': self.fde / n,
            'brier_minFDE': (self.fde + self.brier) / n,
            'MR': self.mr / n,
            'count': self.n,
        }
