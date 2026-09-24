"""Residual dynamics: a learned correction on top of the KBM/IDM prior.

    z_{t+1} = f_physics(z_t, a_t) + g_theta(z_t, a_t)

The physics stays. `g_theta` learns only what the physics gets wrong, which is a
much smaller and better-posed target than learning dynamics from scratch: it
starts at zero error rather than at chaos, it cannot violate kinematics on its
own, and a failure to learn degrades to the prior instead of to noise.

WHAT THE RESIDUAL ACTUALLY IS HERE. Ego motion is exactly KBM -- the simulator
integrates the same model the world model rolls out -- so the ego residual is
zero by construction and there is nothing to learn. The error is entirely in the
AGENTS: IDM says a vehicle brakes for the ego at a particular rate, and the
logged vehicle did something else. That difference is the target, and it is
supervised by ordinary recorded data. No counterfactuals are needed because the
question is "what did this agent do", not "what would it have done".

WHAT THIS IS DELIBERATELY NOT. It is not differentiable end-to-end into the
planner, and no gradient of a reward flows through it. A policy trained to
maximise a scalar learns to avoid whatever that scalar punishes, and this
project measured that caution is what the metrics punish -- so a policy tuned
that way would learn not to brake. This model predicts; the hard safety filter
still decides, unchanged and non-differentiable, as the final layer.

Its intended consumers are prediction-side: sharper agent forecasts for the risk
model, and a better rollout for MPC and constraint projection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                                     # pragma: no cover
    _TORCH = False

# Per-agent features, all relative to the ego so the model cannot memorise a
# world-frame position: gap along the agent's heading, lateral offset, agent
# speed, ego speed, closing speed, and the IDM decel the prior already applied.
N_FEATURES = 6
N_OUTPUTS = 2                                          # residual on (vx, vy)


@dataclass
class ResidualStats:
    """Held-out error of the prior alone vs prior + residual."""

    physics_rmse: float
    residual_rmse: float
    n: int

    @property
    def improvement(self) -> float:
        if self.physics_rmse <= 0:
            return 0.0
        return 1.0 - self.residual_rmse / self.physics_rmse

    def describe(self) -> str:
        return (f'n={self.n}  physics RMSE {self.physics_rmse:.4f} m/s  '
                f'+residual {self.residual_rmse:.4f} m/s  '
                f'improvement {self.improvement:+.1%}')


def make_mlp(hidden: int = 64):
    """Small by design: the residual is a correction, not a dynamics model.

    Capacity here buys overfitting, not accuracy -- the prior already explains
    most of the signal, and the data available is thousands of agent-steps from
    ten scenes.
    """
    if not _TORCH:
        raise RuntimeError('torch unavailable')
    return nn.Sequential(
        nn.Linear(N_FEATURES, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, N_OUTPUTS),
    )


def agent_features(agent, ego_speed: float, idm_decel: float) -> np.ndarray:
    """(6,) ego-relative features for one agent."""
    v = np.asarray(agent.vxy, dtype=np.float64)
    speed = float(np.linalg.norm(v))
    p = np.asarray(agent.xy, dtype=np.float64)
    dist = float(np.linalg.norm(p))
    if speed > 1e-6:
        fwd = v / speed
        lat = np.array([-fwd[1], fwd[0]])
        rel = -p                                       # ego seen from the agent
        gap, offset = float(rel @ fwd), float(rel @ lat)
    else:
        gap, offset = dist, 0.0
    closing = float(ego_speed) - speed
    return np.array([gap, offset, speed, float(ego_speed), closing, idm_decel],
                    dtype=np.float64)


class ResidualDynamics:
    """Physics prior plus a learned correction, with the prior always available.

    `predict` falls back to the prior alone when the model is untrained, so the
    object is safe to construct and use before any data exists -- degrading to
    the physics rather than to a random initialisation.
    """

    def __init__(self, hidden: int = 64) -> None:
        self.net = make_mlp(hidden) if _TORCH else None
        self.trained = False
        self.mu = np.zeros(N_FEATURES)
        self.sigma = np.ones(N_FEATURES)

    def _norm(self, X):
        return (X - self.mu) / np.maximum(self.sigma, 1e-6)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 300,
            lr: float = 1e-3, val_frac: float = 0.0) -> "ResidualDynamics":
        if not _TORCH:
            raise RuntimeError('torch unavailable')
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self.mu, self.sigma = X.mean(0), X.std(0)
        xt = torch.tensor(self._norm(X), dtype=torch.float32)
        yt = torch.tensor(Y, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.mse_loss(self.net(xt), yt)
            loss.backward()
            opt.step()
        self.trained = True
        return self

    def residual(self, X: np.ndarray) -> np.ndarray:
        if not (self.trained and _TORCH):
            return np.zeros((len(np.atleast_2d(X)), N_OUTPUTS))
        with torch.no_grad():
            x = torch.tensor(self._norm(np.atleast_2d(X)), dtype=torch.float32)
            return self.net(x).numpy().astype(np.float64)

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> ResidualStats:
        """Held-out RMSE of the prior alone vs prior + residual.

        The prior's error IS `Y`, since Y is defined as (actual - physics). So
        `physics_rmse` is the RMS of the targets and the comparison is honest by
        construction rather than by a separately computed baseline.
        """
        Y = np.asarray(Y, dtype=np.float64)
        phys = float(np.sqrt((Y ** 2).mean()))
        res = float(np.sqrt(((Y - self.residual(X)) ** 2).mean()))
        return ResidualStats(physics_rmse=phys, residual_rmse=res, n=len(Y))
