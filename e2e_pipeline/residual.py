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


def make_mlp(hidden: int = 64, dropout: float = 0.0):
    """Small by design: the residual is a correction, not a dynamics model.

    Capacity here buys overfitting, not accuracy -- the prior already explains
    most of the signal, and the data available is thousands of agent-steps from
    ten scenes.
    """
    if not _TORCH:
        raise RuntimeError('torch unavailable')
    layers = [nn.Linear(N_FEATURES, hidden), nn.Tanh()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, N_OUTPUTS))
    return nn.Sequential(*layers)


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

    def __init__(self, hidden: int = 64, dropout: float = 0.0) -> None:
        self.net = make_mlp(hidden, dropout) if _TORCH else None
        self.trained = False
        self.mu = np.zeros(N_FEATURES)
        self.sigma = np.ones(N_FEATURES)
        self._gate_thresh = None

    def _norm(self, X):
        return (X - self.mu) / np.maximum(self.sigma, 1e-6)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 300,
            lr: float = 1e-3, weight_decay: float = 1e-2,
            shrink: float = 1e-2, val_frac: float = 0.0) -> "ResidualDynamics":
        """Fit with regularisation toward the prior.

        Two penalties, both pulling the same direction. `weight_decay` is
        ordinary L2 on the parameters; `shrink` penalises the OUTPUT magnitude,
        which is the one that matters here. A residual is a correction, so its
        right default is zero -- an unpenalised fit will happily explain
        scene-specific noise with a large correction, which is exactly what the
        first attempt did (train +6%, held-out scenes -9.5%).

        Shrinking the output is preferable to shrinking the weights alone
        because it is scale-aware: it costs the model for *claiming* a large
        correction, regardless of how the weights achieve it.
        """
        if not _TORCH:
            raise RuntimeError('torch unavailable')
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self.mu, self.sigma = X.mean(0), X.std(0)
        xt = torch.tensor(self._norm(X), dtype=torch.float32)
        yt = torch.tensor(Y, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr,
                               weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            out = self.net(xt)
            loss = nn.functional.mse_loss(out, yt) + shrink * (out ** 2).mean()
            loss.backward()
            opt.step()
        self.net.eval()          # dropout off for inference
        self.trained = True
        # Feature-space spread of the training set, for the in-distribution gate.
        self._train_z = np.abs(self._norm(X)).mean(1)
        self._gate_thresh = float(np.quantile(self._train_z, 0.95))
        return self

    def residual(self, X: np.ndarray, gate: bool = True) -> np.ndarray:
        """Correction, zeroed where the input is out of the training distribution.

        The gate is the second half of generalising. Regularisation limits how
        large a correction the model will claim; the gate limits *where* it is
        allowed to claim one at all. Outside the region the fit saw, the honest
        prediction is the physics prior, not an extrapolation from a small MLP.

        Distance is mean absolute standardised deviation across features, with
        the threshold at the training set's 95th percentile -- a crude measure,
        chosen because a 6-dimensional Gaussian fitted on thousands of points
        from ten scenes would give a confident Mahalanobis distance that is no
        better founded than this.
        """
        X = np.atleast_2d(X)
        if not (self.trained and _TORCH):
            return np.zeros((len(X), N_OUTPUTS))
        with torch.no_grad():
            z = self._norm(X)
            out = self.net(torch.tensor(z, dtype=torch.float32)).numpy().astype(np.float64)
        if gate and getattr(self, '_gate_thresh', None) is not None:
            far = np.abs(z).mean(1) > self._gate_thresh
            out[far] = 0.0
        return out

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


# --- online system identification -------------------------------------------
#
# The learned residual does not generalise across scenes, and the two standard
# remedies were measured rather than assumed:
#
#   split            unregularised   regularised   regularised + gated
#   0-4 / 5-9            -1.4%          +0.0%            +0.0%
#   5-9 / 0-4            -6.9%          -0.0%            -0.1%
#
# Regularisation strong enough to stop the overfitting is strong enough to
# erase the signal: the model shrinks to the prior and the gate then has nothing
# to gate. There is no useful middle setting on ten scenes.
#
# The reason is visible in the baseline -- physics RMSE is 0.128 m/s on scenes
# 0-4 and 0.279 on 5-9. The error is not a fixed function of the features; it
# differs by SCENE. A model fitted offline must average over that, and averaging
# over a 2.2x spread is how you get a correction that is right nowhere.
#
# So stop fitting a function and estimate a parameter instead. IDM already has
# the right structure; what it lacks is the local value of its own gain. A
# scalar estimated online from the last few observations tracks the scene it is
# actually in, needs no training set, and cannot overfit a distribution because
# it never sees one.

@dataclass
class SysIDStats:
    gain: float
    n: int
    physics_rmse: float
    adapted_rmse: float

    @property
    def improvement(self) -> float:
        if self.physics_rmse <= 0:
            return 0.0
        return 1.0 - self.adapted_rmse / self.physics_rmse

    def describe(self) -> str:
        return (f'gain {self.gain:.3f}  n={self.n}  '
                f'physics {self.physics_rmse:.4f} -> {self.adapted_rmse:.4f} m/s  '
                f'{self.improvement:+.1%}')


class OnlineIDMGain:
    """Recursive least squares on one number: IDM's deceleration gain.

    The model is `observed_delta_v ~= gain * idm_delta_v`. One parameter,
    estimated by RLS with forgetting, so it tracks the current scene rather than
    averaging over a corpus.

    Deliberately not a network:
      * it cannot overfit a training distribution, because it has none
      * it adapts to the 2.2x between-scene spread that defeated the offline fit
      * `gain = 1` recovers the unmodified prior exactly, so a bad estimate
        degrades gracefully rather than catastrophically
      * it is one readable number, so a wrong value is diagnosable

    `forgetting` under 1 discounts old observations geometrically; 0.98 keeps
    roughly the last 50 effective samples, which at 2 Hz is about half a minute
    of driving.
    """

    def __init__(self, forgetting: float = 0.98, prior_strength: float = 10.0,
                 clip: tuple[float, float] = (0.2, 3.0)) -> None:
        self.lam = float(forgetting)
        self.clip = clip
        # RLS state, initialised at gain = 1 (the unmodified prior) with a
        # strength that decides how much evidence is needed to move off it.
        self._num = float(prior_strength)
        self._den = float(prior_strength)

    @property
    def gain(self) -> float:
        g = self._num / max(self._den, 1e-9)
        return float(np.clip(g, *self.clip))

    def update(self, idm_delta_v: float, observed_delta_v: float) -> float:
        """One observation. Returns the updated gain."""
        x, y = float(idm_delta_v), float(observed_delta_v)
        if abs(x) < 1e-6:
            return self.gain                      # prior predicted no change
        self._num = self.lam * self._num + x * y
        self._den = self.lam * self._den + x * x
        return self.gain

    def apply(self, idm_delta_v):
        """Scale a prior prediction by the current estimate."""
        return self.gain * np.asarray(idm_delta_v, dtype=np.float64)
