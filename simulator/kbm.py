"""
Kinematic Bicycle Model (KBM) — ego vehicle dynamics for the closed-loop sim.
===========================================================================

This is the "Target Network" mandated by the project CLAUDE.md: a transparent,
pure-PyTorch dynamics model that turns control inputs into ego motion. It is the
piece that *closes the loop* — the Sparse4D-v3 pipeline plans a trajectory, the
controller turns that into (steering, acceleration), and this model integrates
those into a new ego pose.

Model
-----
We use the standard **rear-axle** bicycle model with a discrete forward-Euler
update. The state is carried in the GLOBAL (world) frame so the simulated path
can be compared directly against the logged nuScenes ego trajectory:

    x_{t+1}   = x   + v · cos(yaw) · dt
    y_{t+1}   = y   + v · sin(yaw) · dt
    yaw_{t+1} = yaw + (v / L) · tan(delta) · dt
    v_{t+1}   = clamp(v + a · dt, 0, v_max)

where
    (x, y)  : rear-axle position in the world frame (m)
    yaw     : heading in the world frame (rad)
    v       : forward speed (m/s)
    delta   : front-wheel steering angle (rad, +left)   -- a control input
    a       : longitudinal acceleration (m/s²)          -- a control input
    L       : wheelbase (m)

Why the rear-axle form: it is the simplest version with no slip-angle term, which
keeps the geometry transparent (CLAUDE.md style rule) while still capturing the
non-holonomic turn coupling (yaw rate scales with v and tan(delta)).

All state lives in a torch tensor on MPS per the hardware rule. The math is
plain torch ops so it stays readable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EgoState:
    """A single ego pose + speed in the world frame (plain Python floats)."""
    x: float      # world x (m)
    y: float      # world y (m)
    yaw: float    # world heading (rad)
    v: float      # forward speed (m/s)

    def as_tensor(self, device) -> torch.Tensor:
        """Pack the state into a (4,) float32 tensor on ``device``."""
        return torch.tensor([self.x, self.y, self.yaw, self.v],
                            dtype=torch.float32, device=device)


class KinematicBicycleModel:
    """Forward-integrates an ego state under (steering, acceleration) controls."""

    def __init__(self, wheelbase: float = 2.85,
                 max_steer_rad: float = 0.6,
                 v_max: float = 25.0,
                 device: str | torch.device = "mps"):
        """
        Parameters
        ----------
        wheelbase : L in the model (m). 2.85 m ≈ a typical sedan.
        max_steer_rad : hard limit on |delta| (rad); ~0.6 rad ≈ 34°.
        v_max : speed clamp (m/s) so integration can't run away.
        device : torch device; "mps" per the hardware profile.
        """
        self.L = float(wheelbase)
        self.max_steer = float(max_steer_rad)
        self.v_max = float(v_max)
        self.device = torch.device(device)
        # state vector [x, y, yaw, v] lives on-device per CLAUDE.md
        self.state = torch.zeros(4, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def reset(self, ego: EgoState):
        """Initialise the model to a known ego pose (e.g. the first GT frame)."""
        self.state = ego.as_tensor(self.device)

    @property
    def ego(self) -> EgoState:
        """Current state as a convenient :class:`EgoState` (floats, off-device)."""
        s = self.state.tolist()
        return EgoState(x=s[0], y=s[1], yaw=s[2], v=s[3])

    # ------------------------------------------------------------------
    def step(self, delta: float, accel: float, dt: float,
             substeps: int = 10) -> torch.Tensor:
        """Integrate the model over ``dt`` seconds with control held constant.

        We use a zero-order hold (the controller's output is held for the whole
        frame interval) and sub-step the integration to keep the explicit-Euler
        error small over the ~0.5 s nuScenes keyframe gap.

        Parameters
        ----------
        delta : steering angle (rad, +left); clamped to ±max_steer.
        accel : longitudinal acceleration (m/s²).
        dt : frame interval to integrate over (s).
        substeps : number of Euler sub-steps within ``dt``.

        Returns
        -------
        (substeps+1, 2) tensor: the world-frame xy path traced during this step
        (including the start point) — handy for plotting the actual rollout.
        """
        # clamp the steering command to the physical limit before integrating
        delta = float(max(-self.max_steer, min(self.max_steer, delta)))
        h = dt / substeps                                 # sub-step duration
        L, v_max = self.L, self.v_max
        path = [self.state[:2].clone()]                   # record xy along the way
        x, y, yaw, v = self.state.unbind()                # scalar tensors
        # tan(delta) and accel are constant over the step -> precompute once
        tan_delta = torch.tan(torch.tensor(delta, device=self.device))
        a = torch.tensor(accel, dtype=torch.float32, device=self.device)
        for _ in range(substeps):
            # forward-Euler bicycle update (see module docstring)
            x = x + v * torch.cos(yaw) * h
            y = y + v * torch.sin(yaw) * h
            yaw = yaw + (v / L) * tan_delta * h
            v = torch.clamp(v + a * h, min=0.0, max=v_max)  # no reverse, capped
            path.append(torch.stack([x, y]))
        self.state = torch.stack([x, y, yaw, v])          # commit the new state
        return torch.stack(path)                          # (substeps+1, 2) world xy
