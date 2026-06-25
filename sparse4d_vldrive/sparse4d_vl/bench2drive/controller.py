"""
Trajectory → vehicle control for the Sparse4D-v3 planner (Bench2Drive / CARLA).

The ego planner outputs future waypoints (displacements from the current ego
position, in the ego frame: x forward, y left). This converts the selected
trajectory into steer / throttle / brake:

  • lateral      : pure-pursuit on a lookahead waypoint
  • longitudinal : PID on (target speed - current speed), target speed inferred
                   from the planned first step

Pure numpy — no carla dependency, so it's unit-testable off-simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Control:
    steer: float     # [-1, 1]  (left negative in CARLA convention; see agent)
    throttle: float  # [0, 1]
    brake: float     # [0, 1]


class TrajectoryController:
    def __init__(self, wheelbase: float = 2.85, dt: float = 0.5,
                 max_steer_rad: float = 0.6, lookahead_k: float = 1.5,
                 min_lookahead: float = 3.0, kp: float = 0.4, ki: float = 0.05,
                 max_throttle: float = 0.75):
        self.L = wheelbase
        self.dt = dt
        self.max_steer_rad = max_steer_rad
        self.lk = lookahead_k
        self.min_ld = min_lookahead
        self.kp, self.ki = kp, ki
        self.max_throttle = max_throttle
        self._i = 0.0

    def reset(self):
        self._i = 0.0

    def control(self, waypoints: np.ndarray, current_speed: float) -> Control:
        """
        waypoints : (T, 2) future ego-frame displacements (metres), x fwd / y left
        current_speed : m/s
        """
        wp = np.asarray(waypoints, dtype=np.float64)

        # ---- target speed from the first planned step (displacement / dt) ----
        target_speed = float(np.linalg.norm(wp[0])) / self.dt

        # ---- longitudinal PID ----
        err = target_speed - current_speed
        self._i = float(np.clip(self._i + err * self.dt, -5.0, 5.0))
        u = self.kp * err + self.ki * self._i
        throttle = float(np.clip(u, 0.0, self.max_throttle))
        brake = float(np.clip(-u, 0.0, 1.0))

        # ---- lateral pure-pursuit ----
        ld = max(self.lk * current_speed, self.min_ld)
        # cumulative arc length along the planned path
        seg = np.linalg.norm(np.diff(np.vstack([[0, 0], wp]), axis=0), axis=1)
        cum = np.cumsum(seg)
        idx = int(np.searchsorted(cum, ld))
        idx = min(idx, len(wp) - 1)
        tx, ty = wp[idx]
        # angle to the lookahead point in the ego frame (x fwd, y left)
        alpha = math.atan2(ty, tx)
        ld_eff = max(float(np.hypot(tx, ty)), 1e-3)
        steer_rad = math.atan2(2.0 * self.L * math.sin(alpha), ld_eff)
        steer = float(np.clip(steer_rad / self.max_steer_rad, -1.0, 1.0))

        return Control(steer=steer, throttle=throttle, brake=brake)
