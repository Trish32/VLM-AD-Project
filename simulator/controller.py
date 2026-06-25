"""
Trajectory → physical control for the closed-loop KBM simulator.
================================================================

The Sparse4D-v3 ego planner emits future waypoints as DISPLACEMENTS from the
current ego position, in the ego frame (x forward, y left). This controller turns
the selected trajectory into the two physical inputs the Kinematic Bicycle Model
consumes:

    • steering angle  delta (rad, +left)  — geometric pure-pursuit on a lookahead
                                            waypoint
    • acceleration    a (m/s²)            — PID on (target speed − current speed),
                                            with the target read from the plan

Everything here is ego-relative (waypoints are displacements, the lookahead is a
distance), so the controller is frame-agnostic and never needs the world pose.
It reads the *simulated* speed each step, which is what makes the overall loop
closed. Pure numpy / math — no carla, no external deps.

Sign / unit conventions
-----------------------
  * +delta steers LEFT (matches the model's y-left ego frame).
  * +a accelerates, −a decelerates; clamped to [−decel_max, +accel_max].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Control:
    """The two physical inputs the KBM consumes."""
    delta: float   # steering angle (rad, +left)
    accel: float   # longitudinal acceleration (m/s²)


class TrajectoryController:
    def __init__(self, wheelbase: float = 2.85, dt_plan: float = 0.5,
                 max_steer_rad: float = 0.6, lookahead_k: float = 1.5,
                 min_lookahead: float = 3.0, kp: float = 1.2, ki: float = 0.1,
                 accel_max: float = 3.0, decel_max: float = 6.0):
        """
        Parameters
        ----------
        wheelbase : L used in the pure-pursuit steering law (m).
        dt_plan : time between consecutive planner waypoints (s). The planner
            emits 0.5 s steps, so the first step's length / dt_plan ≈ target speed.
        max_steer_rad : steering clamp (rad), matched to the KBM's own limit.
        lookahead_k, min_lookahead : the lookahead distance is
            max(lookahead_k · speed, min_lookahead) — longer at speed, with a
            floor so the car still steers when nearly stopped.
        kp, ki : longitudinal PID gains (no derivative term; the speed signal is
            noisy and D would amplify it).
        accel_max, decel_max : acceleration clamp magnitudes (m/s²).
        """
        self.L = wheelbase
        self.dt_plan = dt_plan
        self.max_steer = max_steer_rad
        self.lk = lookahead_k
        self.min_ld = min_lookahead
        self.kp, self.ki = kp, ki
        self.accel_max, self.decel_max = accel_max, decel_max
        self._i = 0.0                       # integral accumulator (anti-windup clamped)

    def reset(self):
        """Clear the PID integral term (call once per scene)."""
        self._i = 0.0

    # ------------------------------------------------------------------
    def control(self, waypoints: np.ndarray, current_speed: float) -> Control:
        """Convert one planned trajectory into a :class:`Control`.

        Parameters
        ----------
        waypoints : (T, 2) ego-frame displacements (m), x fwd / y left. The
            planner's selected-command trajectory for this frame.
        current_speed : the SIMULATED ego speed (m/s) — using the sim speed (not
            the GT speed) is what closes the loop.
        """
        wp = np.asarray(waypoints, dtype=np.float64)

        # ---- target speed from the first planned step (displacement / dt) ----
        target_speed = float(np.linalg.norm(wp[0])) / self.dt_plan

        # ---- longitudinal PID → acceleration ----
        err = target_speed - current_speed
        # integrate the error, clamped to bound integral wind-up
        self._i = float(np.clip(self._i + err * self.dt_plan, -10.0, 10.0))
        a = self.kp * err + self.ki * self._i
        accel = float(np.clip(a, -self.decel_max, self.accel_max))

        # ---- lateral pure-pursuit → steering angle ----
        # 1) pick a lookahead distance that grows with speed (with a floor)
        ld = max(self.lk * current_speed, self.min_ld)
        # 2) walk the planned path by arc length to find the lookahead waypoint
        seg = np.linalg.norm(np.diff(np.vstack([[0, 0], wp]), axis=0), axis=1)
        cum = np.cumsum(seg)                              # cumulative arc length
        idx = min(int(np.searchsorted(cum, ld)), len(wp) - 1)
        tx, ty = wp[idx]                                  # lookahead target (ego frame)
        # 3) pure-pursuit steering law: delta = atan2(2 L sin(alpha), lookahead)
        alpha = math.atan2(ty, tx)                        # bearing to the target
        ld_eff = max(float(np.hypot(tx, ty)), 1e-3)       # actual distance to it
        delta = math.atan2(2.0 * self.L * math.sin(alpha), ld_eff)
        delta = float(np.clip(delta, -self.max_steer, self.max_steer))

        return Control(delta=delta, accel=accel)
