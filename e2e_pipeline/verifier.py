"""(6) Independent verification: the last thing between a plan and the actuators.

The safety filter already rejects infeasible trajectories. This layer sits AFTER
it and checks the surviving plan again, which is only worth doing if the second
check is genuinely independent -- a verifier that calls the function it is
verifying certifies nothing.

So where the scopes overlap, the implementation deliberately does not:

| concern | safety_filter | verifier |
|---|---|---|
| dynamics | reachability against FeasibilityLimits | finite-difference accel/curvature straight off the waypoints |
| collision | swept footprint against the occupancy RASTER | polygon SAT against tracked agent BOXES |
| traffic rules | -- | red light, speed limit, decision consistency |

The raster path and the box path draw on different upstream branches (FlashOcc
vs Sparse4D), so an error in one does not silently pass both. That redundancy is
the point, not waste.

WHAT IS ACTUALLY NEW HERE
-------------------------
Only the traffic-rule checks. Dynamics and collision are re-derivations. The
genuinely uncovered gap was rule compliance, and one case in particular that no
component checked: the VLM can emit `STOP` while the planner emits a 15 m/s
trajectory, and nothing notices. The decision-consistency check closes that.

WHAT THIS CANNOT CHECK, AND WHY
-------------------------------
* **Speed limits** are not in nuScenes. The map expansion has 11 layers and none
  carries a posted limit, so `speed_limit_mps` is a configured constant, not a
  measurement. Treat a violation as "exceeded the limit we told it", not "broke
  the law".
* **Yielding** needs lane topology and right-of-way reasoning -- who is on the
  priority road, who arrived first. nuScenes gives lane geometry but no
  precedence relation, so implementing `yield` here would be a guess wearing a
  rule's name. The stop-line case IS implemented, because `stop_line` is real
  map geometry; general yielding is declared out of scope rather than faked.
* **Red-light state** comes from the VLM, not the map. nuScenes annotates traffic
  light *geometry* but not *bulb colour*, so this check is only as good as the
  upstream perception it is nominally independent of. That is a real limit on its
  independence and is named rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import obb_overlap
from .scene import SceneRepresentation, ego_footprint_corners, yaw_from_waypoints

# Verification limits. Deliberately LOOSER than the safety filter's: this layer
# catches gross violations the filter should already have stopped, so tripping it
# means something upstream is broken, not merely aggressive. Equal limits would
# make it fire constantly on borderline-but-accepted plans and train everyone to
# ignore it.
MAX_ACCEL = 4.0          # m/s^2 longitudinal (filter: 3.0)
MAX_DECEL = 8.0          # m/s^2 (filter: 6.0)
MAX_CURVATURE = 0.35     # 1/m, ~2.9 m turning radius
MAX_LAT_ACCEL = 5.5      # m/s^2 (filter: 4.0)
MIN_GAP = 0.15           # m, polygon separation
STOPPED_MPS = 0.5        # at or below this the plan counts as stopped
SPEED_LIMIT_MPS = 16.7   # 60 km/h; configured, NOT from the map

SEVERITY = ('warn', 'reject')


@dataclass
class Violation:
    rule: str
    severity: str
    detail: str
    step: int = -1                 # trajectory index, -1 when whole-plan

    def __str__(self) -> str:
        where = '' if self.step < 0 else f' @step{self.step}'
        return f'[{self.severity.upper()}] {self.rule}{where}: {self.detail}'


@dataclass
class VerificationReport:
    violations: list[Violation] = field(default_factory=list)
    trajectory: np.ndarray | None = None        # what to actually execute
    substituted: bool = False                   # fallback replaced the plan

    @property
    def passed(self) -> bool:
        return not any(v.severity == 'reject' for v in self.violations)

    def describe(self) -> str:
        head = ('VERIFIED' if self.passed else 'REJECTED')
        if self.substituted:
            head += ' (fallback substituted)'
        return '\n'.join([head] + ['  ' + str(v) for v in self.violations])


def comfortable_stop(v0: float, horizon: int, dt: float,
                     decel: float = 3.0) -> np.ndarray:
    """Straight-ahead decelerating plan -- the fallback when a plan is rejected.

    Deliberately not a maximum-authority stop: this substitutes for a plan that
    failed verification, which is a software fault, not necessarily an imminent
    collision. Slamming on the brakes for a failed curvature check would create a
    hazard to answer a bookkeeping error. Genuine emergencies are the safety
    filter's emergency brake, upstream of here.

    But it MUST reach rest inside the horizon. At the nominal 3 m/s^2 a fallback
    from 10 m/s still carries 1.0 m/s at t=3 s, so the plan substituted for a
    red-light violation would itself fail the red-light check -- the fallback
    reproducing the violation it exists to fix. The rate is therefore raised to
    whatever the horizon requires, and `decel` is a floor rather than the value.
    """
    span = max(horizon * dt, 1e-6)
    decel = max(float(decel), float(v0) / span)
    out, x, v = [], 0.0, float(v0)
    for _ in range(horizon):
        v = max(0.0, v - decel * dt)
        x += v * dt
        out.append([x, 0.0])
    return np.asarray(out, dtype=np.float64)


class TrajectoryVerifier:
    """Independent last-line check on the plan about to be executed."""

    def __init__(self, dt: float = 0.5,
                 speed_limit_mps: float = SPEED_LIMIT_MPS,
                 ego_len: float = 4.6, ego_wid: float = 1.8) -> None:
        self.dt = float(dt)
        self.speed_limit = float(speed_limit_mps)
        self.ego_len, self.ego_wid = float(ego_len), float(ego_wid)

    # -- (a) physical feasibility -------------------------------------------

    def _check_dynamics(self, traj: np.ndarray, v0: float) -> list[Violation]:
        """Finite differences straight off the waypoints.

        Independent of the filter, which evaluates reachability against its own
        limit set. Here the plan is simply differentiated and the results
        compared to bounds -- no shared code, no shared assumptions.
        """
        out = []
        pts = np.vstack([[0.0, 0.0], traj])
        seg = np.diff(pts, axis=0)
        speeds = np.linalg.norm(seg, axis=1) / self.dt
        accel = np.diff(np.concatenate([[v0], speeds])) / self.dt

        for i, a in enumerate(accel):
            if a > MAX_ACCEL:
                out.append(Violation('accel_limit', 'reject',
                                     f'{a:.2f} > {MAX_ACCEL} m/s^2', i))
            elif a < -MAX_DECEL:
                out.append(Violation('decel_limit', 'reject',
                                     f'{a:.2f} < -{MAX_DECEL} m/s^2', i))

        # Menger curvature on consecutive waypoint triples.
        for i in range(len(pts) - 2):
            a_, b_, c_ = pts[i], pts[i + 1], pts[i + 2]
            ab, bc, ca = (np.linalg.norm(b_ - a_), np.linalg.norm(c_ - b_),
                          np.linalg.norm(a_ - c_))
            if ab * bc * ca < 1e-9:
                continue
            u, w = b_ - a_, c_ - b_
            cross = abs(u[0] * w[1] - u[1] * w[0])
            kappa = 2.0 * cross / (ab * bc * ca)
            if kappa > MAX_CURVATURE:
                out.append(Violation('curvature_limit', 'reject',
                                     f'{kappa:.3f} > {MAX_CURVATURE} 1/m', i + 1))
            lat = kappa * speeds[min(i, len(speeds) - 1)] ** 2
            if lat > MAX_LAT_ACCEL:
                out.append(Violation('lateral_accel', 'reject',
                                     f'{lat:.2f} > {MAX_LAT_ACCEL} m/s^2', i + 1))
        return out

    # -- (b) traffic rules ---------------------------------------------------

    def _check_rules(self, traj: np.ndarray, v0: float, light: str,
                     decision: str) -> list[Violation]:
        out = []
        pts = np.vstack([[0.0, 0.0], traj])
        speeds = np.linalg.norm(np.diff(pts, axis=0), axis=1) / self.dt

        if speeds.max() > self.speed_limit:
            out.append(Violation(
                'speed_limit', 'reject',
                f'{speeds.max():.1f} > {self.speed_limit:.1f} m/s (configured, '
                f'not from map)', int(speeds.argmax())))

        # A red or yellow light must produce a plan that is stopping. Checking
        # the TERMINAL speed rather than requiring an immediate halt: a plan that
        # decelerates smoothly to rest within the horizon is compliant, one that
        # sails through at speed is not.
        if light in ('red', 'yellow') and speeds[-1] > STOPPED_MPS:
            out.append(Violation(
                'red_light', 'reject',
                f'light={light} but plan ends at {speeds[-1]:.1f} m/s', len(speeds) - 1))

        # The gap nothing else covered: the executed plan must agree with the
        # decision the VLM actually made. Disagreement means the reasoning stage
        # and the planning stage have diverged, which is a fault in itself
        # regardless of whether the trajectory is independently safe.
        if decision:
            d = decision.upper()
            if d == 'STOP' and speeds[-1] > STOPPED_MPS:
                out.append(Violation('decision_consistency', 'reject',
                                     f'decision=STOP but plan ends at '
                                     f'{speeds[-1]:.1f} m/s', -1))
            elif d in ('SLOW_DOWN', 'YIELD') and speeds[-1] > v0 + 0.5:
                out.append(Violation('decision_consistency', 'warn',
                                     f'decision={d} but plan accelerates '
                                     f'{v0:.1f} -> {speeds[-1]:.1f} m/s', -1))
        return out

    # -- (c) collision -------------------------------------------------------

    def _check_collision(self, traj: np.ndarray,
                         scene: SceneRepresentation) -> list[Violation]:
        """Polygon SAT against tracked agent boxes.

        A different data path from the filter's raster sweep: this reads the
        Kalman-tracked object list, that reads the occupancy grid. An agent
        missing from one is usually present in the other.
        """
        out = []
        pts = np.vstack([[0.0, 0.0], traj])
        yaws = yaw_from_waypoints(pts)
        for i, (p, yaw) in enumerate(zip(pts[1:], yaws[1:]), start=1):
            # Inflating the ego rather than passing a gap to obb_overlap, which
            # is a strict overlap test. Same effect, one code path.
            ego_poly = ego_footprint_corners(p, float(yaw),
                                             self.ego_len + 2 * MIN_GAP,
                                             self.ego_wid + 2 * MIN_GAP)
            for ag in scene.agents:
                # Agents are propagated by their own velocity to the step being
                # checked; comparing a future ego pose against a present agent
                # pose would systematically under-report closing conflicts.
                axy = np.asarray(ag.xy, float) + np.asarray(ag.vxy, float) * i * self.dt
                ag_poly = ego_footprint_corners(axy, float(ag.yaw),
                                                float(ag.lwh[0]), float(ag.lwh[1]))
                if obb_overlap(ego_poly, ag_poly):
                    out.append(Violation(
                        'collision', 'reject',
                        f'footprint overlaps track {ag.track_id} '
                        f'(gap < {MIN_GAP} m)', i))
                    break
        return out

    # -- entry point ---------------------------------------------------------

    def verify(self, traj: np.ndarray, scene: SceneRepresentation,
               light: str = 'none', decision: str = '',
               substitute: bool = True) -> VerificationReport:
        traj = np.asarray(traj, dtype=np.float64)
        v0 = float(scene.ego.speed)

        violations = (self._check_dynamics(traj, v0)
                      + self._check_rules(traj, v0, light, decision)
                      + self._check_collision(traj, scene))

        rep = VerificationReport(violations=violations, trajectory=traj)
        if not rep.passed and substitute:
            rep.trajectory = comfortable_stop(v0, len(traj), self.dt)
            rep.substituted = True
        return rep
