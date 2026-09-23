"""Control signals and action labels derived from a DiffusionDrive plan.

DiffusionDrive emits a *trajectory*, not actuator commands. To show "control signals" the
way a vehicle demo does, the trajectory has to be closed into a controller. That is what
this module does, and it is deliberately kept separate from the rendering so the numbers
on screen have one obvious place to be read and checked.

Three sources are kept distinct throughout, because conflating them would make the video
claim more than the model does:

  measured   CAN bus / IMU, i.e. what the human driver actually did in the log.
  command    `gt_ego_fut_cmd`, the high-level navigation command *fed into* the planner.
  planned    derived from `final_planning`, i.e. what DiffusionDrive asked for.

The plan -> actuator map is a pure-pursuit lateral controller plus a PI longitudinal
controller, the same pairing used in `sparse4d_vldrive/bench2drive`. Nothing here is
learned; it is the standard geometric controller, so the steering trace on screen is a
deterministic function of the plotted trajectory.

Frame convention (verified against `lidar2img` on this dataset, see bug_log entry):
`final_planning` and `gt_ego_fut_trajs` live in a frame with **+x = right, +y = forward**.
Yaw is reported left-positive (so a right turn is a negative yaw rate), which matches the
sign convention of the CAN `yaw_rate` channel.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Renault Zoe, the nuScenes ego vehicle.
WHEELBASE = 2.588       # m
EGO_WIDTH = 1.730       # m
EGO_LENGTH = 4.084      # m
PLAN_DT = 0.5           # s, one `ego_fut_ts` step
STEER_LIMIT_DEG = 470.0 # steering wheel lock, used only to scale the gauge

CMD_LIST = ["TURN RIGHT", "TURN LEFT", "GO STRAIGHT"]   # order of gt_ego_fut_cmd


def _heading(seg: np.ndarray) -> np.ndarray:
    """Left-positive heading of each path segment, relative to straight ahead."""
    return np.arctan2(-seg[..., 0], seg[..., 1])


def as_waypoints(traj: np.ndarray) -> np.ndarray:
    """Prepend the ego origin so a (T, 2) plan becomes a (T+1, 2) path."""
    return np.concatenate([np.zeros((1, 2), traj.dtype), np.asarray(traj)], axis=0)


def resample(path: np.ndarray, n: int = 120) -> np.ndarray:
    """Arc-length resample a polyline to `n` points, for smooth ribbons and lookahead."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(path[:1], n, axis=0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, path[:, 0]), np.interp(q, s, path[:, 1])], axis=1)


@dataclass
class PlannedControl:
    """What the controller asks the actuators for, given one planned trajectory."""

    speed: float            # m/s, commanded over the first plan step
    speed_profile: np.ndarray   # m/s per 0.5 s step, len 6
    accel: float            # m/s^2, commanded longitudinal acceleration
    curvature: float        # 1/m, left-positive
    yaw_rate: float         # rad/s, left-positive
    road_wheel: float       # rad, left-positive
    steer_wheel_deg: float  # deg at the steering wheel, left-positive
    throttle: float         # [0, 1]
    brake: float            # [0, 1]
    lookahead: np.ndarray   # the pure-pursuit target point, for drawing
    lookahead_dist: float


def plan_to_control(
    plan: np.ndarray,
    v_measured: float,
    steer_ratio: float,
    kp: float = 0.9,
    ki: float = 0.15,
    integral: float = 0.0,
) -> tuple[PlannedControl, float]:
    """Close a 3 s plan into actuator commands.

    Lateral is pure pursuit: fit the arc from the ego origin, tangent to straight ahead,
    through a lookahead point on the plan. Its curvature is ``2 x / d^2`` and the
    road-wheel angle follows from the bicycle model, ``delta = atan(L * kappa)``.

    Longitudinal is a PI on the speed the plan implies over its first step. The plan is
    the reference; the CAN speed is the feedback.
    """
    path = as_waypoints(plan)
    seg = np.diff(path, axis=0)
    speed_profile = np.linalg.norm(seg, axis=1) / PLAN_DT
    v_cmd = float(speed_profile[0])
    # Acceleration the plan itself implies over its first second, not a difference
    # against the measurement: this keeps it a property of the plan.
    a_cmd = float((speed_profile[1] - speed_profile[0]) / PLAN_DT) if len(speed_profile) > 1 else 0.0

    # Lookahead grows with speed, floored so a stopped ego still has a steering target.
    # It is measured along the path rather than as a chord: arc length is monotone for
    # any path, whereas distance-from-origin is not once a turn passes 90 degrees.
    dense = resample(path, 200)
    ld = float(np.clip(1.1 * max(v_measured, v_cmd), 3.0, 12.0))
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    idx = min(int(np.searchsorted(arc, ld)), len(dense) - 1)
    target = dense[idx]
    d2 = float(target @ target)
    kappa = float(-2.0 * target[0] / d2) if d2 > 1e-6 else 0.0
    delta = float(np.arctan(WHEELBASE * kappa))

    v_ref = v_cmd
    err = v_ref - v_measured
    integral = float(np.clip(integral + err * PLAN_DT, -5.0, 5.0))
    u = kp * err + ki * integral
    throttle = float(np.clip(u, 0.0, 1.0))
    brake = float(np.clip(-u, 0.0, 1.0))

    return PlannedControl(
        speed=v_cmd,
        speed_profile=speed_profile,
        accel=a_cmd,
        curvature=kappa,
        yaw_rate=kappa * max(v_measured, 1e-3),
        road_wheel=delta,
        steer_wheel_deg=float(np.degrees(delta) * steer_ratio),
        throttle=throttle,
        brake=brake,
        lookahead=target,
        lookahead_dist=float(np.linalg.norm(target)),
    ), integral


def maneuver_label(plan: np.ndarray, v_measured: float) -> tuple[str, str]:
    """Name the manoeuvre the plan encodes, as (lateral, longitudinal).

    Thresholds are on the quantities a driver would describe: total heading change over
    the 3 s horizon for lateral, and the speed change the plan implies for longitudinal.
    """
    path = as_waypoints(plan)
    seg = np.diff(path, axis=0)
    speeds = np.linalg.norm(seg, axis=1) / PLAN_DT
    psi = float(np.degrees(_heading(seg[-1])))
    lateral_off = float(-path[-1, 0])   # left-positive lateral offset at 3 s

    if psi > 14.0:
        lat = "TURN LEFT"
    elif psi < -14.0:
        lat = "TURN RIGHT"
    elif psi > 4.0 or lateral_off > 1.6:
        lat = "BEAR LEFT"
    elif psi < -4.0 or lateral_off < -1.6:
        lat = "BEAR RIGHT"
    else:
        lat = "LANE KEEP"

    dv = float(speeds[-1] - speeds[0])
    if speeds[0] < 0.4 and speeds[-1] < 0.8:
        lon = "HOLD / STOP"
    elif dv > 1.0:
        lon = "ACCELERATE"
    elif dv < -1.8:
        lon = "BRAKE"
    elif dv < -0.6:
        lon = "DECELERATE"
    else:
        lon = "MAINTAIN SPEED"
    return lat, lon


class CanBus:
    """Time-indexed nuScenes CAN channels, linearly interpolated to any timestamp.

    The keyframes the model runs on are 2 Hz; CAN is ~100 Hz. Sampling CAN at the video's
    frame rate rather than the keyframe rate is what makes the gauges move continuously
    between planner updates instead of stepping.
    """

    #: channel -> (file suffix, key, scale, component index or None)
    #:
    #: Where a quantity exists on both a 2 Hz and a ~100 Hz bus, the fast one wins:
    #: `steeranglefeedback` and `ms_imu` carry the same steering angle and yaw rate as
    #: `vehicle_monitor` (verified to match at the shared samples) at 50x the rate, which
    #: is what makes the gauges move between planner updates instead of stepping.
    CHANNELS = {
        "steer_wheel_deg": ("steeranglefeedback", "value", 180.0 / np.pi, None),
        "yaw_rate_dps": ("ms_imu", "rotation_rate", 180.0 / np.pi, 2),
        "accel_long": ("ms_imu", "linear_accel", 1.0, 0),
        "speed_kph": ("vehicle_monitor", "vehicle_speed", 1.0, None),
        "throttle": ("vehicle_monitor", "throttle", 1.0, None),
        "brake": ("vehicle_monitor", "brake", 1.0, None),
        # 2 Hz fallbacks, kept so the fast channels can be sanity-checked against them
        "steer_wheel_deg_slow": ("vehicle_monitor", "steering", 1.0, None),
        "yaw_rate_dps_slow": ("vehicle_monitor", "yaw_rate", 1.0, None),
        "pedal": ("zoe_veh_info", "pedal_cc", 1.0, None),
    }

    def __init__(self, can_root: str | Path, scene_name: str):
        self.root = Path(can_root)
        self.scene = scene_name
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._files: dict[str, list] = {}

    def _load(self, suffix: str) -> list:
        if suffix not in self._files:
            p = self.root / f"{self.scene}_{suffix}.json"
            self._files[suffix] = json.loads(p.read_text()) if p.exists() else []
        return self._files[suffix]

    def series(self, channel: str) -> tuple[np.ndarray, np.ndarray]:
        if channel not in self._cache:
            suffix, key, scale, component = self.CHANNELS[channel]
            rows = self._load(suffix)
            if not rows:
                self._cache[channel] = (np.empty(0), np.empty(0))
                return self._cache[channel]
            t = np.array([r["utime"] for r in rows], dtype=np.float64)
            raw = np.array([r[key] for r in rows], dtype=np.float64)
            v = (raw if component is None else raw[:, component]) * scale
            order = np.argsort(t)
            self._cache[channel] = (t[order], v[order])
        return self._cache[channel]

    def at(self, channel: str, utime: float, default: float = 0.0) -> float:
        t, v = self.series(channel)
        if len(t) == 0:
            return default
        return float(np.interp(utime, t, v))

    def steer_ratio(self, fallback: float = 15.0) -> float:
        """Least-squares steering-wheel : road-wheel ratio, fit from this scene's CAN.

        Rather than quote a spec-sheet number, recover it from the log: the yaw rate and
        speed give the instantaneous curvature, the bicycle model turns that into a
        road-wheel angle, and the ratio is the slope against the measured wheel angle.
        Only samples with enough speed and steer to be informative are used.
        """
        rows = self._load("vehicle_monitor")
        if not rows:
            return fallback
        wheel = np.array([r["steering"] for r in rows], dtype=np.float64)          # deg
        yaw = np.radians([r["yaw_rate"] for r in rows])                            # rad/s
        v = np.array([r["vehicle_speed"] for r in rows], dtype=np.float64) / 3.6   # m/s
        m = (v > 2.0) & (np.abs(wheel) > 20.0)
        if m.sum() < 5:
            return fallback
        delta = np.degrees(np.arctan(WHEELBASE * yaw[m] / v[m]))
        denom = float(delta @ delta)
        if denom < 1e-6:
            return fallback
        return float(np.clip(wheel[m] @ delta / denom, 8.0, 22.0))


def ego_status_fields(ego_status: np.ndarray) -> dict:
    """Unpack SparseDrive's 10-dim `ego_status`.

    Layout is [accel(3), angular velocity(3), velocity(3), steering], with accel and
    angular velocity from the IMU and steering in radians at the wheel.
    """
    e = np.asarray(ego_status, dtype=np.float64)
    return {
        "accel": e[0:3],
        "ang_vel": e[3:6],
        "vel": e[6:9],
        "speed": float(np.linalg.norm(e[6:8])),
        "steer_rad": float(e[9]),
    }
