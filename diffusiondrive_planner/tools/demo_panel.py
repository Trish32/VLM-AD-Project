"""Control-signal and action-label panel for the DiffusionDrive demo video.

Everything here is drawn with OpenCV so it can be redrawn on every output frame. That
matters: the planner runs at 2 Hz but CAN is ~100 Hz, so redrawing the gauges per frame
is what lets the measured traces move continuously between planner updates. The BEV and
camera panels behind them are cached per keyframe.

Two colours carry the whole legend and are used nowhere else:
  orange = planned, i.e. derived from DiffusionDrive's trajectory
  cyan   = measured, i.e. CAN bus / IMU from the log
"""

from __future__ import annotations

import numpy as np

import cv2

from demo_render import (BAD, CARD, COMMAND, FONT, GOOD, LINE, MEASURED, MUTED, PANEL,
                         PLANNED, TEXT, text, text_c, text_r)
from demo_control import PLAN_DT, STEER_LIMIT_DEG

PANEL_BG = (12, 14, 18)


def rounded(img, x, y, w, h, color, r=6, thickness=-1):
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r),
                   (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(img, (cx, cy), r, color, thickness, cv2.LINE_AA)


def card(img, x, y, w, h, title):
    rounded(img, x, y, w, h, PANEL)
    text(img, title, (x + 14, y + 21), 0.44, MUTED, 1)


def steering_wheel(img, cx, cy, radius, angle_deg, color, thickness=3):
    """A rim plus three spokes, rotated by `angle_deg` (left-positive)."""
    a = np.radians(-angle_deg)     # screen y is down, so a left turn rotates CCW on screen
    cv2.circle(img, (cx, cy), radius, color, thickness, cv2.LINE_AA)
    for base in (90, 210, 330):
        t = np.radians(base) + a
        p = (int(round(cx + radius * np.cos(t))), int(round(cy - radius * np.sin(t))))
        cv2.line(img, (cx, cy), p, color, thickness, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(3, radius // 7), color, -1, cv2.LINE_AA)


def rim_tick(img, cx, cy, radius, angle_deg, color, length=12, thickness=3):
    """Mark an angle on the outside of the wheel, for the second (measured) signal."""
    a = np.radians(90.0 - angle_deg)
    c, s = np.cos(a), np.sin(a)
    p0 = (int(round(cx + radius * c)), int(round(cy - radius * s)))
    p1 = (int(round(cx + (radius + length) * c)), int(round(cy - (radius + length) * s)))
    cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def arc_gauge(img, cx, cy, radius, frac, color, span=240.0, thickness=9, bg=LINE):
    """A sweep gauge with its gap at the bottom.

    OpenCV measures ellipse angles clockwise from +x because image y points down, so a
    downward gap means starting at ``90 + (360 - span) / 2``.
    """
    start = 90.0 + (360.0 - span) / 2.0
    cv2.ellipse(img, (cx, cy), (radius, radius), 0, start, start + span, bg,
                thickness, cv2.LINE_AA)
    f = float(np.clip(frac, 0.0, 1.0))
    if f > 0:
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, start, start + span * f, color,
                    thickness, cv2.LINE_AA)


def bipolar_bar(img, x, y, w, h, value, limit, color, label="", value_text=None):
    """A bar that grows left or right of a centre tick, for signed quantities."""
    cv2.rectangle(img, (x, y), (x + w, y + h), LINE, 1, cv2.LINE_AA)
    mid = x + w // 2
    cv2.line(img, (mid, y - 2), (mid, y + h + 2), MUTED, 1, cv2.LINE_AA)
    f = float(np.clip(value / limit, -1.0, 1.0))
    end = int(mid + f * (w // 2 - 2))
    lo, hi = (min(mid, end), max(mid, end))
    if hi - lo > 1:
        cv2.rectangle(img, (lo, y + 2), (hi, y + h - 2), color, -1, cv2.LINE_AA)
    if label:
        text(img, label, (x, y - 5), 0.38, MUTED, 1)
    if value_text:
        text_r(img, value_text, x + w, y - 5, 0.38, color, 1)


def level_bar(img, x, y, w, h, frac, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), LINE, 1, cv2.LINE_AA)
    f = int(np.clip(frac, 0.0, 1.0) * (w - 4))
    if f > 0:
        cv2.rectangle(img, (x + 2, y + 2), (x + 2 + f, y + h - 2), color, -1, cv2.LINE_AA)


def trace(img, x, y, w, h, series, t_now, color, lo, hi, lw=2, dots=False):
    """One time series across the whole scene; the future is drawn faint."""
    t, v = series
    if len(t) < 2:
        return
    span = t[-1] - t[0]
    if span <= 0:
        return
    xs = x + ((t - t[0]) / span * w).astype(np.int32)
    ys = y + h - ((np.clip(v, lo, hi) - lo) / max(hi - lo, 1e-6) * h).astype(np.int32)
    pts = np.stack([xs, ys], axis=1)
    past = t <= t_now
    if past.sum() > 1:
        cv2.polylines(img, [pts[past]], False, color, lw, cv2.LINE_AA)
    if (~past).sum() > 1:
        faded = tuple(int(c * 0.32) for c in color)
        cv2.polylines(img, [pts[~past]], False, faded, 1, cv2.LINE_AA)
    if dots:
        for p, is_past in zip(pts, past):
            cv2.circle(img, tuple(int(q) for q in p), 2,
                       color if is_past else tuple(int(c * 0.32) for c in color), -1,
                       cv2.LINE_AA)


class ControlPanel:
    """The bottom block: gauges, actuator bars, action labels, and scene-long traces."""

    GAUGE_H = 296

    def __init__(self, width=1920, height=460):
        self.w, self.h = width, height

    def render(self, state) -> np.ndarray:
        img = np.empty((self.h, self.w, 3), np.uint8)
        img[:] = PANEL_BG
        self._gauges(img, 0, 0, self.w, self.GAUGE_H, state)
        self._traces(img, 0, self.GAUGE_H, self.w, self.h - self.GAUGE_H, state)
        return img

    # -- top row --------------------------------------------------

    def _gauges(self, img, x0, y0, w, h, s):
        pad, gap = 10, 10
        widths = [278, 240, 250]
        widths.append(w - 2 * pad - 3 * gap - sum(widths))
        x, y, hh = x0 + pad, y0 + pad, h - 2 * pad
        self._steering(img, x, y, widths[0], hh, s); x += widths[0] + gap
        self._speed(img, x, y, widths[1], hh, s);    x += widths[1] + gap
        self._actuators(img, x, y, widths[2], hh, s); x += widths[2] + gap
        self._actions(img, x, y, widths[3], hh, s)

    def _steering(self, img, x, y, w, h, s):
        card(img, x, y, w, h, "STEERING")
        cx, cy, r = x + w // 2, y + 102, 50
        steering_wheel(img, cx, cy, r, s["steer_plan_deg"], PLANNED, 3)
        rim_tick(img, cx, cy, r + 4, s["steer_meas_deg"], MEASURED, 12, 3)

        text_c(img, f"{s['steer_plan_deg']:+.0f}", cx, y + 200, 0.9, PLANNED, 2, FONT)
        text_c(img, "deg at the wheel, planned", cx, y + 214, 0.38, MUTED, 1)
        bipolar_bar(img, x + 18, y + 232, w - 36, 11, s["steer_plan_deg"],
                    STEER_LIMIT_DEG, PLANNED)
        bipolar_bar(img, x + 18, y + 248, w - 36, 11, s["steer_meas_deg"],
                    STEER_LIMIT_DEG, MEASURED)
        text(img, f"CAN {s['steer_meas_deg']:+.0f}", (x + 18, y + 272), 0.4, MEASURED, 1)
        text_r(img, f"road wheel {np.degrees(s['road_wheel']):+.1f}", x + w - 18,
               y + 272, 0.4, PLANNED, 1)

    def _speed(self, img, x, y, w, h, s):
        card(img, x, y, w, h, "SPEED")
        cx, cy, r = x + w // 2, y + 112, 68
        v_max = 40.0     # km/h, full scale
        arc_gauge(img, cx, cy, r, s["speed_meas_kph"] / v_max, MEASURED, thickness=11)
        arc_gauge(img, cx, cy, r - 15, s["speed_plan_kph"] / v_max, PLANNED, thickness=7)
        text_c(img, f"{s['speed_meas_kph']:.1f}", cx, cy + 4, 1.0, TEXT, 2, FONT)
        text_c(img, "km/h  CAN", cx, cy + 25, 0.4, MUTED, 1)
        text_c(img, f"plan {s['speed_plan_kph']:.1f} km/h", cx, y + 208, 0.46, PLANNED, 1)

        bipolar_bar(img, x + 18, y + 232, w - 36, 11, s["accel_plan"], 4.0, PLANNED,
                    "longitudinal accel  m/s2")
        bipolar_bar(img, x + 18, y + 248, w - 36, 11, s["accel_meas"], 4.0, MEASURED)
        text(img, f"plan {s['accel_plan']:+.2f}", (x + 18, y + 272), 0.4, PLANNED, 1)
        text_r(img, f"IMU {s['accel_meas']:+.2f}", x + w - 18, y + 272, 0.4, MEASURED, 1)

    def _actuators(self, img, x, y, w, h, s):
        card(img, x, y, w, h, "ACTUATORS")
        rows = [
            ("throttle  cmd", s["throttle_plan"], PLANNED, f"{s['throttle_plan']*100:.0f}%"),
            ("brake  cmd", s["brake_plan"], PLANNED, f"{s['brake_plan']*100:.0f}%"),
            ("throttle  CAN", s["throttle_meas"], MEASURED, f"{s['throttle_raw']:.0f}"),
        ]
        yy = y + 44
        for name, val, color, label in rows:
            text(img, name, (x + 16, yy), 0.4, MUTED, 1)
            text_r(img, label, x + w - 16, yy, 0.4, color, 1)
            level_bar(img, x + 16, yy + 7, w - 32, 10, val, color)
            yy += 38

        bipolar_bar(img, x + 16, y + 176, w - 32, 11, np.degrees(s["yaw_plan"]), 40.0,
                    PLANNED, "yaw rate  deg/s")
        bipolar_bar(img, x + 16, y + 192, w - 32, 11, s["yaw_meas_dps"], 40.0, MEASURED)
        text(img, f"plan {np.degrees(s['yaw_plan']):+.1f}", (x + 16, y + 216), 0.4,
             PLANNED, 1)
        text_r(img, f"CAN {s['yaw_meas_dps']:+.1f}", x + w - 16, y + 216, 0.4, MEASURED, 1)

        text(img, "path curvature", (x + 16, y + 244), 0.4, MUTED, 1)
        text_r(img, f"{s['curvature']:+.4f} 1/m", x + w - 16, y + 244, 0.4, PLANNED, 1)
        text(img, "lookahead", (x + 16, y + 268), 0.4, MUTED, 1)
        text_r(img, f"{s['lookahead_dist']:.1f} m", x + w - 16, y + 268, 0.4, PLANNED, 1)

    def _actions(self, img, x, y, w, h, s):
        card(img, x, y, w, h, "ACTION")
        half = (w - 38) // 2

        text(img, "NAV COMMAND", (x + 16, y + 44), 0.4, MUTED, 1)
        text(img, "planner input, from the route", (x + 140, y + 44), 0.38, LINE, 1)
        rounded(img, x + 16, y + 52, w - 32, 42, CARD, 6)
        text_c(img, s["nav_command"], x + w // 2, y + 81, 0.8, COMMAND, 2, FONT)

        text(img, "MANOEUVRE", (x + 16, y + 122), 0.4, MUTED, 1)
        text(img, "read off DiffusionDrive's trajectory", (x + 130, y + 122), 0.38, LINE, 1)
        rounded(img, x + 16, y + 130, half, 42, CARD, 6)
        rounded(img, x + 28 + half, y + 130, half, 42, CARD, 6)
        text_c(img, s["lateral"], x + 16 + half // 2, y + 158, 0.62, PLANNED, 1, FONT)
        text_c(img, s["longitudinal"], x + 28 + half + half // 2, y + 158, 0.62,
               PLANNED, 1, FONT)

        # bottom row: mode confidence on the left, the planned speed profile on the right
        by = y + 196
        bh = 46
        self._modes(img, x + 16, by, half, bh, s)
        self._profile(img, x + 28 + half, by, half, bh, s)

    def _modes(self, img, x, y, w, h, s):
        l2 = s["l2"]
        text(img, "MODE CONFIDENCE", (x, y), 0.4, MUTED, 1)
        if np.isnan(l2[3]):
            text_r(img, "L2  logged future ends", x + w, y, 0.38, LINE, 1)
        else:
            text_r(img, f"L2 {l2[1]:.2f} / {l2[2]:.2f} / {l2[3]:.2f} m", x + w, y, 0.38,
                   GOOD if l2[3] < 1.0 else MUTED, 1)
        p = np.asarray(s["mode_probs"])
        bw = (w - 5 * 5) // 6
        for k, prob in enumerate(p):
            bx = x + k * (bw + 5)
            top = y + 8
            cv2.rectangle(img, (bx, top), (bx + bw, top + h), LINE, 1, cv2.LINE_AA)
            fh = int(prob / max(p.max(), 1e-6) * (h - 4))
            color = PLANNED if k == s["best_mode"] else (96, 116, 92)
            if fh > 0:
                cv2.rectangle(img, (bx + 2, top + h - 2 - fh), (bx + bw - 2, top + h - 2),
                              color, -1, cv2.LINE_AA)
            text_c(img, f"{prob * 100:.0f}", bx + bw // 2, top + h + 15, 0.36,
                   TEXT if k == s["best_mode"] else MUTED, 1)

    def _profile(self, img, x, y, w, h, s):
        text(img, "PLANNED SPEED PROFILE", (x, y), 0.4, MUTED, 1)
        text_r(img, "0.5 - 3.0 s horizon", x + w, y, 0.38, LINE, 1)
        prof = np.asarray(s["speed_profile"]) * 3.6
        top = y + 8
        # Fixed full scale, shared with the speed gauge, so the bars mean the same thing
        # from frame to frame. Auto-scaling would flatten every profile to full height.
        v_max = 40.0
        bw = (w - 5 * 5) // 6
        for k, v in enumerate(prof):
            bx = x + k * (bw + 5)
            cv2.rectangle(img, (bx, top), (bx + bw, top + h), LINE, 1, cv2.LINE_AA)
            fh = int(np.clip(v / v_max, 0, 1) * (h - 4))
            if fh > 0:
                cv2.rectangle(img, (bx + 2, top + h - 2 - fh), (bx + bw - 2, top + h - 2),
                              PLANNED, -1, cv2.LINE_AA)
            text_c(img, f"{(k + 1) * PLAN_DT:.1f}", bx + bw // 2, top + h + 15, 0.36,
                   MUTED, 1)
        # Current measured speed, dashed and drawn last: when it sits flush with the bar
        # tops the plan is holding speed, which is the common case and must stay readable.
        yb = top + h - 2 - int(s["speed_meas_kph"] / v_max * (h - 4))
        for dx in range(0, w, 10):
            cv2.line(img, (x + dx, yb), (x + min(dx + 5, w), yb), MEASURED, 1, cv2.LINE_AA)

    # -- traces ---------------------------------------------------

    def _traces(self, img, x0, y0, w, h, s):
        pad, gap = 10, 10
        specs = [
            ("SPEED  km/h", "speed", 0.0, 30.0),
            ("STEERING WHEEL  deg", "steer", -300.0, 300.0),
            ("YAW RATE  deg/s", "yaw", -45.0, 45.0),
            ("L2 vs LOG @3s  m", "l2", 0.0, 4.0),
        ]
        cw = (w - 2 * pad - (len(specs) - 1) * gap) // len(specs)
        for k, (title, key, lo, hi) in enumerate(specs):
            x = x0 + pad + k * (cw + gap)
            rounded(img, x, y0, cw, h - pad, PANEL)
            text(img, title, (x + 12, y0 + 19), 0.4, MUTED, 1)
            bx, by = x + 12, y0 + 27
            bw, bh = cw - 24, h - pad - 38
            if lo < 0 < hi:
                zy = by + bh - int((0 - lo) / (hi - lo) * bh)
                cv2.line(img, (bx, zy), (bx + bw, zy), LINE, 1, cv2.LINE_AA)
            tr = s["traces"][key]
            if "meas" in tr:
                trace(img, bx, by, bw, bh, tr["meas"], s["t_now"], MEASURED, lo, hi)
            trace(img, bx, by, bw, bh, tr["plan"], s["t_now"], PLANNED, lo, hi,
                  lw=2, dots=True)
            ref = tr.get("meas", tr["plan"])[0]
            if len(ref) > 1:
                f = float(np.clip((s["t_now"] - ref[0]) / (ref[-1] - ref[0]), 0, 1))
                px = bx + int(f * bw)
                cv2.line(img, (px, by), (px, by + bh), (110, 118, 128), 1, cv2.LINE_AA)
            text_r(img, f"{hi:.0f}", bx + bw, by + 10, 0.32, MUTED, 1)
            text_r(img, f"{lo:.0f}", bx + bw, by + bh, 0.32, MUTED, 1)


def title_bar(img, w, h, scene, frame, n_frames, timestamp, subtitle, scene_l2):
    img[:] = (9, 11, 14)
    text(img, "DiffusionDrive", (20, 39), 0.86, TEXT, 2, FONT)
    text(img, subtitle, (236, 39), 0.48, MUTED, 1)

    for k, (name, color) in enumerate((("planned", PLANNED), ("measured", MEASURED))):
        x = 760 + k * 140
        cv2.line(img, (x, 34), (x + 26, 34), color, 3, cv2.LINE_AA)
        text(img, name, (x + 34, 39), 0.46, color, 1)

    text_r(img, f"nuScenes mini_val   {scene}   keyframe {frame + 1}/{n_frames}"
                f"   t = {timestamp:5.1f} s", w - 20, 26, 0.48, TEXT, 1)
    text_r(img, f"scene L2  1s {scene_l2[1]:.3f}   2s {scene_l2[2]:.3f}   "
                f"3s {scene_l2[3]:.3f} m", w - 20, 48, 0.42,
           GOOD if scene_l2[3] < 1.0 else MUTED, 1)
    cv2.line(img, (0, h - 1), (w, h - 1), LINE, 1)
    return img
