"""
Bird's-eye-view (BEV) renderer for the closed-loop KBM simulator.
================================================================

Each call to :func:`render_frame` draws ONE top-down frame, in the **log-ego
frame** (the LIDAR_TOP frame the perception actually ran in). Because we draw in
that frame, the logged / ground-truth ego sits at the **origin by construction**
— there is no separate "GT car" floating elsewhere; the origin *is* the GT pose
for this frame.

Screen convention
-----------------
Forward (lidar +x) points UP, left (lidar +y) points LEFT, so an ego-frame point
``(x, y)`` maps to screen ``(-y, x)`` (see :func:`_to_screen`). The plot is a
square window of ``±lim`` metres around the ego.

COLOR LEGEND  (this is the answer to "which colour is which car")
----------------------------------------------------------------
There are only TWO ego rectangles per frame, plus the obstacles:

  * GT / log ego (= the car AT THE ORIGIN)   ->  GREY dashed outline   (COLOR_GT_EGO)
        The logged human ego. It is at the origin because we render in its own
        frame. "ego car at origin" and "GT car" are the SAME rectangle.

  * Predicted / simulated ego (KBM)          ->  BLUE outline + star   (COLOR_SIM_EGO)
        Where the Kinematic Bicycle Model has driven the ego under closed-loop
        control. Its offset from the origin IS the closed-loop divergence; the
        printed "ego divergence" equals the distance from the grey box to the
        blue box (a rotation preserves that length).

  * Selected ego plan                        ->  GREEN poly-line       (COLOR_PLAN)
        The planner's intended future waypoints for the active command, drawn
        from the origin (the plan is expressed in the log-ego frame).

  * Tracked obstacles                        ->  tab20 colour BY TRACK ID
        Each detected/tracked agent's box; its best-mode motion forecast is a
        thin poly-line in the SAME colour. Untracked boxes -> COLOR_UNKNOWN grey.

Rendering uses matplotlib's Agg backend; the GIF is stitched with PIL, so no
ffmpeg is required (there is none on this Mac).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

# --- the canonical colour palette (single source of truth for the legend) ---
COLOR_GT_EGO = "#9aa0a6"   # grey dashed — GT / logged human ego, drawn at the origin
COLOR_SIM_EGO = "#4c9aff"  # blue outline — predicted closed-loop KBM (simulated) ego
COLOR_PLAN = "#39d353"     # green — selected ego plan (planner intent)
COLOR_UNKNOWN = "#888888"  # grey  — a detection with no valid track id
COLOR_COLLISION = "#ff3b30"  # red — a box overlapping the sim ego, and the ego itself

_EGO_L, _EGO_W = 4.08, 1.85   # nuScenes-ish car footprint, length × width (m)


def _to_screen(pts: np.ndarray) -> np.ndarray:
    """Map ego-frame points to screen coordinates.

    Ego frame is (x forward, y left); we want forward UP and left LEFT on screen,
    so screen_x = -y (left -> negative x is wrong; we keep left on the left by
    negating y) and screen_y = x (forward -> up).

    Parameters
    ----------
    pts : (N, 2) array of ego-frame (x, y) points.

    Returns
    -------
    (N, 2) array of screen (right, up) points.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return np.stack([-pts[:, 1], pts[:, 0]], axis=1)


def _box_corners(cx, cy, yaw, length, width) -> np.ndarray:
    """Return the four corners of an oriented box in the ego frame.

    ``length`` runs along the heading (yaw), ``width`` is perpendicular. Corners
    are ordered front-left, front-right, rear-right, rear-left so the polygon is
    convex and closes cleanly.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = length / 2.0, width / 2.0
    # corners in the box's own frame (heading along +x)
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    R = np.array([[c, -s], [s, c]])           # box -> ego rotation
    return (local @ R.T) + np.array([cx, cy])  # rotate then translate to (cx, cy)


def render_frame(out_path: str, *, boxes=None, track_ids=None,
                 trajectories=None, traj_scores=None,
                 ego_plan=None, sim_delta=None, collision_idx=None,
                 divergence=None, speed=None, control=None,
                 frame_idx=0, n_tracks=0, lim=40.0, title=""):
    """Render one BEV frame to ``out_path`` (PNG).

    All array inputs are numpy in the LOG-ego frame.

    Parameters
    ----------
    boxes : (N, 9) detection boxes [x, y, z, w, l, h, yaw, vx, vy] (lidar frame).
    track_ids : (N,) per-box track id (drives the obstacle colour); <0 = untracked.
    trajectories : (N, K, T, 2) per-box, per-mode motion forecast DISPLACEMENTS.
    traj_scores : (N, K) per-mode probabilities; the arg-max mode is drawn.
    ego_plan : (Te, 2) selected ego-plan displacements from the origin (GREEN).
    sim_delta : (dx, dy, dyaw) pose of the simulated (KBM) ego expressed in this
        frame's log-ego frame (drives the BLUE rectangle). ``hypot(dx, dy)``
        equals ``divergence``.
    collision_idx : indices of ``boxes`` overlapping the sim ego. These boxes are
        outlined in RED and the sim ego turns RED, so a collision is unmistakable.
    divergence, speed, control : scalars / Control for the HUD text.
    frame_idx, n_tracks, lim, title : cosmetics.
    """
    collision_set = set(int(j) for j in (collision_idx or []))
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_facecolor("#0e0e12")                       # dark background
    ax.grid(True, color="#2a2a33", lw=0.4)            # faint 10 m grid

    # ---- tracked obstacles + their motion forecasts (colour BY TRACK ID) ----
    cmap = plt.colormaps.get_cmap("tab20")
    if boxes is not None and len(boxes) > 0:
        for i, b in enumerate(boxes):
            # unpack the box: centre (cx, cy), size (w, l) and heading (yaw)
            cx, cy, _z, w, l, _h, yaw = b[0], b[1], b[2], b[3], b[4], b[5], b[6]
            tid = int(track_ids[i]) if track_ids is not None else i
            # stable colour per track id; grey if the box has no track
            col = cmap((tid % 20) / 20.0) if tid >= 0 else COLOR_UNKNOWN
            corners = _box_corners(cx, cy, yaw, l, w)
            # a box overlapping the sim ego is outlined thick RED so it stands out
            hit = i in collision_set
            ax.add_patch(Polygon(_to_screen(corners), closed=True,
                                 facecolor=(COLOR_COLLISION if hit else col),
                                 edgecolor=(COLOR_COLLISION if hit else "white"),
                                 alpha=(0.85 if hit else 0.55),
                                 lw=(2.2 if hit else 0.8)))
            ax.plot(*_to_screen([[cx, cy]]).T, marker=".", color=col, ms=3)
            # best-mode motion forecast, drawn from the box centre in its colour
            if trajectories is not None and len(trajectories) > i:
                tr = np.asarray(trajectories[i])          # (K, T, 2) displacements
                k = int(np.argmax(traj_scores[i])) if traj_scores is not None else 0
                fut = tr[k] + np.array([cx, cy])          # displacement -> ego xy
                sc = _to_screen(np.vstack([[cx, cy], fut]))
                ax.plot(sc[:, 0], sc[:, 1], "-", color=col, lw=1.0, alpha=0.8)

    # ---- GT / log ego footprint at the origin (GREY dashed) ----
    # This is the ground-truth human ego: it is at the origin because the whole
    # frame is drawn in its own (log-ego) coordinates.
    ego0 = _box_corners(0, 0, 0, _EGO_L, _EGO_W)
    ax.add_patch(Polygon(_to_screen(ego0), closed=True, fill=False,
                         edgecolor=COLOR_GT_EGO, ls="--", lw=1.2))

    # ---- selected ego plan (GREEN poly-line from the origin) ----
    if ego_plan is not None and len(ego_plan) > 0:
        pp = _to_screen(np.vstack([[0, 0], np.asarray(ego_plan)]))
        ax.plot(pp[:, 0], pp[:, 1], "-o", color=COLOR_PLAN, lw=1.6, ms=3)

    # ---- predicted / simulated (KBM) ego, offset by the divergence ----
    # BLUE normally; turns RED on any collision so the ego state is obvious.
    if sim_delta is not None:
        ego_col = COLOR_COLLISION if collision_set else COLOR_SIM_EGO
        dx, dy, dyaw = sim_delta                          # sim pose in log-ego frame
        egos = _box_corners(dx, dy, dyaw, _EGO_L, _EGO_W)
        ax.add_patch(Polygon(_to_screen(egos), closed=True, fill=False,
                             edgecolor=ego_col, lw=2.2 if collision_set else 1.8))
        ds = _to_screen([[dx, dy]])
        ax.plot(ds[:, 0], ds[:, 1], marker="*", color=ego_col, ms=10)

    # ---- legend so the colours are self-explanatory in every frame ----
    handles = [
        Line2D([0], [0], color=COLOR_GT_EGO, ls="--", lw=1.5,
               label="GT / log ego (origin)"),
        Line2D([0], [0], color=COLOR_SIM_EGO, lw=2.0, marker="*",
               label="predicted KBM ego"),
        Line2D([0], [0], color=COLOR_PLAN, lw=2.0, marker="o", ms=4,
               label="ego plan"),
        Line2D([0], [0], color="#cccccc", lw=1.5,
               label="tracked agent + forecast"),
        Line2D([0], [0], color=COLOR_COLLISION, lw=2.2,
               label="COLLISION (box ∩ sim ego)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              facecolor="#1b1b22", edgecolor="#2a2a33", labelcolor="white")

    # ---- heads-up display (top-left) ----
    lines = [f"frame {frame_idx:02d}   tracks {n_tracks}"]
    if speed is not None:
        lines.append(f"sim speed {speed:4.1f} m/s")
    if control is not None:
        lines.append(f"steer {math.degrees(control.delta):+5.1f}deg  "
                     f"accel {control.accel:+4.1f} m/s2")
    if divergence is not None:
        lines.append(f"ego divergence {divergence:4.2f} m")
    # collision banner: red text + count when the sim ego is hit
    if collision_set:
        lines.append(f"** COLLISION x{len(collision_set)} **")
    ax.text(-lim + 1.5, lim - 2.5, "\n".join(lines),
            color=(COLOR_COLLISION if collision_set else "white"),
            fontsize=9, va="top", family="monospace")

    if title:
        ax.set_title(title, color="white", fontsize=10)
    ax.set_xlabel("← left   (m)   right →", color="#9aa0a6", fontsize=8)
    ax.set_ylabel("← back   (m)   forward →", color="#9aa0a6", fontsize=8)
    ax.tick_params(colors="#9aa0a6", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="#0e0e12")
    plt.close(fig)                                        # free the figure


def make_gif(png_paths, gif_path, duration_ms=400):
    """Stitch saved PNG frames into an animated GIF (PIL only, no ffmpeg).

    Returns the gif path, or ``None`` if there were no frames.
    """
    from PIL import Image
    frames = [Image.open(p).convert("RGB") for p in png_paths]
    if not frames:
        return None
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    return gif_path
