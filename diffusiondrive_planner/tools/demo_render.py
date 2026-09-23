"""Drawing primitives for the DiffusionDrive demo video.

Three renderers, deliberately split by cost:

  `BEVPanel`    matplotlib, ~150 ms/frame, redrawn once per 2 Hz keyframe.
  `CameraPanel` OpenCV, redrawn once per keyframe.
  `ControlPanel` OpenCV only, redrawn every output frame so the gauges track the
                ~100 Hz CAN signal instead of stepping at the planner rate.

Colour choices follow the paper's own legend so the video reads as DiffusionDrive:
`autumn` (orange -> yellow, t=0 -> t=3s) for the top-1 scoring trajectory, `winter`
(blue -> green) for the remaining scored modes, and mode confidence as opacity.
"""

from __future__ import annotations

import numpy as np

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners

from demo_control import resample

# ---------------------------------------------------------------- theme

# All theme colours are literal BGR, the order OpenCV actually writes.
BG        = (12, 14, 18)            # near-black with a warm cast
PANEL     = (20, 23, 27)
CARD      = (28, 32, 38)
LINE      = (58, 62, 70)
TEXT      = (232, 238, 243)
MUTED     = (139, 148, 158)
PLANNED   = (60, 150, 255)          # orange - DiffusionDrive output
MEASURED  = (232, 206, 92)          # cyan   - CAN / IMU
COMMAND   = (120, 196, 120)         # green  - navigation command (an input, not output)
GOOD      = (90, 200, 110)
BAD       = (80, 90, 240)

BEV_BG        = "#14120e"
BEV_GRID      = "#33302c"
MAP_COLORS    = {0: "#5fb0ff", 1: "#e8c86a", 2: "#7f8b97"}   # ped_crossing, divider, boundary
MAP_NAMES     = {0: "ped crossing", 1: "lane divider", 2: "road boundary"}
CLASS_NAMES = ["car", "truck", "constr", "bus", "trailer", "barrier",
               "motorcycle", "bicycle", "pedestrian", "cone"]
VEHICLE_IDS = {0, 1, 2, 3, 4, 6, 7}

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX

AGENT_PALETTE = np.asarray([
    [255, 179, 0], [166, 189, 215], [255, 104, 0], [193, 0, 32], [206, 162, 98],
    [0, 125, 52], [246, 118, 142], [0, 83, 138], [255, 122, 92], [131, 112, 200],
    [255, 142, 0], [179, 40, 81], [244, 200, 0], [147, 170, 0], [112, 224, 255],
    [70, 184, 160], [153, 110, 255], [113, 255, 0], [255, 0, 163], [0, 204, 255],
]) / 255.0


def text(img, s, org, scale=0.5, color=TEXT, thick=1, font=FONT_S):
    cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)


def text_r(img, s, right_x, y, scale=0.5, color=TEXT, thick=1, font=FONT_S):
    (w, _), _ = cv2.getTextSize(s, font, scale, thick)
    cv2.putText(img, s, (right_x - w, y), font, scale, color, thick, cv2.LINE_AA)


def text_c(img, s, cx, y, scale=0.5, color=TEXT, thick=1, font=FONT_S):
    (w, _), _ = cv2.getTextSize(s, font, scale, thick)
    cv2.putText(img, s, (cx - w // 2, y), font, scale, color, thick, cv2.LINE_AA)


def project(points_3d: np.ndarray, lidar2img: np.ndarray):
    """Project (N,3) lidar-frame points to (N,2) pixels plus their camera depth."""
    p = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1) @ lidar2img.T
    depth = p[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    return np.stack([p[:, 0] / safe, p[:, 1] / safe], axis=1), depth


# ---------------------------------------------------------------- BEV

class BEVPanel:
    """Top-down view: online map, tracked agents, their forecasts, and the plan.

    The plan is drawn in three layers so the truncated-diffusion story is visible in one
    glance: the command-selected *anchors* the denoiser starts from, the six *denoised
    modes* it produces, and the *top-1* mode the controller actually follows.
    """

    def __init__(self, width=740, height=1020, x_range=21.8, y_range=30.0, dpi=100):
        self.w, self.h = width, height
        self.xr, self.yr = x_range, y_range
        self.dpi = dpi
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.fig.patch.set_facecolor(BEV_BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])

    def _reset(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(BEV_BG)
        ax.set_xlim(-self.xr, self.xr)
        ax.set_ylim(-self.yr, self.yr)
        ax.set_aspect("equal")
        ax.axis("off")
        for r in (10, 20, 30):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=BEV_GRID, lw=0.8,
                                    ls=(0, (4, 5)), zorder=0))
            if r < self.yr:
                ax.text(0.6, r - 1.4, f"{r} m", color=BEV_GRID, fontsize=7, zorder=0)
        ax.axhline(0, color=BEV_GRID, lw=0.6, zorder=0)
        ax.axvline(0, color=BEV_GRID, lw=0.6, zorder=0)

    # -- layers ---------------------------------------------------

    def draw_map(self, result, thresh=0.3):
        for i in range(len(result["scores"])):
            if result["scores"][i] < thresh:
                continue
            label = int(result["labels"][i])
            pts = np.asarray(result["vectors"][i])
            ls = (0, (6, 6)) if label == 1 else "-"
            self.ax.plot(pts[:, 0], pts[:, 1], color=MAP_COLORS.get(label, "#888"),
                         lw=2.0, ls=ls, zorder=1, alpha=0.85)

    def draw_agents(self, result, thresh=0.3, motion_top_k=1):
        boxes = result["boxes_3d"].numpy()
        scores = result["scores_3d"].numpy()
        labels = result["labels_3d"].numpy()
        ids = result["instance_ids"].numpy()
        keep = np.where(scores >= thresh)[0]
        if len(keep) == 0:
            return 0
        corners = box3d_to_corners(boxes[keep])
        trajs = result["trajs_3d"].numpy()[keep]
        tscore = result["trajs_score"].numpy()[keep]
        for n, i in enumerate(keep):
            color = AGENT_PALETTE[int(ids[i]) % len(AGENT_PALETTE)]
            ring = corners[n][[0, 3, 7, 4, 0]]
            self.ax.add_patch(MplPolygon(ring[:, :2], closed=True, fc=color, ec=color,
                                         alpha=0.22, lw=0, zorder=2))
            self.ax.plot(ring[:, 0], ring[:, 1], color=color, lw=1.6, zorder=3)
            nose = ring[1:3].mean(axis=0)
            mid = ring[:4].mean(axis=0)
            self.ax.plot([mid[0], nose[0]], [mid[1], nose[1]], color=color, lw=1.6, zorder=3)

            order = np.argsort(-tscore[n])[:motion_top_k]
            for rank, m in enumerate(order):
                traj = np.concatenate([boxes[i, :2][None], trajs[n, m]], axis=0)
                w = float(np.exp(tscore[n, m] - tscore[n].max()))
                self._ribbon(traj, "winter", w * (1.0 if rank == 0 else 0.4),
                             size=9 if int(labels[i]) in VEHICLE_IDS else 4, zorder=2)
        return len(keep)

    def draw_anchors(self, anchors):
        """The six command-selected kmeans anchors the diffusion is seeded from."""
        for a in anchors:
            p = np.concatenate([np.zeros((1, 2)), a], axis=0)
            self.ax.plot(p[:, 0], p[:, 1], color="#6f6a63", lw=1.0, ls=(0, (2, 3)),
                         zorder=4, alpha=0.75)

    def draw_modes(self, modes, scores, best):
        """Denoised modes, opacity by softmax confidence; top-1 in the paper's autumn."""
        p = np.exp(scores - scores.max())
        p = p / p.sum()
        for m in np.argsort(p):
            if m == best:
                continue
            traj = np.concatenate([np.zeros((1, 2)), modes[m]], axis=0)
            self._ribbon(traj, "winter", float(np.clip(p[m] / p.max(), 0.12, 1.0)),
                         size=14, zorder=5)
        traj = np.concatenate([np.zeros((1, 2)), modes[best]], axis=0)
        self._ribbon(traj, "autumn", 1.0, size=34, zorder=7)

    def draw_gt(self, gt_deltas, masks=None):
        if masks is not None and not bool(np.asarray(masks).astype(bool)[0]):
            return
        traj = np.asarray(gt_deltas).copy()
        traj[np.abs(traj) < 0.01] = 0.0
        traj = np.concatenate([np.zeros((1, 2)), traj.cumsum(axis=0)], axis=0)
        self.ax.plot(traj[:, 0], traj[:, 1], color="#f2ede6", lw=1.4, ls=(0, (5, 4)),
                     zorder=6, alpha=0.8)
        self.ax.scatter(traj[1:, 0], traj[1:, 1], s=12, facecolors="none",
                        edgecolors="#f2ede6", lw=1.0, zorder=6, alpha=0.8)

    def draw_lookahead(self, point):
        self.ax.scatter([point[0]], [point[1]], s=70, marker="P", color="#ffffff",
                        zorder=9, lw=0)

    def draw_ego(self, length=4.084, width=1.730):
        rear = -1.0     # rear axle sits behind the sensor origin
        rect = np.array([[-width / 2, rear], [width / 2, rear],
                         [width / 2, rear + length], [-width / 2, rear + length]])
        self.ax.add_patch(MplPolygon(rect, closed=True, fc="#f2ede6", ec="#ffffff",
                                     lw=1.2, alpha=0.95, zorder=8))
        self.ax.plot([0, 0], [rear + length, rear + length + 1.1], color="#ffffff",
                     lw=1.4, zorder=8)

    def _ribbon(self, path, cmap, weight, size=20, zorder=5, steps=20):
        n = (len(path) - 1) * steps + 1
        colors = matplotlib.colormaps[cmap](np.linspace(0, 1, n))[:, :3]
        colors = colors * weight + (1 - weight) * np.full_like(colors, 0.08)
        xy = np.zeros((n, 2))
        for i in range(n - 1):
            k = i // steps
            xy[i] = path[k] + (i / steps - k) * (path[k + 1] - path[k])
        xy[-1] = path[-1]
        self.ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=size, zorder=zorder, lw=0)

    def legend(self, n_agents, n_modes):
        """Legend in a solid gutter at the bottom, so it never sits over the map."""
        gy = -self.yr + 9.4                       # top of the gutter
        self.ax.add_patch(plt.Rectangle((-self.xr, -self.yr), 2 * self.xr, gy + self.yr,
                                        fc=BEV_BG, ec="none", zorder=20))
        self.ax.plot([-self.xr, self.xr], [gy, gy], color=BEV_GRID, lw=1.0, zorder=21)

        items = [("top-1 plan", "#ff8c2e"), ("scored modes", "#3fb6a8"),
                 ("kmeans anchors", "#6f6a63"), ("logged path", "#f2ede6")]
        for k, (name, c) in enumerate(items):
            col, row = k % 2, k // 2
            x0 = -self.xr + 1.4 + col * 13.0
            y0 = gy - 2.6 - row * 2.6
            self.ax.plot([x0, x0 + 1.8], [y0, y0], color=c, lw=2.6, zorder=22)
            self.ax.text(x0 + 2.3, y0 - 0.6, name, color="#b9b1a7", fontsize=8, zorder=22)
        for k, label in enumerate(MAP_COLORS):
            self.ax.text(self.xr - 1.4, gy - 2.6 - k * 2.0, MAP_NAMES[label],
                         color=MAP_COLORS[label], fontsize=8, ha="right", zorder=22)

        self.ax.text(-self.xr + 1.4, self.yr - 2.2,
                     f"BEV  forward is up   {n_agents} agents tracked   "
                     f"{n_modes} denoised modes", color="#b9b1a7", fontsize=8.5,
                     zorder=22)

    def render(self, data, result, plan, anchors, best, lookahead=None) -> np.ndarray:
        self._reset()
        self.draw_map(result)
        n_agents = self.draw_agents(result)
        self.draw_anchors(anchors)
        self.draw_gt(data["gt_ego_fut_trajs"], data.get("gt_ego_fut_masks"))
        self.draw_modes(plan["modes"], plan["scores"], best)
        if lookahead is not None:
            self.draw_lookahead(lookahead)
        self.draw_ego()
        self.legend(n_agents, len(plan["modes"]))
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- cameras

class CameraPanel:
    """Surround images, with the plan re-projected onto the ground plane.

    The ribbon is the ego's swept corridor, not a centre line: two edges offset by half
    the vehicle width, projected per-quad and depth-clipped. That makes it read as a
    drivable path and makes a lateral error visible against the lane it sits in.
    """

    ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    STRIP = ["CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT",
             "CAM_BACK", "CAM_BACK_RIGHT"]

    def __init__(self, hero_size=(1180, 420), tile_size=(236, 133), crop_top=0.30):
        self.hero_size = hero_size
        self.tile_size = tile_size
        self.crop_top = crop_top

    def _load(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def ground_z(self, data, default=-1.84):
        """Height of the ground plane in the frame the plan lives in.

        Taken from the sensor extrinsics, not from the detections. nuScenes puts the ego
        frame origin on the ground at the rear-axle midpoint, so the LiDAR's z offset is
        exactly its height above the road, and the plan's frame is the LiDAR frame.

        The obvious alternative -- the median bottom face of confident detections -- is
        badly behaved: box-bottom spread within a single frame runs over 2 m, and on
        nuScenes mini_val it produced per-frame estimates from -1.2 to -2.3. Half a metre
        of error is enough to throw the re-projected corridor clean off the bottom of the
        image, which is exactly what it did on scene-0103.
        """
        t = data.get("lidar2ego_translation")
        return -float(t[2]) if t is not None else default

    def draw_ribbon(self, img, path, lidar2img, z, half_width=0.95, cmap="autumn",
                    alpha=0.55, n=90, start_dist=2.5):
        """Draw the plan as a swept corridor on the ground plane.

        `start_dist` skips the first few metres. The path begins under the ego, and a
        quad straddling the camera's optical centre projects to something enormous --
        the near end has to be trimmed or it floods the frame.
        """
        dense = resample(path, n + 20)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
        dense = dense[s >= start_dist]
        if len(dense) < 3:
            return img
        dense = resample(dense, n)
        d = np.gradient(dense, axis=0)
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.maximum(norm, 1e-6)
        perp = np.stack([d[:, 1], -d[:, 0]], axis=1)        # right-hand normal
        left = dense - perp * half_width
        right = dense + perp * half_width

        def to_px(pts):
            p3 = np.concatenate([pts, np.full((len(pts), 1), z)], axis=1)
            return project(p3, lidar2img)

        lp, ld = to_px(left)
        rp, rd = to_px(right)
        colors = (matplotlib.colormaps[cmap](np.linspace(0, 1, n))[:, :3] * 255)[:, ::-1]

        h_img, w_img = img.shape[:2]
        overlay = img.copy()
        drawn = 0
        for i in range(n - 1):
            if min(ld[i], ld[i + 1], rd[i], rd[i + 1]) < 1.5:
                continue
            quad = np.array([lp[i], lp[i + 1], rp[i + 1], rp[i]], dtype=np.float64)
            if not np.isfinite(quad).all():
                continue
            extent = quad.max(axis=0) - quad.min(axis=0)
            # A quad wider than the frame means the segment is grazing the image plane;
            # rasterising it would paint a wedge across everything.
            if extent.max() > 2.0 * max(h_img, w_img) or np.abs(quad).max() > 1e4:
                continue
            cv2.fillConvexPoly(overlay, quad.astype(np.int32), colors[i].tolist(),
                               cv2.LINE_AA)
            drawn += 1
        if drawn:
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            # crisp edges on top of the translucent fill
            for edge, dep in ((lp, ld), (rp, rd)):
                vis = dep > 0.8
                pts = edge[vis]
                if len(pts) > 1 and np.isfinite(pts).all() and np.abs(pts).max() < 1e5:
                    cv2.polylines(img, [pts.astype(np.int32)], False, (255, 255, 255),
                                  1, cv2.LINE_AA)
        return img

    def draw_boxes(self, img, result, lidar2img, thresh=0.4, max_depth=45.0):
        boxes = result["boxes_3d"].numpy()
        scores = result["scores_3d"].numpy()
        labels = result["labels_3d"].numpy()
        ids = result["instance_ids"].numpy()
        keep = np.where(scores >= thresh)[0]
        if len(keep) == 0:
            return img
        corners = box3d_to_corners(boxes[keep])
        h, w = img.shape[:2]
        edges = [(0, 3), (3, 7), (7, 4), (4, 0), (1, 2), (2, 6), (6, 5), (5, 1),
                 (0, 1), (3, 2), (7, 6), (4, 5)]
        order = np.argsort(-boxes[keep, 1])       # far to near
        for n in order:
            i = keep[n]
            px, dep = project(corners[n], lidar2img)
            if dep.min() < 0.5 or dep.mean() > max_depth:
                continue
            if not np.isfinite(px).all() or np.abs(px).max() > 1e4:
                continue
            if px[:, 0].max() < 0 or px[:, 0].min() > w or px[:, 1].max() < 0:
                continue
            c = (AGENT_PALETTE[int(ids[i]) % len(AGENT_PALETTE)][::-1] * 255).tolist()
            p = px.astype(np.int32)
            for a, b in edges:
                cv2.line(img, tuple(p[a]), tuple(p[b]), c, 2, cv2.LINE_AA)
            top = p[[1, 2, 6, 5]].mean(axis=0).astype(int)
            name = CLASS_NAMES[int(labels[i])]
            text(img, f"{name} {dep.mean():.0f}m", (top[0] - 32, max(top[1] - 8, 12)),
                 0.42, c, 1)
        return img

    def hero(self, data, result, plan, draw_boxes=True):
        idx = self.ORDER.index("CAM_FRONT")
        img = self._load(data["img_filename"][idx])
        l2i = np.asarray(data["lidar2img"][idx])
        z = self.ground_z(data)
        if draw_boxes:
            self.draw_boxes(img, result, l2i)
        self.draw_ribbon(img, plan, l2i, z)
        h, w = img.shape[:2]
        img = img[int(h * self.crop_top):, :]
        out = cv2.resize(img, self.hero_size, interpolation=cv2.INTER_AREA)
        label = "CAM_FRONT   + DiffusionDrive plan re-projected onto the road"
        text(out, label, (14, 28), 0.55, (0, 0, 0), 3, FONT)   # outline for legibility
        text(out, label, (14, 28), 0.55, TEXT, 1, FONT)
        return out

    def strip(self, data, result, plan):
        tiles = []
        z = self.ground_z(data)
        for cam in self.STRIP:
            idx = self.ORDER.index(cam)
            img = self._load(data["img_filename"][idx])
            l2i = np.asarray(data["lidar2img"][idx])
            if cam in ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"):
                self.draw_ribbon(img, plan, l2i, z, alpha=0.45)
            t = cv2.resize(img, self.tile_size, interpolation=cv2.INTER_AREA)
            name = cam.replace("CAM_", "")
            text(t, name, (7, 16), 0.38, (0, 0, 0), 3)
            text(t, name, (7, 16), 0.38, TEXT, 1)
            tiles.append(t)
        return tiles
