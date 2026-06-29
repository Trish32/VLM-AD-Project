"""QCNet qualitative visualisation → GIF (pure-PyTorch port, Apple MPS).

Renders an Argoverse 2 scenario in the official QCNet "Qualitative Results" style
and animates it over the 11 s window:

  • HD map      — dark drivable area, light lane boundaries, salmon crosswalks
  • agents      — oriented boxes; the FOCAL agent is blue, others are grey
  • history     — the focal agent's past trajectory (solid blue)
  • prediction  — QCNet's K multimodal future trajectories, fanned out and
                  coloured per mode, opacity ∝ mode probability
  • ground truth— the focal agent's actual future (dashed green)

The prediction is made at t = 4.9 s (the last history step); as the clip plays the
agents drive and the focal agent's true future can be compared against the modes.

Example:
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
        python visualize.py --root "/Users/trish/Downloads/Argoverse 2" --index 0
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection
from PIL import Image
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder
from utils.data_utils import to_device

C_BG = "#5b5b5e"; C_ROAD = "#14141e"; C_LANE = "#9aa0a6"; C_XWALK = "#e0875a"
C_FOCAL = "#2b6cd6"; C_OTHER = "#c9ccd1"; C_GT = "#39d353"
NUM_HIST = 50                                       # AV2: 5 s history @ 10 Hz
SIZES = {"VEHICLE": (4.6, 2.0), "BUS": (12.0, 3.0), "PEDESTRIAN": (0.8, 0.8),
         "CYCLIST": (2.0, 0.8), "MOTORCYCLIST": (2.2, 0.9), "RIDERLESS_BICYCLE": (2.0, 0.8)}


def _box(cx, cy, th, l, w):
    c, s = np.cos(th), np.sin(th)
    pts = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return pts @ np.array([[c, -s], [s, c]]).T + [cx, cy]


def _agent_tracks(scenario, T=110):
    tracks = []
    for tr in scenario.tracks:
        pos = np.full((T, 2), np.nan); head = np.full(T, np.nan)
        for st in tr.object_states:
            if 0 <= st.timestep < T:
                pos[st.timestep] = st.position; head[st.timestep] = st.heading
        tracks.append(dict(pos=pos, head=head,
                           type=str(tr.object_type).split(".")[-1],
                           focal=str(tr.category).endswith("FOCAL_TRACK")))
    return tracks


def _draw_map(ax, amap):
    for da in amap.vector_drivable_areas.values():
        ax.add_patch(MplPolygon(da.xyz[:, :2], closed=True, facecolor=C_ROAD,
                                edgecolor="none", zorder=1))
    segs = []
    for ls in amap.vector_lane_segments.values():
        segs.append(ls.left_lane_boundary.xyz[:, :2])
        segs.append(ls.right_lane_boundary.xyz[:, :2])
    ax.add_collection(LineCollection(segs, colors=C_LANE, linewidths=0.6,
                                     alpha=0.45, zorder=2))
    for pc in amap.vector_pedestrian_crossings.values():
        ax.add_patch(MplPolygon(pc.polygon[:, :2], closed=True, facecolor=C_XWALK,
                                edgecolor="none", alpha=0.55, zorder=2))


def render_frame(out_path, amap, tracks, focal, preds, pi, t, center, R, scene_id):
    fig, ax = plt.subplots(figsize=(7.6, 7.6), dpi=110)
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    _draw_map(ax, amap)

    # agents present at this timestep
    for ag in tracks:
        p, h = ag["pos"][t], ag["head"][t]
        if np.isnan(p[0]):
            continue
        l, w = SIZES.get(ag["type"], (4.0, 1.8))
        ax.add_patch(MplPolygon(_box(p[0], p[1], h, l, w), closed=True,
                                facecolor=C_FOCAL if ag["focal"] else C_OTHER,
                                edgecolor="black", linewidth=0.4, zorder=5))

    fp = focal["pos"]
    # focal history (up to "now") + ground-truth future
    th = min(t, NUM_HIST - 1)
    ax.plot(fp[:th + 1, 0], fp[:th + 1, 1], color=C_FOCAL, lw=1.6, zorder=4)
    ax.plot(fp[NUM_HIST - 1:, 0], fp[NUM_HIST - 1:, 1], color=C_GT, lw=1.8,
            ls=(0, (4, 2)), zorder=4, label="ground truth")

    # K multimodal predictions, opacity ∝ probability, colour per mode
    order = np.argsort(-pi)
    cmap = plt.cm.cool(np.linspace(0, 1, len(order)))
    pn = (pi - pi.min()) / (pi.max() - pi.min() + 1e-9)
    for rank, k in enumerate(order):
        tr = preds[k]
        ax.plot(tr[:, 0], tr[:, 1], color=cmap[rank], lw=1.6, ls=(0, (5, 2)),
                alpha=0.35 + 0.6 * pn[k], zorder=3)
        d = tr[-1] - tr[-2]
        ax.annotate("", xy=tr[-1], xytext=tr[-1] - d * 0.01, zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=cmap[rank],
                                    alpha=0.35 + 0.6 * pn[k], lw=1.4))

    ax.set_xlim(center[0] - R, center[0] + R)
    ax.set_ylim(center[1] - R, center[1] + R)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"QCNet — multimodal forecast    t = {t / 10:.1f} s"
                 + ("   (prediction @ 4.9 s)" if t >= NUM_HIST - 1 else "   (history)"),
                 color="white", fontsize=11)
    fig.text(0.012, 0.012, f"scenario {scene_id}", color="#d0d0d0", fontsize=7.5,
             family="monospace", va="bottom")
    fig.text(0.988, 0.012, "blue = focal agent    green ┄ ground truth"
             "    fan = K-modal prediction (opacity ∝ prob)", color="#d0d0d0",
             fontsize=7.5, va="bottom", ha="right")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, facecolor=C_BG)
    plt.close(fig)


def make_gif(paths, gif_path, duration_ms=140):
    frames = [Image.open(p).convert("RGB") for p in paths]
    pframes = [f.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)
               for f in frames]
    pframes[0].save(gif_path, save_all=True, append_images=pframes[1:],
                    duration=duration_ms, loop=0, disposal=2)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt_path", default="ckpt/QCNet_AV2.ckpt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--step", type=int, default=3, help="timestep stride per frame")
    ap.add_argument("--out", default="viz_outputs")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = QCNet.from_checkpoint(args.ckpt_path, map_location="cpu").to(device).eval()
    dataset = ArgoverseV2Dataset(
        root=args.root, split=args.split,
        transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
        dim=3, num_historical_steps=model.num_historical_steps,
        num_future_steps=model.num_future_steps)

    data = to_device(dataset[args.index], device)
    pred = model(data)
    traj = torch.cat([pred["loc_refine_pos"][..., :model.output_dim],
                      pred["scale_refine_pos"][..., :model.output_dim]], dim=-1)
    mask = data["agent"]["category"] == 3                       # focal track
    origin = data["agent"]["position"][mask, model.num_historical_steps - 1]
    theta = data["agent"]["heading"][mask, model.num_historical_steps - 1]
    cos, sin = theta.cos(), theta.sin()
    rot = torch.zeros(int(mask.sum()), 2, 2, device=device)
    rot[:, 0, 0] = cos; rot[:, 0, 1] = sin; rot[:, 1, 0] = -sin; rot[:, 1, 1] = cos
    traj_world = (torch.matmul(traj[mask, :, :, :2], rot.unsqueeze(1))
                  + origin[:, :2].reshape(-1, 1, 1, 2))
    preds = traj_world[0].cpu().numpy()                          # (K, T, 2)
    pi = F.softmax(pred["pi"][mask], dim=-1)[0].cpu().numpy()    # (K,)

    sid = data["scenario_id"]
    base = Path(args.root) / args.split / sid
    scenario = scenario_serialization.load_argoverse_scenario_parquet(
        base / f"scenario_{sid}.parquet")
    amap = ArgoverseStaticMap.from_json(base / f"log_map_archive_{sid}.json")
    tracks = _agent_tracks(scenario)
    focal = next(a for a in tracks if a["focal"])

    center = focal["pos"][NUM_HIST - 1]
    pts = np.concatenate([focal["pos"][~np.isnan(focal["pos"][:, 0])],
                          preds.reshape(-1, 2)], 0)
    R = float(np.clip(np.abs(pts - center).max() + 12.0, 35.0, 80.0))

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.out):
        if f.startswith("qc_") and f.endswith(".png"):
            os.remove(os.path.join(args.out, f))
    print(f"[qc-viz] scenario {sid}  agents={len(tracks)}  R={R:.0f}m  "
          f"mode probs={np.round(np.sort(pi)[::-1], 3)}")

    paths = []
    frames_t = list(range(0, 110, args.step)) + [109] * 6        # hold last frame
    for j, t in enumerate(frames_t):
        p = os.path.join(args.out, f"qc_{j:03d}.png")
        render_frame(p, amap, tracks, focal, preds, pi, t, center, R, sid)
        paths.append(p)

    gif = os.path.join(args.out, f"{sid}_qcnet.gif")
    make_gif(paths, gif)
    print(f"[qc-viz] wrote {gif}  ({len(paths)} frames)")


if __name__ == "__main__":
    main()
