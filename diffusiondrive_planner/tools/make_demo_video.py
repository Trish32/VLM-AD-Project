"""Render a DiffusionDrive demo video on one nuScenes mini_val scene.

Reproduces the layout of upstream's `final_github.mp4` -- surround cameras with the plan
projected onto the road, a BEV panel, and a control/action readout -- from the results
this repo already reproduced on mini_val (L2 avg 0.584 m, see README).

It does **not** re-run the model. `tools/eval_planning.py` already dumped every planner
output for all 81 mini_val samples, so this is a pure rendering pass over that cache and
runs in about a minute on the Mac. Point `--results` at `planning_results_ours.pt` to
render our ported `TruncatedDDIM`, or `planning_results.pt` for upstream's scheduler.

What is on screen, and where each number comes from:

  planned (orange)  derived from `final_planning`, DiffusionDrive's 3 s trajectory.
                    Steering and throttle come from a pure-pursuit + PI controller
                    closed around that trajectory (`demo_control.py`) -- the model
                    outputs a path, not actuator commands, and the video says so.
  measured (cyan)   nuScenes CAN bus and IMU: what the human driver actually did.
  nav command       `gt_ego_fut_cmd`, the high-level command fed *into* the planner.

Usage:
    cd diffusiondrive_planner/upstream-nusc
    PYTHONPATH=.:../.. python ../tools/make_demo_video.py \
        --config projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
        --results ../data/planning_results_ours.pt \
        --scene scene-0916 --out ../data/demo_scene-0916.mp4
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))          # tools/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # repo root

import cv2

from demo_control import (CMD_LIST, CanBus, PLAN_DT, ego_status_fields, maneuver_label,
                          plan_to_control)
from demo_panel import ControlPanel, title_bar
from demo_render import BEVPanel, CameraPanel

# ---- frame geometry (1920x1080) -------------------------------------------
W, H = 1920, 1080
TITLE_H = 60
HERO = (1180, 420)
TILE = (236, 133)
STRIP_H = 140
BEV_W = W - HERO[0]                     # 740
PANEL_H = H - TITLE_H - HERO[1] - STRIP_H
CAN_THROTTLE_FULL = 250.0               # raw CAN counts at full scale


def build_dataset(config_path):
    from mmcv import Config
    import projects.mmdet3d_plugin  # noqa: F401
    from mmdet3d.datasets import build_dataset as _build

    cfg = Config.fromfile(config_path)
    cfg.version = "mini"
    c = dict(cfg.data.val if "val" in cfg.data else cfg.data.test)
    c["ann_file"] = "data/infos/mini/nuscenes_infos_val.pkl"
    c["data_root"] = "data/nuscenes/"
    return _build(c), c["ann_file"]


def scene_frames(infos, nusc_root, scene_name):
    """Indices of one scene, in order, plus its nuScenes scene name."""
    from nuscenes import NuScenes

    nusc = NuScenes("v1.0-mini", dataroot=nusc_root, verbose=False)
    names = {s["token"]: s["name"] for s in nusc.scene}
    by_scene: dict[str, list[int]] = {}
    for i, info in enumerate(infos):
        by_scene.setdefault(names.get(info["scene_token"], info["scene_token"]), []).append(i)
    if scene_name not in by_scene:
        raise SystemExit(f"scene {scene_name!r} not in this split; have {sorted(by_scene)}")
    return by_scene[scene_name], sorted(by_scene)


def gather(ds, results, infos, idxs, anchors, can, steer_ratio):
    """Precompute every per-keyframe quantity, so rendering is a straight loop."""
    frames = []
    integral = 0.0
    for i in idxs:
        data = ds.get_data_info(i)
        res = results[i]["img_bbox"]
        utime = float(infos[i]["timestamp"])

        cmd = int(np.argmax(np.asarray(data["gt_ego_fut_cmd"])))
        modes = res["planning"][cmd].numpy()              # (6, 6, 2) absolute waypoints
        scores = res["planning_score"][cmd].numpy()       # (6,)
        best = int(scores.argmax())
        plan = res["final_planning"].numpy()              # (6, 2)

        v_meas = can.at("speed_kph", utime, ego_status_fields(data["ego_status"])["speed"] * 3.6) / 3.6
        ctrl, integral = plan_to_control(plan, v_meas, steer_ratio, integral=integral)
        lateral, longitudinal = maneuver_label(plan, v_meas)

        # L2 exactly as upstream's PlanningMetric defines it: the mean displacement
        # error over every 0.5 s step up to the horizon, not the endpoint error. This
        # reproduces the README's 0.2434 / 0.5492 / 0.9604 when averaged over mini_val.
        gt = np.asarray(data["gt_ego_fut_trajs"]).copy()
        gt[np.abs(gt) < 0.01] = 0.0
        gt = gt.cumsum(axis=0)
        step_err = np.linalg.norm(plan - gt, axis=-1)
        # Upstream's PlanningMetric drops a sample entirely unless its whole 3 s future is
        # logged, so the last keyframes of a scene are unscored rather than scored on a
        # short horizon. Keeping that gate is what makes this number comparable to the
        # README's 0.2434 / 0.5492 / 0.9604 on mini_val.
        valid = bool(np.asarray(data["gt_ego_fut_masks"]).astype(bool).all())
        l2 = {h: (float(step_err[:2 * h].mean()) if valid else float("nan"))
              for h in (1, 2, 3)}

        p = np.exp(scores - scores.max())
        frames.append(dict(
            index=i, data=data, result=res, utime=utime, cmd=cmd, modes=modes,
            scores=scores, best=best, plan=plan, ctrl=ctrl, l2=l2, l2_3s=l2[3],
            anchors=anchors[cmd], mode_probs=p / p.sum(),
            nav_command=CMD_LIST[cmd], lateral=lateral, longitudinal=longitudinal,
        ))
    return frames


def build_traces(frames, can):
    """Measured series straight from CAN; planned series sampled at the planner's 2 Hz."""
    t0, t1 = frames[0]["utime"], frames[-1]["utime"]

    def can_window(channel):
        t, v = can.series(channel)
        m = (t >= t0 - 5e5) & (t <= t1 + 5e5)
        return t[m], v[m]

    kt = np.array([f["utime"] for f in frames])
    scored = ~np.isnan([f["l2_3s"] for f in frames])
    return {
        "speed": {"meas": can_window("speed_kph"),
                  "plan": (kt, np.array([f["ctrl"].speed * 3.6 for f in frames]))},
        "steer": {"meas": can_window("steer_wheel_deg"),
                  "plan": (kt, np.array([f["ctrl"].steer_wheel_deg for f in frames]))},
        "yaw":   {"meas": can_window("yaw_rate_dps"),
                  "plan": (kt, np.array([np.degrees(f["ctrl"].yaw_rate) for f in frames]))},
        # No measured counterpart: this is the plan scored against the logged path.
        # Keyframes whose 3 s future runs off the end of the scene are dropped, not
        # zeroed -- a zero here would read as a perfect prediction.
        "l2":    {"plan": (kt[scored], np.array([f["l2_3s"] for f in frames])[scored])},
    }


def interp_planned(frames, k, alpha, attr):
    """Linear blend of a planned scalar between keyframe k and k+1.

    The planner itself runs at 2 Hz; a real stack smooths its command between updates,
    and blending here is what keeps the gauges from stepping. The trace panel still
    plots the planned series at its true 2 Hz sample points.
    """
    a = attr(frames[k])
    b = attr(frames[min(k + 1, len(frames) - 1)])
    return float(a + (b - a) * alpha)


def compose_static(cam: CameraPanel, bev: BEVPanel, f) -> np.ndarray:
    """Everything that only changes at the planner rate: cameras and BEV."""
    canvas = np.zeros((H - TITLE_H - PANEL_H, W, 3), np.uint8)
    canvas[:] = (12, 14, 18)[::-1]

    hero = cam.hero(f["data"], f["result"], f["plan"])
    canvas[0:HERO[1], 0:HERO[0]] = hero

    tiles = cam.strip(f["data"], f["result"], f["plan"])
    y = HERO[1] + (STRIP_H - TILE[1]) // 2
    for k, t in enumerate(tiles):
        x = k * TILE[0]
        canvas[y:y + TILE[1], x:x + TILE[0]] = t

    bev_img = bev.render(f["data"], f["result"],
                         {"modes": f["modes"], "scores": f["scores"]},
                         f["anchors"], f["best"], lookahead=f["ctrl"].lookahead)
    bh, bw = bev_img.shape[:2]
    target_h = canvas.shape[0]
    if (bh, bw) != (target_h, BEV_W):
        bev_img = cv2.resize(bev_img, (BEV_W, target_h), interpolation=cv2.INTER_AREA)
    canvas[:, HERO[0]:] = bev_img
    cv2.line(canvas, (HERO[0], 0), (HERO[0], target_h), (70, 62, 58)[::-1], 1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default="../data/planning_results_ours.pt")
    ap.add_argument("--checkpoint", default="../checkpoints/diffusiondrive_nusc_stage2.pth")
    ap.add_argument("--scene", default="scene-0916")
    ap.add_argument("--out", default="../data/demo.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--sub", type=int, default=10, help="output frames per 2 Hz keyframe")
    ap.add_argument("--limit", type=int, default=None, help="only the first N keyframes")
    ap.add_argument("--frames-dir", default=None, help="also dump PNGs here")
    ap.add_argument("--no-boxes", action="store_true", help="skip 3D boxes on the hero cam")
    args = ap.parse_args()

    ds, ann_file = build_dataset(args.config)
    infos = pickle.load(open(ann_file, "rb"))["infos"]
    results = torch.load(args.results, map_location="cpu")
    assert len(results) == len(infos) == len(ds), "results/infos/dataset are misaligned"

    nusc_root = "data/nuscenes"
    idxs, available = scene_frames(infos, nusc_root, args.scene)
    if args.limit:
        idxs = idxs[:args.limit]
    print(f"[scene] {args.scene}: {len(idxs)} keyframes (split has {available})")

    sd = torch.load(args.checkpoint, map_location="cpu")
    anchors = sd.get("state_dict", sd)["head.motion_plan_head.plan_anchor"].numpy()
    print(f"[anchor] plan_anchor {anchors.shape} from the official checkpoint")

    can = CanBus(Path(nusc_root) / "can_bus" / "can_bus", args.scene)
    steer_ratio = can.steer_ratio()
    print(f"[can] steering ratio fit from this scene's yaw rate: {steer_ratio:.2f}:1")

    frames = gather(ds, results, infos, idxs, anchors, can, steer_ratio)
    traces = build_traces(frames, can)
    t_start = frames[0]["utime"]

    cam = CameraPanel(hero_size=HERO, tile_size=TILE)
    bev = BEVPanel(width=BEV_W, height=H - TITLE_H - PANEL_H)
    panel = ControlPanel(width=W, height=PANEL_H)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (W, H))
    if not writer.isOpened():
        raise SystemExit(f"cannot open VideoWriter for {out_path}")
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    subtitle = ("truncated diffusion planner  |  "
                + ("ours: TruncatedDDIM" if "ours" in Path(args.results).name
                   else "upstream DDIMScheduler"))
    scene_l2 = {h: float(np.nanmean([f["l2"][h] for f in frames])) for h in (1, 2, 3)}
    n_out = 0
    for k, f in enumerate(frames):
        static = compose_static(cam, bev, f)
        nxt = frames[min(k + 1, len(frames) - 1)]
        for j in range(args.sub):
            alpha = j / args.sub
            utime = f["utime"] + (nxt["utime"] - f["utime"]) * alpha
            v_meas_kph = can.at("speed_kph", utime, 0.0)
            state = {
                "steer_plan_deg": interp_planned(frames, k, alpha,
                                                 lambda x: x["ctrl"].steer_wheel_deg),
                "steer_meas_deg": can.at("steer_wheel_deg", utime, 0.0),
                "road_wheel": interp_planned(frames, k, alpha,
                                             lambda x: x["ctrl"].road_wheel),
                "speed_plan_kph": interp_planned(frames, k, alpha,
                                                 lambda x: x["ctrl"].speed * 3.6),
                "speed_meas_kph": v_meas_kph,
                "accel_plan": interp_planned(frames, k, alpha, lambda x: x["ctrl"].accel),
                "accel_meas": can.at("accel_long", utime, 0.0),
                "throttle_plan": interp_planned(frames, k, alpha,
                                                lambda x: x["ctrl"].throttle),
                "brake_plan": interp_planned(frames, k, alpha, lambda x: x["ctrl"].brake),
                "throttle_meas": can.at("throttle", utime, 0.0) / CAN_THROTTLE_FULL,
                "throttle_raw": can.at("throttle", utime, 0.0),
                "yaw_plan": interp_planned(frames, k, alpha, lambda x: x["ctrl"].yaw_rate),
                "yaw_meas_dps": can.at("yaw_rate_dps", utime, 0.0),
                "curvature": interp_planned(frames, k, alpha,
                                            lambda x: x["ctrl"].curvature),
                "lookahead_dist": interp_planned(frames, k, alpha,
                                                 lambda x: x["ctrl"].lookahead_dist),
                "speed_profile": f["ctrl"].speed_profile,
                "nav_command": f["nav_command"],
                "lateral": f["lateral"],
                "longitudinal": f["longitudinal"],
                "mode_probs": f["mode_probs"],
                "best_mode": f["best"],
                "l2": f["l2"],
                "traces": traces,
                "t_now": utime,
            }
            frame = np.zeros((H, W, 3), np.uint8)
            title_bar(frame[0:TITLE_H], W, TITLE_H, args.scene, k, len(frames),
                      (utime - t_start) / 1e6, subtitle, scene_l2)
            frame[TITLE_H:TITLE_H + static.shape[0]] = static
            frame[H - PANEL_H:] = panel.render(state)
            writer.write(frame)
            if frames_dir and j == 0:
                cv2.imwrite(str(frames_dir / f"{k:04d}.png"), frame)
            n_out += 1
        print(f"\r[render] keyframe {k + 1}/{len(frames)}  ({n_out} frames)", end="",
              flush=True)

    writer.release()
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"\n[done] {out_path}  {n_out} frames @ {args.fps} fps "
          f"({n_out / args.fps:.1f} s, {size_mb:.1f} MB)")

    print(f"[check] {args.scene} L2 (upstream PlanningMetric definition): "
          f"1s {scene_l2[1]:.4f}  2s {scene_l2[2]:.4f}  3s {scene_l2[3]:.4f} m")


if __name__ == "__main__":
    main()
