#!/usr/bin/env python3
"""
Closed-loop driving simulator for the Sparse4D-v3 end-to-end vision pipeline.

    6-camera multi-view → DETECT → TRACK → MOTION → PLAN → CONTROL → KBM

The full perception/prediction/planning stack runs on the *logged* nuScenes
6-camera streams (sensors cannot be re-rendered on this machine). The ego side is
genuinely closed-loop: the Kinematic Bicycle Model integrates the controller
output, the controller reads the model's PLANNED trajectory and the *simulated*
ego speed, and we measure how far the closed-loop KBM trajectory drifts from the
logged human ego path (= closed-loop tracking error).

Per-frame log + aggregate metrics (ego divergence vs GT log, collisions, speed
tracking) and an optional BEV GIF.

How the loop is structured
--------------------------
1. Buffer one scene so all timestamps and GT poses are known up front (lets us
   use the real inter-frame dt and seed the KBM with the true initial speed).
2. Initialise the KBM to the first GT pose + GT speed; reset the temporal model.
3. For each frame i:
     a. run the model on the LOGGED 6 cameras -> detections / tracks / motion /
        ego plan (perception always sees the real log);
     b. express the simulated (KBM) ego pose in this frame's log-ego frame
        (``sim_delta``) and record the world-frame divergence vs the GT pose;
     c. feed the plan + the SIMULATED speed to the controller -> (delta, accel);
     d. zero-order-hold: integrate the KBM by the real dt to reach frame i+1.
   The control at step (c) depends on the sim state, and the sim state at (d)
   integrates that control — that feedback is what makes the ego closed-loop.

Frame conventions
-----------------
  * Perception / plan: nuScenes LIDAR_TOP ego frame, x forward, y left.
  * KBM state: world (global) frame, so it compares directly to ego2global.
  * BEV colours (see bev.py): GT/log ego at origin = GREY dashed; predicted KBM
    ego = BLUE; ego plan = GREEN; obstacles coloured by track id.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
    python simulator.py --scene 1 --max-frames 40 --bev

Run from /Users/trish/VLMProjects/simulator.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

# the reproduced Sparse4D-v3 stack
PIPELINE_ROOT = Path("/Users/trish/VLMProjects/sparse4d_vldrive")
sys.path.insert(0, str(PIPELINE_ROOT))

from sparse4d_vl.data.finetune_loader import NuScenesFinetuneLoader  # noqa: E402
from sparse4d_vl.model.sparse4d_v3 import Sparse4Dv3                  # noqa: E402
from sparse4d_vl.model.checkpoint import load_checkpoint             # noqa: E402

from kbm import KinematicBicycleModel, EgoState                       # noqa: E402
from controller import TrajectoryController                           # noqa: E402
import bev                                                            # noqa: E402

_CMD = {0: "right", 1: "straight", 2: "left"}   # planner command codes
_EGO_L, _EGO_W = 4.08, 1.85                     # ego footprint, length × width (m)


# ---------------------------------------------------------------------------
# small transparent geometry helpers (no shapely / mmdet3d) — CLAUDE.md style
# ---------------------------------------------------------------------------
def _yaw_of(mat: np.ndarray) -> float:
    """Extract the 2D heading from a 4×4 (or 3×3) rotation/transform matrix."""
    return math.atan2(mat[1, 0], mat[0, 0])


def _obb_corners(cx, cy, yaw, length, width) -> np.ndarray:
    """Four corners of an oriented box (length along heading) in its parent frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = length / 2.0, width / 2.0
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])  # box frame
    R = np.array([[c, -s], [s, c]])                                 # box -> parent
    return local @ R.T + np.array([cx, cy])


def _obb_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test (SAT) for two convex quads, each (4, 2).

    Two convex polygons are disjoint iff some edge-normal axis separates their
    projections. We test every edge normal of both quads; if any axis separates
    them they do NOT overlap. If none does, they overlap.
    """
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]])      # edge normal = candidate axis
            n = np.linalg.norm(axis)
            if n < 1e-9:
                continue
            axis /= n
            pa, pb = a @ axis, b @ axis               # project both quads onto axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False  # gap on this axis -> separated -> no overlap
    return True


def _collision_indices(boxes: np.ndarray, sim_delta) -> list[int]:
    """Return the indices of obstacles overlapping the SIMULATED ego footprint.

    Boxes arrive in the LOG ego frame; the sim ego sits at ``sim_delta`` in that
    same frame. We move each box into the SIM ego frame (so the ego is at the
    origin) and OBB-test it against the ego rectangle. Evaluating in the sim frame
    means collisions reflect the diverged closed-loop pose, not the logged pose.
    The returned indices let the BEV highlight exactly which boxes collide.
    """
    if boxes is None or len(boxes) == 0:
        return []
    dx, dy, dyaw = sim_delta
    c, s = math.cos(-dyaw), math.sin(-dyaw)
    R = np.array([[c, -s], [s, c]])               # log-ego -> sim-ego rotation
    ego = _obb_corners(0, 0, 0, _EGO_L, _EGO_W)   # ego rect at the sim-ego origin
    hits = []
    for j, b in enumerate(boxes):                 # box: [x,y,z,w,l,h,yaw,vx,vy]
        cx, cy, yaw, w, l = b[0], b[1], b[6], b[3], b[4]
        # translate by -sim_pos then rotate by -dyaw to land in the sim-ego frame
        rel = R @ (np.array([cx, cy]) - np.array([dx, dy]))
        corners = _obb_corners(rel[0], rel[1], yaw - dyaw, l, w)
        if _obb_overlap(ego, corners):
            hits.append(j)
    return hits


# ---------------------------------------------------------------------------
def main():
    """Parse args, build the stack, run one scene closed-loop, print + render."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default=str(PIPELINE_ROOT / "checkpoints/train_v3_plan3/epoch_05.pt"))
    p.add_argument("--dataroot", default="/Users/trish/Downloads/nuScenes_miniV1.0")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=40)
    p.add_argument("--bev", action="store_true", help="render BEV PNGs + GIF")
    p.add_argument("--out", default="sim_outputs")
    args = p.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # peek at the checkpoint keys to decide whether the planner uses map tokens
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with_map = any("map_encoder" in k for k in ck.get("model", ck))
    print(f"[sim] checkpoint={args.checkpoint}  with_map={with_map}  device={dev}")

    model = Sparse4Dv3(with_planning=True, with_map=with_map, ego_steps=6).to(dev)
    model.eval()
    load_checkpoint(model, args.checkpoint, version="v3")
    loader = NuScenesFinetuneLoader(args.dataroot, version=args.version,
                                    future_steps=12, plan=True, with_map=with_map)

    kbm = KinematicBicycleModel(wheelbase=2.85, device=dev)
    ctrl = TrajectoryController(wheelbase=2.85)

    # ---- buffer the scene so we know all timestamps / GT poses up front ----
    frames = []
    for i, fr in enumerate(loader.iter_scene(args.scene)):
        if i >= args.max_frames:
            break
        frames.append(fr)
    n = len(frames)
    if n < 2:
        print("[sim] scene too short"); return

    # GT timeline from the log: timestamps (s), ego world xy + heading per frame
    ts = np.array([f["img_metas"]["timestamp"] for f in frames])
    gt_xy = np.array([f["img_metas"]["ego2global"][:2, 3] for f in frames])
    gt_yaw = np.array([_yaw_of(f["img_metas"]["ego2global"]) for f in frames])
    dts = np.diff(ts)                                          # real inter-frame Δt
    # GT speed from consecutive translations (used to seed v0 and for the HUD)
    gt_speed = np.linalg.norm(np.diff(gt_xy, axis=0), axis=1) / np.clip(dts, 1e-3, None)
    v0 = float(gt_speed[0])

    # start the simulated ego exactly on the first GT pose + GT speed
    kbm.reset(EgoState(x=float(gt_xy[0, 0]), y=float(gt_xy[0, 1]),
                       yaw=float(gt_yaw[0]), v=v0))
    ctrl.reset(); model.reset_state()

    outdir = Path(args.out); outdir.mkdir(exist_ok=True)
    # wipe stale frames so we never mix old (e.g. pre-collision-highlight) PNGs
    # with this run's output — every frame in outdir belongs to THIS run.
    if args.bev:
        for old in outdir.glob("frame_*.png"):
            old.unlink()
    png_paths = []

    print(f"\n[sim] scene {args.scene}: {n} frames — closed-loop ego (KBM), log sensors\n")
    divergences, sim_speeds, col_frames = [], [], 0
    for i in range(n):
        fr = frames[i]
        metas = fr["img_metas"]
        with torch.no_grad():
            out = model(fr["imgs"].float(), metas)
        det = out["detections"][0]
        boxes = det["boxes_3d"].cpu().numpy()
        tids = det["track_ids"].cpu().numpy() if "track_ids" in det else None
        trajs = det["trajectories"].cpu().numpy() if "trajectories" in det else None
        tsco = det["traj_scores"].cpu().numpy() if "traj_scores" in det else None
        cmd = int(fr["command"])
        plan = out["ego_traj"][0, cmd].cpu().numpy()        # (Te, 2) ego-frame disp

        # closed-loop pose of the simulated ego, expressed in this frame's log-ego frame
        ego = kbm.ego
        dp = np.array([ego.x - gt_xy[i, 0], ego.y - gt_xy[i, 1]])
        cy, sy = math.cos(-gt_yaw[i]), math.sin(-gt_yaw[i])
        rel = np.array([[cy, -sy], [sy, cy]]) @ dp           # into log-ego frame
        sim_delta = (float(rel[0]), float(rel[1]),
                     float((ego.yaw - gt_yaw[i] + math.pi) % (2 * math.pi) - math.pi))
        divergence = float(np.linalg.norm(dp))
        divergences.append(divergence)
        sim_speeds.append(ego.v)

        # controller reads the PLAN + the SIMULATED speed (closed loop)
        c = ctrl.control(plan, ego.v)
        col_idx = _collision_indices(boxes, sim_delta)   # which boxes hit the sim ego
        hits = len(col_idx)
        col_frames += int(hits > 0)

        n_trk = int((tids >= 0).sum()) if tids is not None else 0
        ego_end = plan[-1]
        print(f"  f{i:02d} | boxes={len(boxes):3d} tracks={n_trk:3d} "
              f"| cmd={_CMD[cmd]:8s} plan_end=({ego_end[0]:5.1f},{ego_end[1]:5.1f})m "
              f"| v_sim={ego.v:4.1f}(gt {gt_speed[min(i, n-2)]:4.1f}) "
              f"steer={math.degrees(c.delta):+5.1f}d accel={c.accel:+4.1f} "
              f"| div={divergence:4.2f}m hits={hits}")

        if args.bev:
            png = str(outdir / f"frame_{i:03d}.png")
            bev.render_frame(png, boxes=boxes, track_ids=tids,
                             trajectories=trajs, traj_scores=tsco,
                             ego_plan=plan, sim_delta=sim_delta,
                             collision_idx=col_idx,
                             divergence=divergence, speed=ego.v, control=c,
                             frame_idx=i, n_tracks=n_trk,
                             title=f"scene {args.scene}  cmd={_CMD[cmd]}")
            png_paths.append(png)

        # advance the KBM to the next frame with zero-order-hold control
        if i < n - 1:
            kbm.step(c.delta, c.accel, float(dts[i]), substeps=10)

    div = np.array(divergences)
    print(f"\n[sim] === closed-loop summary (scene {args.scene}, {n} frames) ===")
    print(f"  ego divergence vs GT log : mean {div.mean():.2f} m | "
          f"max {div.max():.2f} m | final {div[-1]:.2f} m")
    print(f"  mean sim speed           : {np.mean(sim_speeds):.2f} m/s "
          f"(GT mean {gt_speed.mean():.2f} m/s)")
    print(f"  collision frames         : {col_frames}/{n}")
    if args.bev and png_paths:
        gif = str(outdir / f"scene{args.scene}_closedloop.gif")
        bev.make_gif(png_paths, gif, duration_ms=400)
        print(f"  BEV frames               : {len(png_paths)} PNGs + {gif}")
    print("[sim] full chain ran: multi-view→track→motion→plan→control→KBM ✓")


if __name__ == "__main__":
    main()
