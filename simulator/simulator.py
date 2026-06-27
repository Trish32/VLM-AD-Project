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
from pyquaternion import Quaternion

# the reproduced Sparse4D-v3 stack
PIPELINE_ROOT = Path("/Users/trish/VLMProjects/sparse4d_vldrive")
sys.path.insert(0, str(PIPELINE_ROOT))

from sparse4d_vl.data.finetune_loader import NuScenesFinetuneLoader  # noqa: E402
from sparse4d_vl.model.sparse4d_v3 import Sparse4Dv3                  # noqa: E402
from sparse4d_vl.model.checkpoint import load_checkpoint             # noqa: E402
from sparse4d_vl.tools.visualizer import visualise_frame            # noqa: E402

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


def _lidar2ego(nusc, sample_token: str):
    """Return (R, t, yaw_off) for the LIDAR_TOP→ego transform of this frame.

    nuScenes mounts LIDAR_TOP rotated ~-90° from the ego frame (lidar +x = vehicle
    right, +y = forward). Perception runs in the lidar frame, but the planner and
    the simulated ego live in the ego frame (x forward, y left). We therefore lift
    every lidar-frame quantity into the ego frame before rendering so the plan,
    boxes and lanes share one "forward = +x = up" convention.
    """
    sample = nusc.get('sample', sample_token)
    cs = nusc.get('calibrated_sensor',
                  nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                  ['calibrated_sensor_token'])
    q = Quaternion(cs['rotation'])
    R = q.rotation_matrix[:2, :2].astype(np.float64)      # lidar→ego (xy block)
    t = np.array(cs['translation'][:2], dtype=np.float64)
    return R, t, float(q.yaw_pitch_roll[0])


def _boxes_to_ego(boxes, R, t, yaw_off):
    """Lift detection boxes [x,y,z,len,wid,h,yaw,vx,vy] from lidar→ego frame."""
    if boxes is None or len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, 0:2] = b[:, 0:2] @ R.T + t        # centre: rotate then translate
    b[:, 6]   = b[:, 6] + yaw_off          # heading rotates with the frame
    b[:, 7:9] = b[:, 7:9] @ R.T            # velocity: rotation only (a direction)
    return b


def _trajs_to_ego(trajs, R):
    """Rotate motion-forecast displacements (N,K,T,2) from lidar→ego frame."""
    if trajs is None or len(trajs) == 0:
        return trajs
    return trajs @ R.T                      # displacements: rotation only


def _combine_panels(cam_path, bev_path, out_path, height: int = 400):
    """SparseDrive-style composite: surround-camera grid (left) | BEV panel (right).

    Both panels are scaled to a common height and concatenated horizontally —
    mirroring SparseDrive's tools/visualization layout (cameras + BEV prediction).
    """
    from PIL import Image
    cam = Image.open(cam_path).convert("RGB")
    bev = Image.open(bev_path).convert("RGB")
    cam = cam.resize((max(1, round(cam.width * height / cam.height)), height))
    bev = bev.resize((height, height))
    combo = Image.new("RGB", (cam.width + bev.width, height), "#0e0e12")
    combo.paste(cam, (0, 0))
    combo.paste(bev, (cam.width, 0))
    combo.save(out_path)


def _boxes_ego_to_world(boxes, ego2global, origin):
    """Lift ego-frame boxes into the world-local frame (global metres − origin)."""
    if boxes is None or len(boxes) == 0:
        return boxes
    R = ego2global[:2, :2]; t = ego2global[:2, 3]
    yaw_e2g = _yaw_of(ego2global)
    b = boxes.copy()
    b[:, 0:2] = b[:, 0:2] @ R.T + t - np.asarray(origin)   # ego → global → local
    b[:, 6]   = b[:, 6] + yaw_e2g                          # heading → world
    return b


def _scene_lanes_world(loader, sample_token, origin, gt_xy, margin: float = 60.0) -> dict:
    """HD-map dividers for the whole scene in the world-local frame (fetched once).

    The map is global, so we pull every divider within the trajectory's bounding
    box (+margin) and shift by the fixed origin — the lanes are then static across
    all frames and the ego visibly drives along them.
    """
    nusc   = loader.nusc
    sample = nusc.get('sample', sample_token)
    location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
    nmap   = loader._get_map(location)
    xs, ys = gt_xy[:, 0], gt_xy[:, 1]
    patch = (xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin)
    dividers = []
    recs = nmap.get_records_in_patch(patch, ['road_divider', 'lane_divider'], mode='intersect')
    for layer in ('road_divider', 'lane_divider'):
        for token in recs.get(layer, []):
            line = nmap.extract_line(nmap.get(layer, token)['line_token'])
            if line.is_empty:
                continue
            g = np.asarray(line.coords)[:, :2] - np.asarray(origin)
            if g.shape[0] >= 2:
                dividers.append(g.astype(np.float32))
    return {'divider': dividers, 'boundary': []}


def _lane_polylines(loader, sample_token: str, radius: float = 55.0) -> dict:
    """HD-map lane / road dividers near the ego, in this frame's LIDAR_TOP frame.

    The BEV is drawn in the log-ego (LIDAR_TOP) frame, so we transform each map
    divider from global coordinates into that frame with the *same* global→lidar
    transform the loader uses for boxes — the lanes then line up exactly with the
    detections. Returns ``{'divider': [(n, 2), ...], 'boundary': [...]}`` (metres).
    """
    nusc   = loader.nusc
    sample = nusc.get('sample', sample_token)
    location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
    nmap   = loader._get_map(location)

    lid_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', lid_sd['ego_pose_token'])
    cs = nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])
    R_e2g = Quaternion(ep['rotation']).rotation_matrix; t_e2g = np.array(ep['translation'])
    R_l2e = Quaternion(cs['rotation']).rotation_matrix; t_l2e = np.array(cs['translation'])
    ex, ey = float(t_e2g[0]), float(t_e2g[1]); r = float(radius)
    patch = (ex - r, ey - r, ex + r, ey + r)

    def to_lidar(coords):
        return np.array(
            [loader._global_to_lidar(np.array([x, y, 0.0]),
                                     R_e2g, t_e2g, R_l2e, t_l2e)[:2] for x, y in coords],
            dtype=np.float32)

    dividers = []
    recs = nmap.get_records_in_patch(patch, ['road_divider', 'lane_divider'],
                                     mode='intersect')
    for layer in ('road_divider', 'lane_divider'):
        for token in recs.get(layer, []):
            line = nmap.extract_line(nmap.get(layer, token)['line_token'])
            if line.is_empty:
                continue
            lid = to_lidar(np.asarray(line.coords)[:, :2])
            near = lid[np.linalg.norm(lid, axis=1) <= r]   # keep the in-view portion
            if near.shape[0] >= 2:
                dividers.append(near)
    return {'divider': dividers, 'boundary': []}


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
        # box[3] = length (extent along heading), box[4] = width — see bev.py note
        cx, cy, yaw, length, width = b[0], b[1], b[6], b[3], b[4]
        # translate by -sim_pos then rotate by -dyaw to land in the sim-ego frame
        rel = R @ (np.array([cx, cy]) - np.array([dx, dy]))
        corners = _obb_corners(rel[0], rel[1], yaw - dyaw, length, width)
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
        for old in outdir.glob("world_*.png"):
            old.unlink()
        for old in outdir.glob("combo_*.png"):
            old.unlink()
    png_paths, world_paths, combo_paths = [], [], []
    # world-frame view: fixed map + both ego paths, shifted by a constant origin
    world_origin = gt_xy[0].copy()
    world_lanes = None                       # fetched once on the first frame
    gt_world_path, sim_world_path = [], []

    print(f"\n[sim] scene {args.scene}: {n} frames — closed-loop ego (KBM), log sensors\n")
    divergences, sim_speeds, col_frames = [], [], 0
    for i in range(n):
        fr = frames[i]
        metas = fr["img_metas"]
        with torch.no_grad():
            out = model(fr["imgs"].float(), metas)
        det = out["detections"][0]
        boxes_lidar = det["boxes_3d"].cpu().numpy()         # lidar frame (for cameras)
        scores = det["scores_3d"].cpu().numpy() if "scores_3d" in det else None
        labels = det["labels_3d"].cpu().numpy() if "labels_3d" in det else None
        tids = det["track_ids"].cpu().numpy() if "track_ids" in det else None
        trajs = det["trajectories"].cpu().numpy() if "trajectories" in det else None
        tsco = det["traj_scores"].cpu().numpy() if "traj_scores" in det else None
        cmd = int(fr["command"])
        plan = out["ego_traj"][0, cmd].cpu().numpy()        # (Te, 2) ego-frame disp

        # Lift the lidar-frame perception (boxes, forecasts) into the EGO frame so
        # it shares the planner's "forward = +x" convention; otherwise the agents
        # render ~90° rotated from the plan (lidar +y = ego forward).
        R_l2e, t_l2e, yaw_l2e = _lidar2ego(loader.nusc, metas["sample_token"])
        boxes = _boxes_to_ego(boxes_lidar, R_l2e, t_l2e, yaw_l2e)
        trajs = _trajs_to_ego(trajs, R_l2e)

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
            lanes = _lane_polylines(loader, metas["sample_token"])  # lidar frame
            map_lines = {k: [pl @ R_l2e.T + t_l2e for pl in v]      # → ego frame
                         for k, v in lanes.items()}
            bev.render_frame(png, boxes=boxes, track_ids=tids,
                             trajectories=trajs, traj_scores=tsco,
                             ego_plan=plan, sim_delta=sim_delta,
                             collision_idx=col_idx, map_lines=map_lines,
                             divergence=divergence, speed=ego.v, control=c,
                             frame_idx=i, n_tracks=n_trk,
                             title=f"scene {args.scene}  cmd={_CMD[cmd]}")
            png_paths.append(png)

            # ---- world-frame view (global, north-up): fixed map + both paths ----
            if world_lanes is None:
                world_lanes = _scene_lanes_world(loader, metas["sample_token"],
                                                 world_origin, gt_xy)
            gt_w  = gt_xy[i] - world_origin
            sim_w = np.array([ego.x, ego.y]) - world_origin
            gt_world_path.append(gt_w)
            sim_world_path.append(sim_w)
            boxes_w = _boxes_ego_to_world(boxes, metas["ego2global"], world_origin)
            wpng = str(outdir / f"world_{i:03d}.png")
            bev.render_world_frame(
                wpng, boxes=boxes_w, track_ids=tids, map_lines=world_lanes,
                gt_ego=(float(gt_w[0]), float(gt_w[1]), float(gt_yaw[i])),
                sim_ego=(float(sim_w[0]), float(sim_w[1]), float(ego.yaw)),
                gt_path=np.array(gt_world_path), sim_path=np.array(sim_world_path),
                center=(float(gt_w[0]), float(gt_w[1])), lim=50.0,
                frame_idx=i, divergence=divergence,
                title=f"scene {args.scene}  world frame")
            world_paths.append(wpng)

            # ---- SparseDrive-style composite: surround cameras (3D boxes) | BEV ----
            if scores is not None and labels is not None:
                camdir = outdir / "cam"; camdir.mkdir(exist_ok=True)
                # projection_mat is lidar→pixel, so use the LIDAR-frame boxes; swap
                # slots 3/4 so the projected box draws length (not width) along yaw.
                cam_boxes = boxes_lidar.copy()
                cam_boxes[:, [3, 4]] = boxes_lidar[:, [4, 3]]
                cam_path = visualise_frame(fr["imgs"][0].numpy(), metas["projection_mat"],
                                           cam_boxes, scores, labels, i, camdir,
                                           score_thresh=0.3)
                combo = str(outdir / f"combo_{i:03d}.png")
                _combine_panels(cam_path, png, combo)
                combo_paths.append(combo)

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
        print(f"  BEV frames (ego)         : {len(png_paths)} PNGs + {gif}")
        wgif = str(outdir / f"scene{args.scene}_world.gif")
        bev.make_gif(world_paths, wgif, duration_ms=400)
        print(f"  BEV frames (world)       : {len(world_paths)} PNGs + {wgif}")
        if combo_paths:
            sdgif = str(outdir / f"scene{args.scene}_sparsedrive.gif")
            bev.make_gif(combo_paths, sdgif, duration_ms=400)
            print(f"  Composite (cams+BEV)     : {len(combo_paths)} PNGs + {sdgif}")
    print("[sim] full chain ran: multi-view→track→motion→plan→control→KBM ✓")


if __name__ == "__main__":
    main()
