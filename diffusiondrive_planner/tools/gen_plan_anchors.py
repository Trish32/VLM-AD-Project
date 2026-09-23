"""Regenerate DiffusionDrive's plan/motion anchors directly from nuScenes.

Upstream builds these in two stages: `tools/data_converter/nuscenes_converter.py`
writes `gt_ego_fut_trajs`/`gt_ego_fut_cmd` into an infos pkl, then
`tools/kmeans/kmeans_plan.py` clusters that pkl. This script collapses both into one
pass over the raw dataset, so the anchors do not depend on an artifact whose provenance
we cannot inspect.

Two things this surfaced that the shipped artifact would have hidden
-------------------------------------------------------------------
1. **Upstream's kmeans_plan.py emits the wrong shape for this head.** As shipped it
   clusters ALL commands together and writes `(K, 6, 2)` to `kmeans_plan_vocab_K.npy`.
   The v13 head does `self.plan_anchor[None].tile(bs,1,1,1,1)` and indexes
   `plan_anchor[bs, cmd]`, i.e. it needs `(3, K, 6, 2)` from `kmeans_plan_K.npy`. The
   per-command path (`navi_trajs = [[], [], []]` ... `np.stack(clusters, axis=0)`) is
   commented out in the shipped file. `--mode per_command` restores it; `--mode vocab`
   reproduces the shipped behaviour for comparison.

2. **The planning frame is x=lateral (right positive), y=forward.** The converter's
   command rule keys on `ego_fut_trajs[-1][0] >= 2` for a RIGHT turn, and the head's
   normalization is correspondingly asymmetric (x/3 vs (y+0.5)/8.1). Getting this
   backwards rotates every anchor 90 degrees while keeping all shapes valid, so the
   script asserts the convention against the data rather than trusting it.

Usage
-----
    conda run -n uniad2.0 python diffusiondrive_planner/tools/gen_plan_anchors.py \
        --dataroot "$NUSCENES_DATAROOT" \
        --version v1.0-mini --out diffusiondrive_planner/data/kmeans

Caveat: nuScenes mini has 10 scenes / ~400 samples. Upstream clustered full trainval
(~28k). Mini-derived anchors are for pipeline validation, NOT for a real training run —
the script prints per-command sample counts so this stays visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

EGO_FUT_TS = 6          # 3s at 2Hz
CMD_RIGHT, CMD_LEFT, CMD_STRAIGHT = 0, 1, 2
CMD_NAMES = {CMD_RIGHT: "right", CMD_LEFT: "left", CMD_STRAIGHT: "straight"}
CMD_THRESHOLD = 2.0     # metres of lateral offset at the final step


def get_global_sensor_pose(sample, nusc):
    """LIDAR_TOP pose in global. Mirrors nuscenes_converter.get_global_sensor_pose."""
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion

    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pose_record = nusc.get("ego_pose", sd["ego_pose_token"])
    cs_record = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    ego2global = transform_matrix(
        pose_record["translation"], Quaternion(pose_record["rotation"]), inverse=False
    )
    sensor2ego = transform_matrix(
        cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False
    )
    return ego2global.dot(sensor2ego), pose_record, cs_record


def ego_future_for_sample(sample, nusc):
    """Return (deltas (T,2), mask (T,), command int) in the current sample's lidar frame.

    Transcribed from nuscenes_converter.py:364-392. The reference frame is the CURRENT
    sample's ego/calibration, applied to all EGO_FUT_TS+1 future poses.
    """
    from pyquaternion import Quaternion

    trajs = np.zeros((EGO_FUT_TS + 1, 3))
    masks = np.zeros((EGO_FUT_TS + 1,))

    cur = sample
    _, pose_record, cs_record = get_global_sensor_pose(sample, nusc)

    for i in range(EGO_FUT_TS + 1):
        pose_mat, _, _ = get_global_sensor_pose(cur, nusc)
        trajs[i] = pose_mat[:3, 3]
        masks[i] = 1
        if cur["next"] == "":
            trajs[i + 1 :] = trajs[i]   # pad by repeating; mask stays 0
            break
        cur = nusc.get("sample", cur["next"])

    # global -> ego (of the current sample)
    trajs = trajs - np.array(pose_record["translation"])
    trajs = np.dot(Quaternion(pose_record["rotation"]).inverse.rotation_matrix, trajs.T).T
    # ego -> lidar
    trajs = trajs - np.array(cs_record["translation"])
    trajs = np.dot(Quaternion(cs_record["rotation"]).inverse.rotation_matrix, trajs.T).T

    # Command from the FINAL absolute lateral offset, before differencing.
    if trajs[-1][0] >= CMD_THRESHOLD:
        command = CMD_RIGHT
    elif trajs[-1][0] <= -CMD_THRESHOLD:
        command = CMD_LEFT
    else:
        command = CMD_STRAIGHT

    deltas = (trajs[1:] - trajs[:-1])[:, :2]
    return deltas.astype(np.float32), masks[1:].astype(np.float32), command


def collect(nusc):
    """One (deltas, mask, cmd) per sample."""
    from tqdm import tqdm

    out = []
    for sample in tqdm(nusc.sample, desc="ego futures"):
        out.append(ego_future_for_sample(sample, nusc))
    return out


def check_frame_convention(abs_trajs: np.ndarray) -> None:
    """Assert x is lateral and y is forward, from the data itself.

    A frame mix-up preserves every shape and silently rotates the anchors, so this is
    checked rather than assumed. Driving is overwhelmingly forward, so |y| should
    dominate |x| and y should be mostly positive.
    """
    mean_abs = np.abs(abs_trajs[:, -1, :]).mean(axis=0)
    forward_frac = float((abs_trajs[:, -1, 1] > 0).mean())
    print(f"  frame check: mean |x_final|={mean_abs[0]:.2f}m  |y_final|={mean_abs[1]:.2f}m  "
          f"y>0 in {forward_frac:.1%} of samples")
    if mean_abs[1] <= mean_abs[0] or forward_frac < 0.8:
        raise SystemExit(
            "Frame convention check FAILED: expected y=forward to dominate x=lateral.\n"
            "Anchors would be rotated. Check the global->ego->lidar transform."
        )


def check_normalization_compat(anchors: np.ndarray) -> None:
    """Check the anchors survive the head's normalize/denormalize round trip.

    The .npy stores ABSOLUTE waypoints (kmeans_plan.py clusters `.cumsum(axis=-2)`),
    but the head differences them before normalizing, so the normalization range
    (x/3, (y+0.5)/8.1) applies to PER-STEP DELTAS, not endpoints. Endpoints reaching
    40m are fine; a single 0.5s delta above 7.6m is not — it would be clipped, and the
    diffusion would be seeded from a truncated anchor with no error raised anywhere.
    """
    import torch

    from diffusiondrive_planner.truncated_diffusion import (
        anchor_to_deltas,
        denormalize_traj,
        normalize_traj,
    )

    a = torch.from_numpy(anchors)
    deltas = anchor_to_deltas(a)
    err = (denormalize_traj(normalize_traj(deltas)) - deltas).abs()

    dx, dy = deltas[..., 0], deltas[..., 1]
    print(f"  delta range: x=[{dx.min():.2f},{dx.max():.2f}] "
          f"y=[{dy.min():.2f},{dy.max():.2f}]  (normalizable: x±3.0, y -0.5..7.6)")
    clipped = int((err > 1e-4).sum())
    if clipped:
        print(f"  WARNING: {clipped}/{err.numel()} delta components clipped by the head's "
              f"normalization (max loss {err.max():.3f}m). Anchors exceed the range the "
              f"head can represent.")
    else:
        print("  normalization round-trip: lossless (all deltas inside range)")


def cluster(trajs: np.ndarray, k: int, seed: int) -> np.ndarray:
    """kmeans on flattened (T*2) absolute trajectories -> (k, T, 2)."""
    from sklearn.cluster import KMeans

    flat = trajs.reshape(len(trajs), -1)
    if len(flat) < k:
        raise SystemExit(f"only {len(flat)} trajectories for k={k}; need at least k")
    centers = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(flat).cluster_centers_
    return centers.reshape(k, -1, 2).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="diffusiondrive_planner/data/kmeans")
    ap.add_argument("--k", type=int, default=6, help="ego_fut_mode")
    ap.add_argument("--mode", choices=["per_command", "vocab", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"{args.version}: {len(nusc.scene)} scenes, {len(nusc.sample)} samples")

    records = collect(nusc)

    # Upstream keeps only samples with a complete 3s future (plan_mask.sum() == 6).
    complete = [(d, c) for d, m, c in records if m.sum() == EGO_FUT_TS]
    print(f"complete {EGO_FUT_TS}-step futures: {len(complete)}/{len(records)}")
    if not complete:
        raise SystemExit("no complete futures found")

    deltas = np.stack([d for d, _ in complete])
    cmds = np.array([c for _, c in complete])
    abs_trajs = deltas.cumsum(axis=-2)   # kmeans_plan.py clusters absolute positions

    check_frame_convention(abs_trajs)
    for c in (CMD_RIGHT, CMD_LEFT, CMD_STRAIGHT):
        print(f"  {CMD_NAMES[c]:9s}: {int((cmds == c).sum()):5d} samples")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("vocab", "both"):
        # Exactly what upstream's shipped script does: all commands pooled -> (K, T, 2).
        vocab = cluster(abs_trajs, args.k, args.seed)
        path = outdir / f"kmeans_plan_vocab_{args.k}.npy"
        np.save(path, vocab)
        print(f"wrote {path}  shape={vocab.shape}  (upstream's shipped variant)")

    if args.mode in ("per_command", "both"):
        # The head's actual expectation: (3, K, T, 2), restoring the commented-out path.
        per_cmd = []
        for c in (CMD_RIGHT, CMD_LEFT, CMD_STRAIGHT):
            subset = abs_trajs[cmds == c]
            if len(subset) < args.k:
                print(
                    f"  WARNING: {CMD_NAMES[c]} has only {len(subset)} samples for k="
                    f"{args.k}; falling back to the pooled vocabulary for this command. "
                    "Expected on mini; do NOT train on these."
                )
                subset = abs_trajs
            per_cmd.append(cluster(subset, args.k, args.seed))
        anchors = np.stack(per_cmd, axis=0)
        path = outdir / f"kmeans_plan_{args.k}.npy"
        np.save(path, anchors)
        print(f"wrote {path}  shape={anchors.shape}  (order: right, left, straight)")
        check_normalization_compat(anchors)

        for i, c in enumerate((CMD_RIGHT, CMD_LEFT, CMD_STRAIGHT)):
            endpoints = anchors[i, :, -1, :]
            print(f"  {CMD_NAMES[c]:9s} endpoints x=[{endpoints[:,0].min():6.2f},"
                  f"{endpoints[:,0].max():6.2f}] y=[{endpoints[:,1].min():6.2f},"
                  f"{endpoints[:,1].max():6.2f}]")


if __name__ == "__main__":
    main()
