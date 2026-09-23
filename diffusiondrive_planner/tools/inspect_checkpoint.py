"""Inspect the official DiffusionDrive nuScenes checkpoint.

Answers the questions that decide whether the port can load it:

  1. What is the top-level structure (nesting, prefixes, tensor count)?
  2. Does it carry `plan_anchor`? If so the regenerated mini anchors are overwritten at
     load time and their quality is irrelevant to reproduction — see DESIGN.md.
  3. How do the planner-head keys map onto our modules, and what is missing on each side?

Deliberately read-only and dependency-free: no mmdet, no model construction. Run it
before attempting any real load, so a structural surprise surfaces as a report rather
than a wall of missing-key errors.

Usage:
    python diffusiondrive_planner/tools/inspect_checkpoint.py \
        diffusiondrive_planner/checkpoints/diffusiondrive_nusc_stage2.pth
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch


def torch_load(path):
    """torch.load that works on torch 1.12 as well as 2.x.

    `weights_only` only exists from torch 1.13, and the mmdet-2.x env pins 1.12. Pass it
    when available (it is the safer default on modern torch) and omit it otherwise.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap(blob):
    """Return (state_dict, path_taken) after peeling the usual wrappers."""
    if not isinstance(blob, dict):
        return blob, "<raw>"
    for key in ("state_dict", "model", "module", "ema_state_dict"):
        if key in blob and isinstance(blob[key], dict):
            return blob[key], key
    # Already a bare state dict?
    if all(isinstance(v, torch.Tensor) for v in blob.values()):
        return blob, "<bare>"
    return blob, "<unknown>"


def top_prefixes(sd, depth=2):
    counts = Counter()
    params = Counter()
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        pref = ".".join(k.split(".")[:depth])
        counts[pref] += 1
        params[pref] += v.numel()
    return counts, params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--grep", default=None, help="print keys matching this regex")
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    print(f"loading {path} ({path.stat().st_size / 1024**2:.0f} MB)...")
    blob = torch_load(path)

    if isinstance(blob, dict):
        meta = {k: type(v).__name__ for k, v in blob.items() if not isinstance(v, dict)}
        if meta:
            print("\ntop-level non-dict entries:")
            for k, t in list(meta.items())[:12]:
                val = blob[k]
                shown = val if isinstance(val, (int, float, str, bool)) else t
                print(f"  {k}: {shown}")

    sd, via = unwrap(blob)
    tensors = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    total = sum(v.numel() for v in tensors.values())
    print(f"\nstate_dict via '{via}': {len(tensors)} tensors, {total/1e6:.1f}M params")

    counts, params = top_prefixes(tensors, args.depth)
    print(f"\ntop-level modules (depth={args.depth}):")
    for pref, n in counts.most_common():
        print(f"  {pref:<44} {n:>5} tensors  {params[pref]/1e6:>8.2f}M")

    # ---- the question that matters most for our port -------------------------
    print("\n" + "=" * 70)
    anchors = [k for k in tensors if "plan_anchor" in k or "motion_anchor" in k]
    if anchors:
        print("ANCHORS ARE IN THE CHECKPOINT — regenerated values get overwritten:")
        for k in anchors:
            print(f"  {k:<56} {tuple(tensors[k].shape)}")
    else:
        print("NO plan_anchor/motion_anchor IN THE CHECKPOINT.")
        print("  => the .npy values are LIVE at inference; mini-derived anchors would")
        print("     directly degrade the reproduced metric. Regenerate from trainval.")
    print("=" * 70)

    # ---- planner-head keys ---------------------------------------------------
    head = {k: v for k, v in tensors.items()
            if re.search(r"motion_plan|diff_|plan_|traj_|time_mlp", k)}
    if head:
        print(f"\nplanner-head tensors: {len(head)}")
        groups = defaultdict(list)
        for k in head:
            m = re.search(r"(diff_layers\.\d+)", k)
            groups[m.group(1) if m else "other"].append(k)
        for g in sorted(groups, key=lambda s: (s == "other", s)):
            keys = groups[g]
            print(f"  {g:<20} {len(keys):>3} tensors")
            if g == "other":
                for k in sorted(keys)[:24]:
                    print(f"      {k:<58} {tuple(tensors[k].shape)}")

    if args.grep:
        pat = re.compile(args.grep)
        print(f"\nkeys matching /{args.grep}/:")
        for k in sorted(tensors):
            if pat.search(k):
                print(f"  {k:<62} {tuple(tensors[k].shape)}")


if __name__ == "__main__":
    main()
