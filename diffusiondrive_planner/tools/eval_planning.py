"""Evaluate planning (L2 + collision) on nuScenes mini_val, on CPU.

Runs upstream's model over the val split and scores it with upstream's own
`PlanningMetric`, so the numbers are comparable to the paper's by construction rather
than by my reimplementation of the metric.

Why compare against upstream's own output, not just the paper
-------------------------------------------------------------
mini_val is 81 planning samples; the published 0.27/0.54/0.90 is on the full val split.
A gap could mean the port is wrong OR that 81 samples is a small, skewed sample. Running
the reference on the SAME 81 samples separates those two. `--planner ours` swaps in our
DiffPlanner for the head's diffusion loop so both paths can be scored identically.

Usage:
    cd diffusiondrive_planner/upstream-nusc
    PYTHONPATH=.:../.. python ../tools/eval_planning.py \
        --config projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
        --checkpoint ../checkpoints/diffusiondrive_nusc_stage2.pth
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build(config_path, checkpoint):
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(config_path)
    cfg.version = "mini"
    import projects.mmdet3d_plugin  # noqa: F401

    from mmcv.cnn.bricks.registry import ATTENTION

    from diffusiondrive_planner.attention_compat import MultiheadAttentionCompat

    ATTENTION.register_module(
        name="MultiheadFlashAttention", module=MultiheadAttentionCompat, force=True
    )

    from mmdet3d.models import build_model

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint, map_location="cpu", strict=False)
    return cfg, model.eval()


def patch_data_paths(c):
    c = dict(c)
    c["ann_file"] = "data/infos/mini/nuscenes_infos_val.pkl"
    c["data_root"] = "data/nuscenes/"
    return c


def build_loader(cfg):
    from mmdet.datasets import build_dataloader
    from mmdet3d.datasets import build_dataset

    ds = build_dataset(patch_data_paths(cfg.data.val if "val" in cfg.data else cfg.data.test))
    loader = build_dataloader(
        ds, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )
    return ds, loader


class _SchedulerAdapter:
    """Expose our TruncatedDDIM through diffusers' DDIMScheduler surface.

    Swapping this into the real head scores OUR scheduler inside the full upstream
    pipeline. `_run_block` is already bit-identical to upstream's loop, so if the metric
    is unchanged with this installed, the outer denoising loop (seed + step) is confirmed
    in situ — the one part the per-op comparison deliberately excluded.
    """

    def __init__(self, reference):
        from diffusiondrive_planner.truncated_diffusion import TruncatedDDIM

        self.ours = TruncatedDDIM()
        self.config = reference.config          # upstream reads .config.num_train_timesteps
        self.timesteps = getattr(reference, "timesteps", None)

    def set_timesteps(self, n, device=None):
        self.timesteps = torch.arange(n - 1, -1, -1, device=device)

    def add_noise(self, original_samples, noise, timesteps):
        return self.ours.add_noise(original_samples, noise, timesteps)

    def step(self, model_output, timestep, sample, **kw):
        prev = self.ours.step(model_output, int(timestep), sample)
        return type("StepOut", (), {"prev_sample": prev})()


def unwrap(obj):
    from mmcv.parallel import DataContainer

    if isinstance(obj, DataContainer):
        return unwrap(obj.data[0])
    if isinstance(obj, dict):
        return {k: unwrap(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(unwrap(v) for v in obj)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--limit", type=int, default=None, help="stop after N samples")
    ap.add_argument("--out", default="../data/planning_results.pt")
    ap.add_argument("--scheduler", choices=["upstream", "ours"], default="upstream",
                    help="'ours' swaps in TruncatedDDIM inside the real head")
    args = ap.parse_args()

    cfg, model = build(args.config, args.checkpoint)

    if args.scheduler == "ours":
        head = model.head.motion_plan_head
        head.diffusion_scheduler = _SchedulerAdapter(head.diffusion_scheduler)
        print("[patch] diffusion_scheduler -> our TruncatedDDIM")

    dataset, loader = build_loader(cfg)
    print(f"[data] {len(dataset)} val samples")

    results = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if args.limit is not None and i >= args.limit:
                break
            out = model(return_loss=False, rescale=True, **unwrap(data))
            results.extend(out)
            if (i + 1) % 10 == 0:
                print(f"[run] {i + 1}/{len(dataset)}")

    print(f"[run] collected {len(results)} results")
    torch.save(results, args.out)

    # Score with upstream's own metric, via the dataset's evaluate().
    # The eval_config drives a second dataset build inside planning_eval, so its data
    # paths need the same mini overrides.
    eval_cfg = getattr(dataset, "eval_config", None) or cfg.get("eval_config")
    if eval_cfg is not None:
        dataset.eval_config = patch_data_paths(eval_cfg)

    # The config's eval_mode is already planning-only (with_det/map/motion all False),
    # which is what we want: det/map/motion scoring needs extra artifacts we do not have
    # on mini, and planning is the metric under test.
    eval_mode = dict(cfg.get("eval_mode", {}))
    eval_mode.update(with_det=False, with_tracking=False, with_map=False, with_motion=False,
                     with_planning=True)

    work_dir = Path(args.out).resolve().parent / "eval_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset.work_dir = str(work_dir)

    print(f"\n[eval] scoring {len(results)} results with upstream PlanningMetric")
    metrics = dataset.evaluate(results, eval_mode)
    print()
    for k, v in (metrics or {}).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
