"""Run upstream's SparseDrive detector on nuScenes mini and dump planner inputs.

This is the bridge between upstream's perception stack and our ported planner. It builds
the real model from the stage-2 config, loads the official checkpoint, runs one or more
val samples, and saves exactly the tensors `DiffPlanner.forward` consumes:

    agent_feature, agent_pos, ego_pos, anchor_query, plan_anchor, metas, feature_maps

Saving them decouples the two halves: the detector runs once here, and the planner can
then be iterated on (and diffed against upstream's planner) without paying for perception
each time.

Two substitutions make this run without CUDA:

  * `MultiheadFlashAttention` -> our `MultiheadAttentionCompat`, re-registered in mmcv's
    ATTENTION registry under the same name. It is state_dict-compatible (verified in
    tests/test_attention_compat.py), so the checkpoint still loads 0/0.
  * `deformable_aggregation_ext` -> `dfa_torch` via the ops patch, already verified
    against the real CUDA kernel on a T4.

Usage:
    cd diffusiondrive_planner/upstream-nusc
    PYTHONPATH=.:../.. python ../tools/run_detector.py \
        --config projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
        --checkpoint ../checkpoints/diffusiondrive_nusc_stage2.pth \
        --out ../data/planner_inputs.pt --samples 2
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")


def patch_attention() -> None:
    """Swap the flash-attention module for the SDPA-compatible one, by registry name.

    MUST run AFTER the plugin has been imported: upstream registers its own
    `MultiheadFlashAttention` at import time without `force`, so overriding first makes
    the plugin import die with 'already registered'. Override second, with force=True.
    """
    from mmcv.cnn.bricks.registry import ATTENTION

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from diffusiondrive_planner.attention_compat import MultiheadAttentionCompat

    ATTENTION.register_module(
        name="MultiheadFlashAttention", module=MultiheadAttentionCompat, force=True
    )
    print("[patch] MultiheadFlashAttention -> MultiheadAttentionCompat (SDPA)")


def build(config_path: str, checkpoint: str, device: str):
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(config_path)
    # The config sets version='mini' then immediately overrides it to 'trainval'.
    # Force mini so anno_root resolves to data/infos/mini/, where our converter wrote.
    cfg.version = "mini"

    if cfg.get("plugin_dir"):
        import importlib
        module = cfg.plugin_dir.rstrip("/").replace("/", ".")
        importlib.import_module(module)
    else:  # the plugin is imported by path in these configs
        import projects.mmdet3d_plugin  # noqa: F401

    # Only now that upstream has registered its own modules can we override one.
    patch_attention()

    from mmdet3d.models import build_model

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    info = load_checkpoint(model, checkpoint, map_location="cpu", strict=False)
    model.to(device).eval()
    return cfg, model, info


def build_val_loader(cfg):
    from mmdet3d.datasets import build_dataset
    from mmdet.datasets import build_dataloader

    test_cfg = cfg.data.val if "val" in cfg.data else cfg.data.test
    test_cfg = dict(test_cfg)
    test_cfg["ann_file"] = "data/infos/mini/nuscenes_infos_val.pkl"
    test_cfg["data_root"] = "data/nuscenes/"
    dataset = build_dataset(test_cfg)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )
    return dataset, loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="../data/planner_inputs.pt")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"[build] {args.config}")
    cfg, model, info = build(args.config, args.checkpoint, args.device)
    missing = getattr(info, "missing_keys", []) if info else []
    print(f"[build] model ready on {args.device}")

    print("[data] building val loader")
    dataset, loader = build_val_loader(cfg)
    print(f"[data] {len(dataset)} val samples")

    captured: list[dict] = []

    # Tap the planner head's inputs rather than reimplementing perception: register a
    # forward pre-hook on the motion/planning head and record what it is handed.
    head = model.head.motion_plan_head

    # `register_forward_pre_hook(..., with_kwargs=True)` only exists from torch 2.0 and
    # this env pins 1.12, so wrap the bound method instead. Same effect, no version gate.
    original_forward = head.forward

    def recording_forward(*fargs, **fkwargs):
        captured.append({"args": fargs, "kwargs": fkwargs})
        return original_forward(*fargs, **fkwargs)

    head.forward = recording_forward

    class _Handle:
        def remove(self):
            head.forward = original_forward

    handle = _Handle()

    from mmcv.parallel import DataContainer

    def unwrap(obj):
        """Strip mmcv DataContainers.

        Normally MMDataParallel.scatter does this, but that path requires CUDA. Running
        the model directly on CPU means unwrapping by hand: DataContainer.data is a
        per-GPU list, and with samples_per_gpu=1 the batch is element 0.
        """
        if isinstance(obj, DataContainer):
            return unwrap(obj.data[0])
        if isinstance(obj, dict):
            return {k: unwrap(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(unwrap(v) for v in obj)
        return obj

    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= args.samples:
                break
            model(return_loss=False, rescale=True, **unwrap(data))
            print(f"[run] sample {i}: captured {len(captured)} head call(s)")

    handle.remove()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, out)
    print(f"[save] {out} ({out.stat().st_size / 1024**2:.1f} MB)")
    if captured:
        print("[keys]", sorted(captured[0])[:12])


if __name__ == "__main__":
    main()
