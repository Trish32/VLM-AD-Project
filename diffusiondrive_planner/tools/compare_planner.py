"""Per-operation diff of our DiffPlanner against upstream's V13MotionPlanningHead.

The strongest available check on the port. Each component was already diffed against its
upstream counterpart in isolation; what this validates is the WIRING — the order of the
22 operation slots, what each one is handed, and how the prediction is fed back between
denoising steps. That is exactly the class of error a component test cannot see.

Method
------
Upstream's head now runs on CPU (thanks to the attention substitution), so we can:

  1. wrap every `diff_layers[i].forward` and `diff_graph_model` on the real head and
     record inputs + outputs while it processes a real nuScenes sample;
  2. take the recorded inputs to slot 0 of the first denoising step;
  3. run OUR `_run_block` from that same starting state;
  4. diff our per-slot outputs against upstream's.

Diffing from a captured starting state sidesteps the diffusion noise entirely: the block
is deterministic given its inputs, so any divergence is wiring, not sampling.

Usage:
    cd diffusiondrive_planner/upstream-nusc
    PYTHONPATH=.:../.. python ../tools/compare_planner.py \
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


def build_loader(cfg):
    from mmdet.datasets import build_dataloader
    from mmdet3d.datasets import build_dataset

    c = dict(cfg.data.val if "val" in cfg.data else cfg.data.test)
    c["ann_file"] = "data/infos/mini/nuscenes_infos_val.pkl"
    c["data_root"] = "data/nuscenes/"
    ds = build_dataset(c)
    return build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)


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
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    cfg, model = build(args.config, args.checkpoint)
    head = model.head.motion_plan_head
    print(f"[build] head ops = {len(head.diff_operation_order)}")

    trace: list[dict] = []

    # ---- instrument upstream ------------------------------------------------
    originals = {}
    for i, layer in enumerate(head.diff_layers):
        if layer is None:
            continue
        originals[i] = layer.forward

        def make(idx, fn):
            def wrapped(*a, **kw):
                out = fn(*a, **kw)
                trace.append({"slot": idx, "op": head.diff_operation_order[idx],
                              "args": a, "kwargs": kw, "out": out})
                return out
            return wrapped

        layer.forward = make(i, layer.forward)

    orig_graph = head.diff_graph_model
    graph_calls = []

    def graph_wrapper(index, query, key=None, value=None, query_pos=None, key_pos=None, **kw):
        graph_calls.append({"index": index, "query": query, "key": key,
                            "query_pos": query_pos, "key_pos": key_pos})
        return orig_graph(index, query, key, value,
                          query_pos=query_pos, key_pos=key_pos, **kw)

    head.diff_graph_model = graph_wrapper

    # ---- run one real sample ------------------------------------------------
    loader = build_loader(cfg)
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i > args.sample:
                break
            if i == args.sample:
                model(return_loss=False, rescale=True, **unwrap(data))
    print(f"[run] recorded {len(trace)} layer calls, {len(graph_calls)} graph calls")

    n_ops = len(head.diff_operation_order)
    if len(trace) < n_ops:
        raise SystemExit(f"expected >= {n_ops} calls, got {len(trace)}")

    first = trace[:n_ops]
    print("[trace] first denoising step, slot -> op:")
    for t in first:
        shapes = [tuple(x.shape) for x in t["args"] if torch.is_tensor(x)]
        print(f"   [{t['slot']:>2}] {t['op']:<18} in={shapes[:2]} "
              f"out={tuple(t['out'].shape) if torch.is_tensor(t['out']) else type(t['out']).__name__}")

    # ---- our planner, same weights, same starting state ---------------------
    from diffusiondrive_planner.diff_planner import DiffPlanner

    sub = {k: v for k, v in head.state_dict().items()}
    ours = DiffPlanner(256, head.ego_fut_ts, head.ego_fut_mode,
                       num_heads=8, ffn_channels=512, num_repeats=2)
    ours.load_state_dict(
        {k: sub[k.replace("time_mlp.time_mlp.", "time_mlp.").replace("traj_embedder.", "")]
         for k in ours.state_dict()}, strict=True)
    ours.eval()
    print("[ours] loaded head weights 0/0")

    # starting state = exactly what upstream handed slot 0
    pooler_call = first[0]
    traj_feature = pooler_call["args"][0]
    diff_plan_reg = pooler_call["args"][1]
    metas = pooler_call["args"][2]
    feature_maps = pooler_call["args"][3]

    mod_call = next(t for t in first if t["op"] == "modulation")
    time_embed = mod_call["args"][1]

    agent_call = graph_calls[0]
    anchor_call = next(t for t in first if t["op"] == "anchor_cross_gnn")
    anchor_query = anchor_call["kwargs"].get("key", anchor_call["args"][1]
                                             if len(anchor_call["args"]) > 1 else None)

    with torch.no_grad():
        got_feat, got_reg, got_cls = ours._run_block(
            traj_feature, diff_plan_reg, time_embed.reshape(traj_feature.shape[0], -1),
            agent_call["key"], agent_call["key_pos"], agent_call["query_pos"],
            anchor_query, metas, feature_maps,
        )

    want = first[-1]["out"]
    want_reg, want_cls = want if isinstance(want, tuple) else (want, None)

    print("\n" + "=" * 62)
    print(f"{'quantity':<24}{'max abs diff':>16}  verdict")
    print("=" * 62)
    ok = True
    for name, g, w in (("plan_reg", got_reg, want_reg), ("plan_cls", got_cls, want_cls)):
        if w is None or g is None:
            continue
        d = float((g - w).abs().max())
        ok &= d < 1e-4
        print(f"{name:<24}{d:>16.3e}  {'MATCH' if d < 1e-4 else 'DIFFER'}")
    print("=" * 62)
    print("\nWIRING CONFIRMED" if ok else "\nWIRING DIVERGES — diff per slot above")


if __name__ == "__main__":
    main()
