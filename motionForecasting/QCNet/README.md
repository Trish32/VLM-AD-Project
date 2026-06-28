# QCNet — pure-PyTorch QCNet (Argoverse 2) on Apple MPS

A faithful reimplementation of the official [QCNet](https://github.com/ZikangZhou/QCNet)
(CVPR 2023, AV2 marginal motion forecasting) with **no PyTorch-Geometric, torch_cluster,
torch_scatter, or PyTorch Lightning** — everything runs on `torch.device("mps")`. The
`av2` API is used only for parsing scenarios/maps.

The goal is correctness reproduction against the **official pretrained `QCNet_AV2`
checkpoint**: the 949-tensor state_dict loads with **0 missing / 0 unexpected** keys, and
validation metrics reproduce the published AV2 numbers.

## What was replaced (and how)
| Official dependency | Replacement (`utils/pyg_compat.py`) |
| --- | --- |
| `torch_geometric.nn.MessagePassing` (AttentionLayer) | explicit gather + segment-softmax + scatter-add (`layers/attention_layer.py`) |
| `torch_geometric.utils.softmax` | `segment_softmax` (stable, scatter-based) |
| `torch_scatter` | `scatter_sum` / `scatter_max` via `index_add_` / `scatter_reduce` |
| `torch_cluster.radius` / `radius_graph` | `cdist` + per-query nearest-k cap, batch-aware |
| `dense_to_sparse` / `bipartite_dense_to_sparse` / `subgraph` / `coalesce` | pure-PyTorch ports |
| `pytorch_lightning.LightningModule` | plain `nn.Module` + standalone `val.py` / `infer.py` |
| `torchmetrics.Metric` | plain sum/count accumulators (`metrics/`) |

The model modules (`modules/`), layers, losses, transform, and dataset feature extraction
are otherwise faithful copies of the official code. Runs at batch_size=1 (one scene per
forward), so PyG batching is unnecessary.

## Setup
```bash
pip install av2 pyarrow                       # data parsing only
bash scripts/download_ckpt.sh                 # -> ckpt/QCNet_AV2.ckpt (gdown)
# download AV2 val.tar/test.tar to "/Users/trish/Downloads/Argoverse 2/", then:
bash scripts/extract_val_subset.sh 150        # extract a subset for a quick check
```

## Run
```bash
# validation (reproduces published AV2 val metrics)
python val.py --root "/Users/trish/Downloads/Argoverse 2" --ckpt_path ckpt/QCNet_AV2.ckpt [--max_scenarios N]
# single-scene inference (world-frame multimodal trajectories for the focal agent)
python infer.py --root "/Users/trish/Downloads/Argoverse 2" --index 0

# fine-tune from the official checkpoint (bs=1 + gradient accumulation; cosine LR)
python finetune.py --root "/Users/trish/Downloads/Argoverse 2" --split val \
    --ckpt_path ckpt/QCNet_AV2.ckpt --max_epochs 3 --accum_steps 16 --lr 5e-5
```
Fine-tune uses the official loss (propose + refine regression NLL + winner-take-all
classification NLL) and AdamW with the official decay/no-decay split. Because the model runs
one scene per forward, the batch size is emulated with `--accum_steps`. Pass `--split train`
once `train.tar` is extracted under the AV2 root; the default `--split val` is a demo on the
extracted val scenes. The result is saved to `ckpt/QCNet_AV2_finetuned.ckpt` (reloadable by
`val.py`).

Published AV2 val (K=6): minADE 0.72, minFDE 1.25, MR 0.16, brier-minFDE 1.87.

See `bug_log.txt` for the root-caused issues found during the port.
