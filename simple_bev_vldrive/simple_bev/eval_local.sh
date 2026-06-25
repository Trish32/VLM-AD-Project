#!/usr/bin/env bash
set -euo pipefail

# Symlink the mini dataset into the path structure compile_data() expects:
#   data_dir/<dset>/  →  the directory that contains v1.0-mini/, samples/, etc.
mkdir -p /tmp/nuscenes_simple_bev
ln -sfn /Users/trish/Downloads/nuScenes_miniV1.0 /tmp/nuscenes_simple_bev/mini

# Run from the simple_bev/ directory so relative imports resolve correctly
cd "$(dirname "$0")"

conda run -n simple_bev_vldrive python eval_nuscenes.py \
  --data_dir=/tmp/nuscenes_simple_bev \
  --dset=mini \
  --init_dir="/Users/trish/VLMProjects/simple_bev_vldrive/simple_bev/checkpoints/8x5_5e-4_rgb12_22:43:46" \
  --batch_size=1 \
  --res_scale=2 \
  --nworkers=2 \
  --device_ids="[0]"
