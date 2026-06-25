set -e
cd /Users/trish/VLMProjects/sparse4d_vldrive
echo "=== [1/2] train_v2 (unfrozen backbone + depth) START $(date) ==="
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive python train_v2.py \
  --epochs 6 --depth_weight 0.05 --save_dir checkpoints/train_v2 \
  > train_logs/train_v2.log 2>&1
echo "=== [1/2] train_v2 DONE $(date) ==="
echo "=== [2/2] train_v3 + DN START $(date) ==="
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive python train_finetune.py \
  --model_version v3 \
  --checkpoint sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth \
  --dn_groups 5 --epochs 6 --save_dir checkpoints/finetune_v3_dn \
  > train_logs/train_v3_dn.log 2>&1
echo "=== [2/2] train_v3 + DN DONE $(date) ==="
