set -e
cd /Users/trish/VLMProjects/sparse4d_vldrive
CK=sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth
echo "=== [1/2] train_v3 FULL (unfrozen + DN=5 + quality + depth) START $(date) ==="
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive python train_v3.py \
  --checkpoint $CK --epochs 6 --dn_groups 5 --depth_weight 0.05 \
  --save_dir checkpoints/train_v3_full > train_logs/train_v3_full.log 2>&1
echo "=== [1/2] train_v3 FULL DONE $(date) ==="
echo "=== [2/2] train_v3 CONTROL (no DN: dn_groups=0) START $(date) ==="
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive python train_v3.py \
  --checkpoint $CK --epochs 6 --dn_groups 0 --depth_weight 0.05 \
  --save_dir checkpoints/train_v3_nodn > train_logs/train_v3_nodn.log 2>&1
echo "=== [2/2] train_v3 CONTROL DONE $(date) ==="
