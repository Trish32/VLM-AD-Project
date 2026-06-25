set -e
cd /Users/trish/VLMProjects/sparse4d_vldrive
echo "=== train_v3 PLANNING (SparseDrive motion+ego) START $(date) ==="
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive python train_v3.py \
  --planning --freeze_backbone --lr 2e-4 --epochs 6 \
  --dn_groups 0 --depth_weight 0 \
  --checkpoint sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth \
  --save_dir checkpoints/train_v3_plan > train_logs/train_v3_plan.log 2>&1
echo "=== train_v3 PLANNING DONE $(date) ==="
