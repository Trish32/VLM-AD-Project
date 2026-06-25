set -e
cd /Users/trish/VLMProjects/sparse4d_vldrive
echo "=== train_v3 PLANNING-ONLY (detector frozen) START $(date) ==="
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive python train_v3.py \
  --planning --planning_only --lr 2e-4 --epochs 6 --dn_groups 0 --depth_weight 0 \
  --checkpoint sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth \
  --save_dir checkpoints/train_v3_plan2 > train_logs/train_v3_plan2.log 2>&1
echo "=== DONE $(date) ==="
echo "[eval] det (mAP) + motion/plan after planning-only training"
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive python sparse4d_vl/tools/eval.py \
  --version v3 --checkpoint checkpoints/train_v3_plan2/epoch_05.pt --eval-set mini_val 2>&1 | grep -E "mAP :|NDS :"
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive python sparse4d_vl/tools/eval_motion.py \
  --checkpoint checkpoints/train_v3_plan2/epoch_05.pt --eval-set mini_val 2>&1 | grep -E "MOTION|PLANNING|minADE|minFDE|brier|miss|L2|col"
echo "=== ALL DONE $(date) ==="
