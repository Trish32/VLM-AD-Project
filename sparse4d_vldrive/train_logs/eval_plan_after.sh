cd /Users/trish/VLMProjects/sparse4d_vldrive
while true; do
  if grep -q "PLANNING DONE" train_logs/run_plan.log 2>/dev/null; then break; fi
  if ! pgrep -f "train_v3.py" >/dev/null 2>&1 && grep -q "PLANNING START" train_logs/run_plan.log 2>/dev/null; then break; fi
  sleep 60
done
echo "[watch] training done, eval $(date)"
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive python sparse4d_vl/tools/eval_motion.py \
  --checkpoint checkpoints/train_v3_plan/epoch_05.pt --eval-set mini_val
echo "[watch] EVAL DONE $(date)"
