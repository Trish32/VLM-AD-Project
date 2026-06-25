cd /Users/trish/VLMProjects/sparse4d_vldrive
CK=sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth
EV="--eval-set mini_val"
echo "[watch] waiting for v3 compare training... $(date)"
while true; do
  if grep -q "CONTROL DONE" train_logs/run_v3_compare.log 2>/dev/null; then break; fi
  if ! pgrep -f "train_v3.py" >/dev/null 2>&1 && grep -q "CONTROL START" train_logs/run_v3_compare.log 2>/dev/null; then
     echo "[watch] training procs gone"; break; fi
  sleep 60
done
echo "[watch] training done, evaluating $(date)"
run_eval () {
  echo ""; echo "########## EVAL: $1 ##########"
  if [ ! -f "$2" ]; then echo "  (missing $2 — skipped)"; return; fi
  PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
    python sparse4d_vl/tools/eval.py --version v3 --checkpoint "$2" $EV 2>&1 | grep -E "ckpt|mAP :|NDS :"
}
run_eval "v3 BASELINE (pretrained)"               "$CK"
run_eval "v3 FULL (unfrozen+DN+quality+depth)"    checkpoints/train_v3_full/epoch_05.pt
run_eval "v3 CONTROL (no DN, same recipe)"        checkpoints/train_v3_nodn/epoch_05.pt
echo ""; echo "[watch] ALL V3 EVALS DONE $(date)"
