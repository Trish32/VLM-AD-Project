cd /Users/trish/VLMProjects/sparse4d_vldrive
CKPT_V2=sparse4d_vl/model/checkpoints/sparse4dv2_r50_HInf_256x704.pth
CKPT_V3=sparse4d_vl/model/checkpoints/sparse4dv3_r50.pth
EV="--eval-set mini_val"

# --- Wait for the training job to finish (DONE marker) or its procs to die ---
echo "[watch] waiting for training to complete... $(date)"
while true; do
  if grep -q "train_v3 + DN DONE" train_logs/run_both.log 2>/dev/null; then break; fi
  if ! pgrep -f "train_v2.py|train_finetune.py" >/dev/null 2>&1 \
     && grep -q "train_v3 + DN START" train_logs/run_both.log 2>/dev/null; then
     echo "[watch] training procs gone"; break; fi
  sleep 60
done
echo "[watch] training finished, starting eval $(date)"

run_eval () {  # $1=tag $2=version $3=checkpoint
  echo ""; echo "########## EVAL: $1 ##########"
  if [ ! -f "$3" ]; then echo "  (missing checkpoint $3 — skipped)"; return; fi
  PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n simple_bev_vldrive \
    python sparse4d_vl/tools/eval.py --version $2 --checkpoint "$3" $EV
}

run_eval "v2 BASELINE (pretrained)"      v2 "$CKPT_V2"
run_eval "v2 TRAINED (unfrozen+depth)"   v2 checkpoints/train_v2/epoch_05.pt
run_eval "v3 BASELINE (pretrained)"      v3 "$CKPT_V3"
run_eval "v3 TRAINED (+ DN)"             v3 checkpoints/finetune_v3_dn/epoch_05.pt
echo ""; echo "[watch] ALL EVALS DONE $(date)"
