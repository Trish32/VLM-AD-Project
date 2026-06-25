python tools/infer.py \
    --dataroot /Users/trish/Downloads/nuScenes_miniV1.0 \
    --checkpoint model/checkpoints/bevformer_tiny_fp16_epoch_24.pth \
    --scene 0 --max-frames 10 --score-thr 0.25 --out-dir bev_outputs --save-cams
