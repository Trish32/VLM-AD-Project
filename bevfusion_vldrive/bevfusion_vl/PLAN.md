# Plan: Pure-PyTorch MPS reproduction of MIT-HAN-LAB BEVFusion (det + seg)

## Context
Reproduce **mit-han-lab/bevfusion** (ICRA 2023, multi-task LC fusion) in pure
PyTorch on MPS (no mmcv/mmdet3d/spconv/bev_pool/CUDA), verifying inference /
eval / fine-tune for BOTH tasks against the shipped checkpoints:
- `bevfusion_vl/model/checkpoints/bevfusion-det.pth` (582 tensors) — 3D detection
- `bevfusion_vl/model/checkpoints/bevfusion-seg.pth` (465 tensors) — BEV map segmentation

Official val metrics: det mAP ~68.5 / NDS ~71.4; seg mIoU ~62.9 (full nuScenes val).
On nuScenes **mini** we verify pipeline correctness + sane metrics on present
classes (same caveat as the BEVFusion-PP / Sparse4D ports). Repo at
`/Users/trish/VLMProjects/bevfusion_vldrive/bevfusion` (the earlier ADLab repo
is now `BEVFusion_robust*`). Run env `conda run -n simple_bev_vldrive` (torch
2.12 MPS); `PYTORCH_ENABLE_MPS_FALLBACK=1`. Data `/Users/trish/Downloads/nuScenes_miniV1.0`.

## Architecture (from checkpoint + configs)
Checkpoint groups (shared unless noted):
- `encoders.camera` (det 270 / seg 235): SwinTransformer(out_indices [1,2,3]) ->
  GeneralizedLSSFPN(->256) -> vtransform. DET=DepthLSSTransform(uses lidar depth),
  SEG=LSSTransform. Both -> 80-ch camera BEV. bev_pool replaced by cumsum-scatter.
- `encoders.lidar` (126, identical arch): **SparseEncoder** (VoxelNet). in=5,
  base16, out128, 4 stages basicblock [[16,16,32],[32,32,64],[64,64,128],[128,128]],
  conv_out SparseConv3d k(1,1,3) s(1,1,2). DET sparse_shape [1440,1440,41] (voxel
  0.075, range ±54); SEG [1024,1024,41] (voxel 0.1, range ±51.2). Output dense
  (N, 256, H, W) [C*D, 128*2].
- `fuser` (6): ConvFuser Conv2d(80+256->256,3,p1,bias=F)+BN+ReLU.
- `decoder.backbone` (72): SECOND in256 out[128,256] layers[5,5] strides[1,2].
- `decoder.neck` (12): SECONDFPN in[128,256] out[256,256] up[1,2]
  **use_conv_for_no_stride=true** (stride-1 = Conv2d, NOT deconv — differs from PP).
  -> concat 512.
- DET `heads.object` (96): **TransFusionHead** (heatmap + transformer decoder,
  in 512, 10 cls). SEG `heads.map` (14): **BEVSegmentationHead** (in 512 ->
  grid_transform -> 6 map classes, focal). 

object_classes (det): car,truck,construction_vehicle,bus,trailer,barrier,
motorcycle,bicycle,pedestrian,traffic_cone.
map_classes (seg): drivable_area,ped_crossing,walkway,stop_line,carpark_area,divider.
Image: 6 cams 256x704, ImageAug3D (test resize 0.48, no flip/rot), ImageNet
normalize (mean .485,.456,.406 / std .229,.224,.225). Points: 9 sweeps,
5-dim [x,y,z,intensity,dt] (NOTE: MIT keeps intensity, unlike ADLab PP).

## Keystone: pure-PyTorch sparse 3D conv (`model/spconv.py`)
Replace spconv with rulebook gather/scatter:
- coords (N,4) [b,z,y,x]; hash = b*Z*Y*X + z*Y*X + y*X + x via int64.
- **SubMConv3d**: output sites == input sites. For each of 27 kernel offsets,
  find input idx whose (coord+offset) is an active site (hashmap lookup);
  gather, matmul kernel[off], scatter-add to output. weight layout (kx,ky,kz,Cin,Cout).
- **SparseConv3d (stride s, pad p)**: out coord = (in + p - (k-1)/2)//s dedup
  (build via the official spconv "get_indice_pairs" downsample rule: out =
  floor((in+pad-dilation*(k-1)... )); simplest faithful: for each active in
  site and each kernel offset, out = (in+pad-off)//s if divisible & in range;
  collect unique out coords, build pair map, gather/matmul/scatter.
- conv_out k(1,1,3) s(1,1,2): only z reduces. Then .dense() onto sparse_shape,
  permute (N,C,H,W,D)->(N,C*D,H,W).
- Validation: build the same module in `uniad2.0` IF importable, else trust
  faithful port + checkpoint key match + downstream eval (proven methodology).
Perf: mini has ~100k voxels; rulebook built per indice_key once. Submanifold
rulebook is shared across the two convs in a basicblock (same indice_key=subm*).

## Reuse from `BEVFusion_robust_vl` (PP port)
- `voxelize.py` hard_voxelize (adapt: keep 5 dims incl intensity).
- `swin.py` Swin (adapt: out_indices [1,2,3]; check mmdet key names vs cbnet;
  `convert_weights` maps timm->mmdet). NOTE MIT uses plain SwinTransformer
  (mmdet style: `stages`/`patch_embed.projection`/`norm0..` may differ — verify
  against checkpoint keys before reusing).
- `data/loader.py` geometry (lidar2image chain) — adapt to ImageAug3D 256x704.
- LSS cumsum pooling pattern from `lss.py`.

## Target layout (`bevfusion_vl/`)
```
config.py                  # det + seg hyperparams
model/ spconv.py sparse_encoder.py swin.py lss_fpn.py vtransform.py
       fuser.py second.py transfusion_head.py seg_head.py bevfusion.py checkpoint.py
data/  loader.py finetune_loader.py
tools/ infer_det.py eval_det.py infer_seg.py eval_seg.py train_finetune.py
bug_log.txt
```

## Phases
1. (done) scope + checkpoint introspection + config scaffold.
2. Pure-PyTorch sparse conv + SparseEncoder; verify load 126 tensors + dense out.
3. Camera: Swin + GeneralizedLSSFPN + LSS/DepthLSS vtransform; verify cam keys.
4. Fuser + SECOND/SECONDFPN decoder; assemble shared core.
5. SEG head + loader + infer_seg + eval_seg (mIoU). [simpler head first]
6. DET head (TransFusion) + infer_det + eval_det (mAP/NDS).
7. Fine-tune smoke tests (both); bug_log + memory.

## Verification
- per-module strict checkpoint key match (target 0 missing/0 unexpected like PP).
- seg: mIoU over map_classes on mini_val.
- det: nuScenes mAP/NDS on mini_val (expect rare-class 0 AP from mini imbalance).
- finetune: loss decreases, checkpoint round-trips.

## Risks
- Sparse conv fidelity + speed (keystone). Dense fallback infeasible (1440²×41).
- TransFusionHead complexity (transformer decoder, query init from heatmap).
- DepthLSSTransform needs a lidar-depth map per camera (project points -> image).
- MIT Swin key layout may differ from the ADLab cbnet Swin — verify first.
