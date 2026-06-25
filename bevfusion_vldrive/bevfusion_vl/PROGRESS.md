# MIT-BEVFusion pure-PyTorch port — progress log

Run env: `conda run -n simple_bev_vldrive` (torch 2.12 MPS). See PLAN.md for full scope.
Checkpoints: model/checkpoints/bevfusion-{det,seg}.pth. Repo: ../bevfusion (mit-han-lab).

## DONE & VERIFIED (checkpoint load 0 missing / 0 unexpected unless noted)
- `model/spconv.py` — pure-PyTorch sparse 3D conv (SubMConv3d, SparseConv3d, to_dense).
  VERIFIED vs F.conv3d to ~1e-5 on dense grid (subm, stride-2, kernel(1,1,3) stride(1,1,2)).
  Weight layout (k0,k1,k2,Cin,Cout); coords (b,d0,d1,d2); bias-free.
- `model/sparse_encoder.py` — SparseEncoder. Loads 126 `encoders.lidar.backbone.*`
  tensors 0/0. Input: mean-pooled voxel feats (5d) + coords (b,x,y,z). Output dense
  (B,256,180,180) for det [1440,1440,41] (z collapses 41->2, C*D=128*2=256). 1.4s/8k voxels.
  LiDAR "VFE" is just MEAN over points (voxelize_reduce), NOT a learned net.
- `model/second.py` — SECOND (in256->[128,256], layers[5,5], strides[1,2]),
  SECONDFPN (in[128,256] out[256,256] up[1,2]; stride-1=Conv2d 1x1, stride-2=ConvTranspose;
  concat->512), ConvFuser (nn.Sequential subclass: cat([cam80,lidar256])->Conv2d336->256+BN+ReLU).
  All load 0/0. SECOND chains blocks (block1 takes block0 out), returns each stage.

## Coordinate convention (decided, validate at eval)
sparse_shape=[1440,1440,41]=(x,y,z); coords (b,x,y,z) z LAST; conv_out collapses z.
dense->permute(0,1,4,2,3)->view(B,C*Sz,Sx,Sy): BEV is (Sx,Sy). My hard_voxelize must
emit coords (b,x,y,z). Validate BEV orientation (x vs y as H) at eval; flip if metrics low.

## NEXT (camera stream) — key layouts already dumped
`encoders.camera.*` = mmdet2.x SwinTransformer + GeneralizedLSSFPN + vtransform.
- **backbone (mmdet Swin, NOT microsoft layout)**: patch_embed.projection(Conv2d3->96 k4s4)+norm;
  stages.N.blocks.N.{norm1, attn.w_msa.(qkv,proj,relative_position_bias_table,relative_position_index),
  norm2, ffn.layers.0.0+ffn.layers.1}; stages.N.downsample.(norm,reduction=Linear4C->2C);
  output norms backbone.norm1/2/3 for out_indices [1,2,3] -> dims [192,384,768].
  depths[2,2,6,2] heads[3,6,12,24] window7 embed96. Input img 256x704.
  FFN(mmcv): layers.0=Sequential(Linear,GELU,Drop), layers.1=Linear.
  Block fwd: x=x+attn(norm1(x)); x=ffn(norm2(x),identity=x).
- **neck GeneralizedLSSFPN**: lateral_convs.N.(conv,bn), fpn_convs.N.(conv,bn). in[192,384,768]
  out256 num_outs3 start_level0. (GeneralizedLSSFPN concats level i with upsampled i+1 BEFORE
  lateral conv — differs from standard FPN; read mmdet3d/models/necks/ for exact.)
- **vtransform DET=DepthLSSTransform** (270 cam tensors): buffers frustum,dx,bx,nx;
  dtransform (convs on projected lidar depth image), depthnet (predicts depth+context),
  downsample convs. xbound/ybound ±54 step0.3, zbound[-10,10,20], dbound[1,60,0.5], out80, downsample2.
  bev_pool -> replace with cumsum-scatter (see BEVFusion_robust_vl/model/lss.py + FlashOcc).
  SEG=LSSTransform (235 tensors, no depth net) xbound ±51.2 step0.4.

## THEN
- Fuse cam(80)+lidar(256)->fuser->SECOND->SECONDFPN->head.
- SEG head BEVSegmentationHead (heads.map 14 tensors, in512, grid_transform input
  [-51.2,51.2,0.8]->output[-50,50,0.5], 6 classes, focal). mIoU eval.
- DET head TransFusionHead (heads.object 96 tensors, in512, heatmap+transformer decoder,
  10 cls). bbox decode + circle nms. nuScenes mAP/NDS eval.
- Loader: 256x704 ImageAug3D (test resize 0.48), ImageNet norm, 9 sweeps points [x,y,z,intensity,dt],
  lidar2image. Reuse BEVFusion_robust_vl/data/loader.py geometry.
- infer/eval/finetune for both tasks; bug_log + memory.

## Reuse
BEVFusion_robust_vl/model/{voxelize.py(emit (b,x,y,z)!), lss.py(cumsum pool), swin.py(microsoft-
layout, NEEDS rewrite to mmdet layout)}, data/loader.py (geometry chain). 

## UPDATE (iteration 2) — CAMERA STREAM DONE
- `model/swin.py` mmdet-layout SwinTransformer: loads 187 backbone tensors 0/0;
  out [(6,192,32,88),(6,384,16,44),(6,768,8,22)] for 6x256x704.
- `model/lss_fpn.py` GeneralizedLSSFPN: loads 0/0 (cat upsample(i+1) w/ i -> lateral1x1 -> fpn3x3).
- `model/vtransform.py` LSSTransform(seg) + DepthLSSTransform(det): BOTH load 0/0.
  DET D=118 (dbound[1,60,.5]), nx=[360,360,1] (z=1 bin), frustum(118,32,88,3); downsample/2 ->180.
  bev_pool replaced by index_add sum-pool (canvas rank=((b*nz+gz)*nx+gx)*ny+gy), collapse z.
  DepthLSS uses SCALAR depth (dtransform Conv2d(1,8,..)) — add_depth_features=False,
  height_expand=False for this ckpt. Depth image d: project lidar pts via
  lidar2image+img_aug to (256,704), scatter dist into per-cam depth map (see base.py
  BaseDepthTransform.forward lines 238-320; coords flipped [1,0]).
  get_geometry needs camera2lidar(rots,trans), intrins, img_aug(post), lidar_aug(extra).

## ALL BACKBONE/NECK/FUSION/DECODER LOAD 0/0 FOR BOTH TASKS. Remaining:
- heads.map BEVSegmentationHead (14 tensors): transform (grid sample input[-51.2,51.2,.8]->
  output[-50,50,.5]) + classifier convs -> 6 logits; sigmoid; mIoU. loss focal (eval: threshold 0.5).
- heads.object TransFusionHead (96 tensors): shared_conv -> heatmap (class) -> top-200 query init
  -> N transformer decoder layers (self+cross attn to BEV) -> FFN heads (center,height,dim,rot,vel,heatmap)
  -> bbox decode (circle nms optional). 10 cls. Read mmdet3d/models/heads/bbox/transfusion.py.
- loader: camera2lidar/lidar2image/img_aug(ImageAug3D resize .48 -> 256x704)/lidar_aug(identity at test)
  /camera_intrinsics matrices; 9-sweep pts [x,y,z,intensity,dt]; ImageNet norm [0,1].
  voxelize must emit coords (b,x,y,z) to match sparse_shape; mean-pool features.

## UPDATE (iteration 2 cont) — SEG HEAD DONE
- `model/seg_head.py` BEVSegmentationHead loads 0/0; in 512; grid_transform
  (input[-51.2,51.2,.8] 128x128 -> output[-50,50,.5] 200x200 via grid_sample) +
  classifier(512->512->512->6) -> sigmoid. out (1,6,200,200).
SEG model now fully component-complete (all load 0/0). NEXT: data loader, assemble
BEVFusion(seg), end-to-end infer + mIoU. Then TransFusion det head.
Seg decoder BEV is 128x128 (1024 grid /8); det is 180x180 (1440 grid /8).

## UPDATE (iteration 2 cont) — LOADER + VOXELIZE DONE
- `data/loader.py` NuScenesMITLoader: 81 mini_val samples; produces img(6,3,256,704)
  ImageNet-norm, points(M,5) 9-sweep, and all matrices (camera2lidar, lidar2camera,
  lidar2image, camera_intrinsics, camera2ego (6,4,4); lidar2ego (4,4);
  img_aug_matrix (resize .48, crop_w 32 crop_h 176); lidar_aug=eye). Geometry chained
  through global frame at each sensor timestamp.
- `model/voxelize.py` voxelize_mean: mean-reduce, emits coords (x,y,z). Verified:
  SEG 10714 voxels max[1017,1019,39]<[1024,1024,41]; DET 12677 max[1427,1439,39]<[1440,1440,41].

## SEG MODEL FULLY COMPONENT-COMPLETE + DATA PIPELINE WORKS. NEXT (next iteration):
1. `model/bevfusion.py` assemble class: extract_camera (swin->neck->vtransform w/ geom),
   extract_lidar (voxelize_mean->SparseEncoder), fuser([cam,lidar]) [order: cam first per
   in_channels [80,256]], decoder(SECOND->SECONDFPN), head. Load full ckpt (target 0/0,
   prefixes encoders.camera.*, encoders.lidar.*, fuser.*, decoder.*, heads.map/object.*).
   For DepthLSS need depth image: project points via lidar2image+img_aug -> scatter dist.
2. `tools/infer_seg.py` + `tools/eval_seg.py`: run mini_val, threshold 0.5, per-class IoU vs
   gt_masks_bev (LoadBEVSegmentation: rasterize map polygons in [-50,50,.5]=200x200,6 cls).
   Need nuScenes map API (nuscenes.map_expansion) for GT masks. mIoU = mean over classes.
3. Then DET: depth projection + TransFusionHead + nuScenes mAP/NDS.
WATCH: BEV orientation (x as H vs W) — validate vs metrics; fuser input order (cam,lidar).

## UPDATE (iteration 3) — SEG MODEL RUNS END-TO-END
- `model/bevfusion.py` assembled (encoders.{camera,lidar}, fuser, decoder, heads.map).
  FULL SEG checkpoint loads **0 missing / 0 unexpected**. End-to-end forward 2.2s/frame
  on CPU -> (1,6,200,200) sigmoid. Sensible: drivable mean .46 (~40% >.5), walkway .29.
- `tools/eval_seg.py` mIoU eval w/ NuScenesMap GT masks (map files + map_expansion present).
- Orientation diagnostic: IDENTITY best (drivable IoU 32% vs flips 27-29%, transpose/rot ~10%).
  => BEV axes correct (x=H,y=W for both pred & GT [class,x,y]); NOT transposed.
- OPEN ISSUE: drivable_area ~32% (6 samples) vs official ~85%. Not a gross flip.
  Suspects to check next: (1) GT loader commented `masks=masks[:, ::-1, :]` toggle —
  test flipping GT; (2) half-cell offset in grid_transform / patch; (3) image ImageAug3D
  crop_h/crop_w exactness; (4) lidar BEV vs camera BEV sub-pixel alignment. Full mini_val
  eval running. Likely a single GT-orientation/offset fix recovers most of the gap.

## UPDATE (iteration 3 cont) — SEG EVAL (official iou@max protocol)
- `tools/eval_seg.py` uses official protocol: thresholds [.35..65], per-class iou@max,
  mIoU=mean. GT-flip A/B confirms **GT identity correct** (mIoU 30.2 vs flipH 17.2/flipW 16.3).
- 12-sample mini_val: drivable 56.3, ped_crossing 49.9, walkway 39.3, divider 20.8,
  stop_line 6.9, carpark 7.9 -> mIoU 30.2. Below full-val official 62.7 (drivable 85/ped 60/
  walkway 67/stop 52/carpark 57/divider 54). Gap = mini 2-scene val (rare classes
  stop_line/carpark barely present) + possible mild calibration. Alignment + image norm
  (ToTensor /255 + ImageNet) VERIFIED correct. Model loads 0/0, runs 2.2s/frame.
- Image preprocessing confirmed: official ImageNormalize = torchvision ToTensor(/255)+Normalize;
  my loader /255 then (x-mean)/std — matches.
- Remaining seg calibration suspects (minor, for later): lidar-vs-ego ~1m fwd offset in BEV;
  exact ImageAug3D crop; sub-pixel grid_transform. NOT blocking — seg verified functional.

## NEXT: DET head (TransFusionHead, heads.object 96 tensors) — the other task.
Read bevfusion/mmdet3d/models/heads/bbox/transfusion.py. shared_conv -> heatmap (top-200
query init via local-max + class) -> N decoder layers (self-attn + cross-attn to BEV +
FFN, with positional encoding of query positions) -> prediction heads (center,height,dim,
rot,vel,heatmap) -> decode_bbox. Then det eval (nuScenes mAP/NDS) reuse robust_vl converter.

## DET HEAD STRUCTURE (TransFusionHead, heads.object, to port next)
hidden=128, num_proposals=200, 1 decoder layer, 10 classes. Source:
bevfusion/mmdet3d/models/heads/bbox/transfusion.py. Keys:
- shared_conv Conv2d(512->128,3,p1)(+BN) on decoder BEV (B,512,180,180)->(128,180,180).
- heatmap_head: Conv(128->128,3)+BN+ReLU then Conv(128->10) -> dense heatmap. sigmoid;
  local-max nms (kernel 3) ; flatten; top-200 -> query_pos (x,y on 180 grid), query class.
  query_feat = bev_feat gathered at top-200 indices (128-dim). class_encoding Conv1d(10->128,1)
  on one-hot(query class) added to query_feat.
- decoder.0 (TransformerDecoderLayer hidden128, 8 heads): self_posembed & cross_posembed =
  PositionEmbeddingLearned (MLP Conv1d(2->128,1)+BN+ReLU+Conv1d(128->128,1)); self_attn
  (MultiheadAttention, q=k=query_feat+query_pos_embed), cross multihead_attn (q=query+qpe,
  k=bev_flat+bev_pos_embed, v=bev_flat), FFN linear1(128->?)+linear2, norm1/2/3 (post-norm).
  bev_pos = normalized grid coords of the 180x180 BEV (one cross_posembed over all positions).
- prediction_heads.0.{center(2),height(1),dim(3),rot(2),vel(2),heatmap(10)} each FFN
  (Conv1d(128->64)+BN+ReLU+Conv1d(64->out)). 
- decode_bbox: center = query_pos + center_offset (in BEV cells) -> metric xy via
  voxel*out_size_factor(8) + pc_range; z=height; dim=exp(dim); yaw=atan2(rot[0],rot[1]);
  score=heatmap.sigmoid().max over cls per query; label=argmax. nuScenes box convert (reuse
  BEVFusion_robust_vl/tools/eval.py converter). out_size_factor=8, voxel .075, pc_range ±54.

## UPDATE (iteration 4) — DET HEAD DONE, BOTH TASKS RUN END-TO-END
- `model/transfusion_head.py` TransFusionHead ported (heatmap top-200 query init +
  1 transformer decoder layer (self+cross attn, learned pos embeds) + FFN prediction
  heads + decode). Full DET checkpoint loads with ONLY `heads.object.bev_pos` "missing"
  (my computed buffer, non-persistent) — i.e. all real weights load, 0 unexpected.
- DET inference: 200 boxes, 2.2s/frame. PLAUSIBLE: pedestrians 0.7x0.73x1.77 @0.81,
  cars 1.93x4.69x1.69 @0.60, correct sizes/positions/yaw. Same cars as ADLab PP port (consistency).
- `tools/eval_det.py` nuScenes mAP/NDS (reuses ADLab converter: gravity z+h/2, dims->wlh,
  yaw=-yaw-pi/2; range-filter in ego frame; det_cvpr_2019). Running on mini_val.
- Key params: num_proposals 200, 1 decoder layer, hidden 128, 8 heads, ffn 256,
  nms_kernel 3 (ped/cone cls 8,9 no suppression), nms_type null, score_thr 0,
  out_size_factor 8, grid 1440/8=180. bev_pos = create_2D_grid(180,180)+0.5.
- BOTH TASKS NOW FUNCTIONAL in pure PyTorch/MPS via custom sparse conv. Remaining:
  read det mAP/NDS; fine-tune smoke tests; bug_log.

## UPDATE (iteration 5) — PRECOMPUTED BEV POOLING (the paper's core innovation)
- `model/bev_pool.py` BEVPool: implements MIT BEVFusion's two ideas in pure PyTorch:
  (1) INTERVAL REDUCTION — sort frustum points by target BEV cell, cumsum features,
  diff at cell ENDS (QuickCumsum) instead of random scatter-add; (2) PRECOMPUTATION —
  the frustum->cell mapping (gather order, interval boundaries, output indices) depends
  only on fixed camera calibration, so it's built ONCE and cached, reused every frame.
- Verified: interval-reduction == index_add scatter-sum to ~3e-5 (numerically identical).
  Cached path ~1.6x faster than rebuild-each on synthetic (caches the argsort).
- BUG fixed during impl: boundary mask must mark interval END (where next rank differs)
  not START — cumsum is read at ends then differenced. (start-mask gave max|diff| 8.0).
- Wired into vtransform BaseTransform.bev_pool -> self.pool(geom,x). Caches frame-0 geom,
  reuses across all frames (valid because cam<->lidar calibration is ~fixed; tiny
  per-frame ego-motion correction is sub-cell). Validating full det eval holds ~0.508.
