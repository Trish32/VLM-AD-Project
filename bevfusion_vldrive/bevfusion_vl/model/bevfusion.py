"""
BEVFusion (MIT) full assembly — pure PyTorch / MPS. Module nesting mirrors the
checkpoint: encoders.{camera.{backbone,neck,vtransform}, lidar.backbone},
fuser, decoder.{backbone,neck}, heads.{map|object}.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .swin import SwinTransformer
from .lss_fpn import GeneralizedLSSFPN
from .vtransform import LSSTransform, DepthLSSTransform
from .sparse_encoder import SparseEncoder
from .second import SECOND, SECONDFPN, ConvFuser
from .seg_head import BEVSegmentationHead
from .voxelize import voxelize_mean


class BEVFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_size = (cfg.IMAGE_SIZE[0] // 8, cfg.IMAGE_SIZE[1] // 8)  # (32,88)

        backbone = SwinTransformer(embed_dims=96, depths=(2, 2, 6, 2),
                                   num_heads=(3, 6, 12, 24), window_size=7,
                                   mlp_ratio=4, out_indices=(1, 2, 3))
        neck = GeneralizedLSSFPN(in_channels=(192, 384, 768), out_channels=256, num_outs=3)
        vt_cls = DepthLSSTransform if cfg.VT_TYPE == 'DepthLSSTransform' else LSSTransform
        vtransform = vt_cls(cfg.VT_IN, cfg.VT_OUT, cfg.IMAGE_SIZE, feat_size,
                            cfg.XBOUND, cfg.YBOUND, cfg.ZBOUND, cfg.DBOUND,
                            cfg.VT_DOWNSAMPLE)

        self.encoders = nn.ModuleDict({
            'camera': nn.ModuleDict({'backbone': backbone, 'neck': neck, 'vtransform': vtransform}),
            'lidar': nn.ModuleDict({'backbone': SparseEncoder(
                in_channels=5, sparse_shape=tuple(cfg.SPARSE_SHAPE))}),
        })
        self.fuser = ConvFuser(in_channels=(80, 256), out_channels=256)
        self.decoder = nn.ModuleDict({'backbone': SECOND(), 'neck': SECONDFPN()})

        self.heads = nn.ModuleDict()
        if cfg.TASK == 'seg':
            self.heads['map'] = BEVSegmentationHead(
                512, dict(input_scope=cfg.SEG_INPUT_SCOPE,
                          output_scope=cfg.SEG_OUTPUT_SCOPE), cfg.CLASSES)
        else:
            from .transfusion_head import TransFusionHead
            self.heads['object'] = TransFusionHead(cfg)

    # ------------------------------------------------------------------
    def extract_camera(self, img, points, frame):
        B, N, C, H, W = img.shape
        x = img.view(B * N, C, H, W)
        x = self.encoders['camera']['backbone'](x)
        x = self.encoders['camera']['neck'](x)
        x = x[0] if isinstance(x, (list, tuple)) else x
        x = x.view(B, N, x.shape[1], x.shape[2], x.shape[3])
        return self.encoders['camera']['vtransform'](
            x, points, frame['camera2lidar'], frame['camera_intrinsics'],
            frame['img_aug_matrix'], frame['lidar_aug_matrix'], frame['lidar2image'])

    def extract_lidar(self, points):
        feats_list, coords_list = [], []
        for b, p in enumerate(points):
            f, c = voxelize_mean(p, self.cfg.VOXEL_SIZE, self.cfg.POINT_CLOUD_RANGE,
                                 self.cfg.MAX_NUM_POINTS, self.cfg.MAX_VOXELS)
            feats_list.append(f)
            bcol = torch.full((c.shape[0], 1), b, dtype=torch.long, device=c.device)
            coords_list.append(torch.cat([bcol, c], dim=1))   # (b,x,y,z)
        feats = torch.cat(feats_list, 0)
        coords = torch.cat(coords_list, 0)
        return self.encoders['lidar']['backbone'](feats, coords, len(points))

    @torch.no_grad()
    def forward(self, frame):
        img = frame['img'].unsqueeze(0) if frame['img'].dim() == 4 else frame['img']
        points = [frame['points']]
        # batched matrices: add batch dim
        for k in ['camera2lidar', 'camera_intrinsics', 'img_aug_matrix', 'lidar2image']:
            if frame[k].dim() == 3:
                frame[k] = frame[k].unsqueeze(0)
        if frame['lidar_aug_matrix'].dim() == 2:
            frame['lidar_aug_matrix'] = frame['lidar_aug_matrix'].unsqueeze(0)

        cam_bev = self.extract_camera(img, points, frame)
        lidar_bev = self.extract_lidar(points)
        x = self.fuser([cam_bev, lidar_bev])
        x = self.decoder['backbone'](x)
        x = self.decoder['neck'](x)
        head = self.heads['map'] if 'map' in self.heads else self.heads['object']
        return head(x)
