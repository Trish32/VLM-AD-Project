"""
BEVF_FasterRCNN (PointPillars) — full pure-PyTorch assembly for bevf_pp.

LiDAR stream + frozen camera lift-splat stream, fused via concat -> reduc_conv
(BN+ReLU) -> SE block, then Anchor3DHead decode. Device defaults to MPS, FP32.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .voxelize import hard_voxelize
from .vfe import HardVFE
from .scatter import PointPillarsScatter
from .second import SECOND, SECONDFPN
from .cbnet import CBSwinTransformer
from .fpnc import FPNC
from .lss import LiftSplatShoot
from .anchor3d_head import Anchor3DHead, AlignedAnchor3DRangeGenerator


class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.att(x)


class ReducConv(nn.Module):
    """ConvModule(conv bias=False -> BN -> ReLU) matching detector reduc_conv."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.01)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class BEVF_FasterRCNN(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        self.cfg = cfg
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.num_views = cfg.NUM_VIEWS
        lic, imc = cfg.LIC, cfg.IMC

        # ---- LiDAR stream ----
        self.pts_voxel_encoder = HardVFE(
            in_channels=4, feat_channels=(64, 64),
            voxel_size=cfg.VOXEL_SIZE, point_cloud_range=cfg.POINT_CLOUD_RANGE)
        self.pts_middle_encoder = PointPillarsScatter(64, (cfg.GRID_SIZE[1], cfg.GRID_SIZE[0]))
        self.pts_backbone = SECOND()
        self.pts_neck = SECONDFPN()

        # ---- Camera stream (frozen) ----
        self.img_backbone = CBSwinTransformer(
            embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            window_size=7, mlp_ratio=4., qkv_bias=True, drop_path_rate=0.2,
            ape=False, patch_norm=True, out_indices=(0, 1, 2, 3))
        self.img_neck = FPNC(in_channels=(96, 192, 384, 768), out_channels=256,
                             num_outs=5, outC=imc, final_dim=cfg.FINAL_DIM,
                             downsample=cfg.DOWNSAMPLE, use_adp=True)
        self.lift_splat_shot_vis = LiftSplatShoot(
            final_dim=cfg.FINAL_DIM, camera_depth_range=cfg.CAMERA_DEPTH_RANGE,
            pc_range=cfg.POINT_CLOUD_RANGE, downsample=cfg.DOWNSAMPLE,
            grid=cfg.LSS_GRID, inputC=imc, camC=64)

        # ---- Fusion ----
        self.reduc_conv = ReducConv(lic + imc, lic)
        self.seblock = SE_Block(lic)

        # ---- Head ----
        gen = AlignedAnchor3DRangeGenerator(
            cfg.ANCHOR_RANGES, cfg.ANCHOR_SIZES, cfg.ANCHOR_ROTATIONS,
            cfg.ANCHOR_CUSTOM_VALUES)
        self.pts_bbox_head = Anchor3DHead(
            cfg.NUM_CLASSES, lic, lic, gen, code_size=cfg.CODE_SIZE,
            dir_offset=cfg.DIR_OFFSET, dir_limit_offset=cfg.DIR_LIMIT_OFFSET,
            test_cfg=cfg.TEST_CFG)

    # ------------------------------------------------------------------
    def voxelize(self, points):
        """points: list[(N,4)] -> voxels, num_points, coors(batch-padded)."""
        voxels, coors, num_points = [], [], []
        max_voxels = self.cfg.MAX_VOXELS_TRAIN if self.training else self.cfg.MAX_VOXELS_TEST
        for res in points:
            v, n, c = hard_voxelize(res, self.cfg.VOXEL_SIZE, self.cfg.POINT_CLOUD_RANGE,
                                    self.cfg.MAX_NUM_POINTS, max_voxels)
            voxels.append(v)
            num_points.append(n)
            coors.append(c)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, c in enumerate(coors):
            coors_batch.append(F.pad(c, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_pts_feat(self, points):
        voxels, num_points, coors = self.voxelize(points)
        vf = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = int(coors[-1, 0].item()) + 1
        x = self.pts_middle_encoder(vf, coors, batch_size)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img):
        """img: (B*N, 3, H, W) -> list of one feat (B*N, 256, 112, 200)."""
        feats = self.img_backbone(img)
        return self.img_neck(list(feats))

    def extract_feat(self, points, img, lidar2img):
        # Camera stream is frozen (freeze_img=True): run it under no_grad so
        # fine-tuning trains only the LiDAR stream + fusion + head.
        with torch.no_grad():
            img_feats = self.extract_img_feat(img)
            BN, C, H, W = img_feats[0].shape
            batch_size = BN // self.num_views
            img_feats_view = img_feats[0].view(batch_size, self.num_views, C, H, W)

            rots, trans = [], []
            for b in range(batch_size):
                rot_list, trans_list = [], []
                for mat in lidar2img[b]:
                    mat = torch.as_tensor(mat, dtype=torch.float32, device=img_feats_view.device)
                    inv = torch.inverse(mat)
                    rot_list.append(inv[:3, :3])
                    trans_list.append(inv[:3, 3].view(-1))
                rots.append(torch.stack(rot_list, 0))
                trans.append(torch.stack(trans_list, 0))
            rots = torch.stack(rots)
            trans = torch.stack(trans)
            img_bev_feat, depth = self.lift_splat_shot_vis(img_feats_view, rots, trans)

        pts_feats = self.extract_pts_feat(points)
        if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
            img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:],
                                         mode='bilinear', align_corners=True)
        fused = self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))
        fused = self.seblock(fused)
        return [fused], depth

    @torch.no_grad()
    def simple_test(self, points, img, lidar2img):
        """Single-sample inference. Returns (bboxes, scores, labels)."""
        feats, _ = self.extract_feat(points, img, lidar2img)
        cls, reg, dir_ = self.pts_bbox_head(feats[0])
        return self.pts_bbox_head.get_bboxes_single(
            cls[0], reg[0], dir_[0], feats[0].device)

    def forward_head(self, feats):
        return self.pts_bbox_head(feats[0])
