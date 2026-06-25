"""
LiftSplatShoot — pure-PyTorch port of cam_stream_lss.py (lss=False path) for
bevf_pp. Lifts per-camera features to a frustum via a predicted depth
distribution, splats into a BEV voxel grid (cumsum-pooling), collapses Z, and
encodes to a 256-ch camera BEV feature.

Geometry uses lidar2img inverse (rots/trans) supplied by the detector; the
frustum is in original (900x1600) pixel coordinates.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.tensor([round((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]],
                      dtype=torch.long)
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


class CamEncode(nn.Module):
    def __init__(self, D, C, inputC):
        super().__init__()
        self.D = D
        self.C = C
        self.depthnet = nn.Conv2d(inputC, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        return depth, new_x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)
        return x, depth


class LiftSplatShoot(nn.Module):
    def __init__(self, final_dim=(900, 1600), camera_depth_range=(4.0, 45.0, 1.0),
                 pc_range=(-50, -50, -5, 50, 50, 3), downsample=8, grid=0.5,
                 inputC=256, camC=64):
        super().__init__()
        self.pc_range = pc_range
        self.grid_conf = {
            'xbound': [pc_range[0], pc_range[3], grid],
            'ybound': [pc_range[1], pc_range[4], grid],
            'zbound': [pc_range[2], pc_range[5], grid],
            'dbound': list(camera_depth_range),
        }
        self.final_dim = final_dim
        self.grid = grid

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'], self.grid_conf['ybound'],
                               self.grid_conf['zbound'])
        self.register_buffer('dx', dx)
        self.register_buffer('bx', bx)
        self.register_buffer('nx', nx)

        self.downsample = downsample
        self.fH = final_dim[0] // downsample
        self.fW = final_dim[1] // downsample
        self.camC = camC
        self.inputC = inputC
        frustum = self.create_frustum()
        self.register_buffer('frustum', frustum)
        self.D = frustum.shape[0]
        self.camencode = CamEncode(self.D, self.camC, self.inputC)

        z = self.grid_conf['zbound']
        cz = int(self.camC * round((z[1] - z[0]) / z[2]))
        self.bevencode = nn.Sequential(
            nn.Conv2d(cz, cz, 3, padding=1, bias=False),
            nn.BatchNorm2d(cz), nn.ReLU(inplace=True),
            nn.Conv2d(cz, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, inputC, 3, padding=1, bias=False),
            nn.BatchNorm2d(inputC), nn.ReLU(inplace=True),
        )

    def create_frustum(self):
        ogfH, ogfW = self.final_dim
        fH, fW = self.fH, self.fW
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D = ds.shape[0]
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        return torch.stack((xs, ys, ds), -1)

    def get_geometry(self, rots, trans):
        B, N, _ = trans.shape
        points = self.frustum.repeat(B, N, 1, 1, 1, 1).unsqueeze(-1)  # B N D H W 3 1
        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        return points

    def get_cam_feats(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x, depth = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, H, W)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, H, W)
        return x, depth

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        x = x.reshape(Nprime, C)

        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full((Nprime // B, 1), ix, device=x.device, dtype=torch.long)
                              for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
            + geom_feats[:, 1] * (self.nx[2] * B) \
            + geom_feats[:, 2] * B + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)

        final = torch.zeros((B, C, int(self.nx[2]), int(self.nx[0]), int(self.nx[1])), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        return final

    def s2c(self, x):
        B, C, H, W, L = x.shape
        bev = torch.reshape(x, (B, C * H, W, L))
        bev = bev.permute((0, 1, 3, 2))
        return bev

    def forward(self, x, rots, trans):
        geom = self.get_geometry(rots, trans)
        x, depth = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, x)  # B C Z X Y
        bev = self.s2c(x)
        x = self.bevencode(bev)
        return x, depth
