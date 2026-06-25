"""
TransFusionHead — pure-PyTorch port for MIT BEVFusion detection.
heatmap query-init (top-200 local-max) -> 1 transformer decoder layer
(self-attn + cross-attn to BEV, learned pos embeds) -> per-query prediction
heads -> decode to 3D boxes. hidden 128, 8 heads, 10 classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN2d(nn.Module):
    def __init__(self, ic, oc, k, p):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(oc)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class ConvBN1d(nn.Module):
    def __init__(self, ic, oc, k=1):
        super().__init__()
        self.conv = nn.Conv1d(ic, oc, k, bias=False)
        self.bn = nn.BatchNorm1d(oc)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class PositionEmbeddingLearned(nn.Module):
    """Encode a 2D position (x,y) into a hidden-dim embedding via a tiny MLP. Used to give
    both queries and BEV keys a learned positional signal in the decoder attention."""

    def __init__(self, in_ch, num_pos):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(in_ch, num_pos, 1), nn.BatchNorm1d(num_pos), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos, num_pos, 1))

    def forward(self, xyz):
        # xyz: (B, P, 2) -> (B, hidden, P)
        return self.position_embedding_head(xyz.transpose(1, 2).contiguous())


class TransformerDecoderLayer(nn.Module):
    """One DETR-style decoder layer. Queries (object proposals) first attend to EACH OTHER
    (self-attn, deduplicate/relate proposals), then attend to the BEV feature map
    (cross-attn, pull in scene evidence), then an FFN. Position embeddings are ADDED to
    q/k (not v) so attention is location-aware while values stay content-only."""

    def __init__(self, d_model, nhead, ffn, self_pos, cross_pos):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)         # query<->query
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)    # query<->BEV
        self.linear1 = nn.Linear(d_model, ffn)
        self.linear2 = nn.Linear(ffn, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_posembed = self_pos      # embeds query (x,y)
        self.cross_posembed = cross_pos    # embeds BEV-cell (x,y)

    @staticmethod
    def _pos(t, p):
        return t if p is None else t + p   # add positional embedding to a tensor

    def forward(self, query, key, query_pos, key_pos):
        # query: (B,C,Pq) object queries; key: (B,C,Pk) flattened BEV; *_pos: (B,P,2) coords
        qpe = self.self_posembed(query_pos).permute(2, 0, 1)   # (Pq,B,C) query pos-embed
        kpe = self.cross_posembed(key_pos).permute(2, 0, 1)    # (Pk,B,C) BEV pos-embed
        query = query.permute(2, 0, 1)     # (Pq,B,C)  nn.MultiheadAttention wants (seq,B,C)
        key = key.permute(2, 0, 1)         # (Pk,B,C)
        # self-attention among queries (q,k,v all = query + its pos embed)
        q = k = v = self._pos(query, qpe)
        query = self.norm1(query + self.self_attn(q, k, v)[0])
        # cross-attention: queries(+qpe) attend to BEV keys(+kpe); values = BEV(+kpe)
        q2 = self.multihead_attn(self._pos(query, qpe), self._pos(key, kpe),
                                 self._pos(key, kpe))[0]
        query = self.norm2(query + q2)
        # feed-forward
        query = self.norm3(query + self.linear2(F.relu(self.linear1(query))))
        return query.permute(1, 2, 0)      # back to (B,C,Pq)


class FFNHeads(nn.Module):
    """Per-query prediction heads: each is a small 1D conv stack mapping the query feature
    to one box attribute. heads = {center:2, height:1, dim:3, rot:2, vel:2, heatmap:10},
    all predicted independently per query (P queries in parallel via Conv1d over the P axis)."""

    def __init__(self, in_ch, heads, head_conv=64):
        super().__init__()
        self.heads = heads
        for name, (out_ch, num_conv) in heads.items():
            layers = []
            c = in_ch
            for _ in range(num_conv - 1):
                layers.append(ConvBN1d(c, head_conv, 1))
                c = head_conv
            layers.append(nn.Conv1d(c, out_ch, 1, bias=True))   # final: -> out_ch attributes
            setattr(self, name, nn.Sequential(*layers))

    def forward(self, x):
        # x: (B, hidden, P) -> dict of (B, out_ch, P)
        return {name: getattr(self, name)(x) for name in self.heads}


class TransFusionHead(nn.Module):
    """Query-based 3D detection head (DETR-style, but queries are INITIALIZED from a class
    heatmap instead of being learned constants). Flow: shared conv -> dense heatmap ->
    top-200 peaks become object queries -> 1 transformer decoder layer refines them against
    the BEV -> per-query heads predict box attributes -> decode to metric 3D boxes.
    hidden 128, 8 heads, 10 nuScenes classes, num_proposals 200."""

    def __init__(self, cfg, in_channels=512, hidden=128, num_proposals=200,
                 num_classes=10, num_heads=8, ffn=256, nms_kernel=3):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_proposals = num_proposals        # = number of object queries (200)
        self.nms_kernel_size = nms_kernel
        self.out_size_factor = 8                  # BEV cell -> metres scale (decoder stride)
        self.voxel_size = cfg.VOXEL_SIZE
        self.pc_range = cfg.POINT_CLOUD_RANGE
        self.post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]   # valid box-center bounds

        self.shared_conv = nn.Conv2d(in_channels, hidden, 3, padding=1, bias=True)  # 512->128
        self.heatmap_head = nn.Sequential(        # BEV -> per-class peak map
            ConvBN2d(hidden, hidden, 3, 1),
            nn.Conv2d(hidden, num_classes, 3, padding=1, bias=True))
        self.class_encoding = nn.Conv1d(num_classes, hidden, 1)   # class one-hot -> query feature
        self.decoder = nn.ModuleList([TransformerDecoderLayer(
            hidden, num_heads, ffn,
            PositionEmbeddingLearned(2, hidden), PositionEmbeddingLearned(2, hidden))])  # 1 layer
        # each prediction head = (output channels, num convs); all per-query box attributes
        heads = dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2),
                     vel=(2, 2), heatmap=(num_classes, 2))
        self.prediction_heads = nn.ModuleList([FFNHeads(hidden, heads)])

        # precompute the (x,y) coordinate of every decoder-BEV cell (180x180), as query/key positions
        xs = cfg.GRID_SIZE[0] // self.out_size_factor    # 1440/8 = 180
        ys = cfg.GRID_SIZE[1] // self.out_size_factor
        self.register_buffer('bev_pos', self._grid(xs, ys))  # (1, xs*ys, 2)

    @staticmethod
    def _grid(xs, ys):
        # build cell-center coords (i+0.5, j+0.5) for an xs-by-ys grid -> (1, xs*ys, 2)
        bx, by = torch.meshgrid(torch.linspace(0, xs - 1, xs),
                                torch.linspace(0, ys - 1, ys), indexing='ij')
        bx = bx + 0.5
        by = by + 0.5
        coord = torch.cat([bx[None], by[None]], 0)[None]
        return coord.view(1, 2, -1).permute(0, 2, 1)

    def _predict(self, inputs):
        """Shared forward: returns raw prediction dict + query metadata.
        Pipeline: BEV -> dense heatmap -> pick top-200 peaks as object queries ->
        decoder refines them against the BEV -> per-query box attributes."""
        B = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)                       # (B,128,H,W) project 512->128
        lidar_flat = lidar_feat.view(B, lidar_feat.shape[1], -1)    # (B,128,HW) BEV as a token set
        bev_pos = self.bev_pos.repeat(B, 1, 1).to(inputs.device)    # (B,HW,2) cell coords

        # --- dense class heatmap over every BEV cell ---
        dense_heatmap = self.heatmap_head(lidar_feat)               # (B,10,H,W) class logits per cell
        heatmap = dense_heatmap.detach().sigmoid()
        # local-max NMS: keep only cells that are a 3x3 peak (suppress neighbors of a stronger cell)
        pad = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        lm = F.max_pool2d(heatmap, self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, pad:-pad, pad:-pad] = lm
        local_max[:, 8] = heatmap[:, 8]   # pedestrian: small/dense -> NO suppression (kernel 1)
        local_max[:, 9] = heatmap[:, 9]   # traffic cone: same
        heatmap = heatmap * (heatmap == local_max)                 # zero out non-peak cells
        heatmap = heatmap.view(B, heatmap.shape[1], -1)            # (B,10,HW)

        # --- pick the top-200 (class, cell) peaks across the whole heatmap as QUERIES ---
        top = heatmap.view(B, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
        top_class = top // heatmap.shape[-1]    # which class channel  (flattened index decode)
        top_index = top % heatmap.shape[-1]     # which BEV cell
        # query feature = the BEV feature at each chosen cell
        query_feat = lidar_flat.gather(
            index=top_index[:, None, :].expand(-1, lidar_flat.shape[1], -1), dim=-1)   # (B,128,200)
        self.query_labels = top_class
        # add a learned class embedding so the query knows which class peak spawned it
        one_hot = F.one_hot(top_class, self.num_classes).permute(0, 2, 1).float()       # (B,10,200)
        query_feat = query_feat + self.class_encoding(one_hot)
        # query position = the (x,y) BEV cell of each peak
        query_pos = bev_pos.gather(
            index=top_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)  # (B,200,2)

        # --- refine queries through the decoder, then predict box attributes ---
        for i, layer in enumerate(self.decoder):
            query_feat = layer(query_feat, lidar_flat, query_pos, bev_pos)   # self+cross attn
            res = self.prediction_heads[i](query_feat)                       # dict of box attrs
            # center head predicts an OFFSET from the query cell -> add the cell position back
            res['center'] = res['center'] + query_pos.permute(0, 2, 1)
        # heatmap confidence at each query's cell (used to scale final scores)
        query_heatmap_score = heatmap.gather(
            index=top_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)
        return res, dense_heatmap, query_heatmap_score, one_hot, query_pos

    @torch.no_grad()
    def forward(self, inputs):
        # inference: predict then decode to (boxes, scores, labels)
        res, _, qhs, one_hot, _ = self._predict(inputs)
        return self._decode(res, qhs, one_hot)

    def forward_train(self, inputs):
        """Training path: return RAW predictions (no decode/no NMS) for the loss, which
        does Hungarian matching in encoded space. Returns (res, dense_heatmap, query_labels)."""
        res, dense_heatmap, _, _, _ = self._predict(inputs)
        return res, dense_heatmap, self.query_labels

    def _decode(self, res, query_heatmap_score, one_hot):
        """Turn per-query raw attributes into metric 3D boxes (x,y,z,w,l,h,yaw,vx,vy) + scores."""
        # final score = per-query class prob * the heatmap confidence at its cell, masked to its class
        score = res['heatmap'].sigmoid() * query_heatmap_score * one_hot   # (B,10,P)
        center = res['center'].clone()
        # center is in BEV cells -> metres: cell * out_size_factor(8) * voxel_size + range_min
        center[:, 0] = center[:, 0] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        center[:, 1] = center[:, 1] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        dim = res['dim'].exp()                          # log-size -> metric size (always positive)
        height = res['height'] - dim[:, 2:3] * 0.5      # box center z = predicted z minus half-height
        rot = torch.atan2(res['rot'][:, 0:1], res['rot'][:, 1:2])   # (sin,cos) -> yaw angle — a standard trick to avoid the angle wraparound discontinuity
        vel = res['vel']
        boxes = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)  # (B,P,9)
        scores = score.max(1).values                   # best class score per query
        labels = score.max(1).indices                  # its class

        # keep only boxes whose center falls inside the valid post-center range
        B = boxes.shape[0]
        out = []
        pcr = torch.tensor(self.post_center_range, device=boxes.device)
        for i in range(B):
            b = boxes[i]
            mask = ((b[:, :3] >= pcr[:3]).all(1) & (b[:, :3] <= pcr[3:]).all(1))
            out.append((b[mask], scores[i][mask], labels[i][mask]))
        return out                                      # list over batch: (boxes, scores, labels)
