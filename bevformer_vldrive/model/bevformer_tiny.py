"""
BEVFormer-Tiny — pure PyTorch, MPS-compatible.
# compared to bevformer_base, bevformer_tiny has
# smaller backbone: R101-DCN -> R50
# smaller BEV: 200*200 -> 50*50
# less encoder layers: 6 -> 3
# smaller input size: 1600*900 -> 800*450
# multi-scale feautres -> single scale features (C5)

Pipeline per frame:
  1. ResNet-50 backbone + FPN neck extract C5 features from all 6 camera images.
  2. Camera and level positional embeddings are added.
  3. BEVFormerEncoder runs 3 × (TSA + SCA + FFN) layers to produce a
     50×50 BEV feature map.
  4. A lightweight DETR-style detection head (6-layer transformer decoder +
     classification / regression branches) predicts 3-D bounding boxes.

Device selection:  MPS → CUDA → CPU, set at construction time.
FP32 only: no torch.cuda.amp blocks anywhere in this file.

Loading the official BEVFormer-tiny_fp16 checkpoint
----------------------------------------------------
The official checkpoint uses mmdet key names:
_build_remap in tools/eval.py remaps these to match the above module structure
  img_backbone.*  →  backbone.*
  img_neck.*      →  neck.*
  pts_bbox_head.* →  encoder.* (TSA + SCA + FFN + norms, 3 layers)

checkpoint loading goes through load_checkpoint / _build_remap in tools/eval.py.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from .backbone    import ResNet50Backbone, FPNNeck
from .encoder     import BEVFormerEncoder
from .deform_attn import ms_deform_attn_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class LearnedPosEmbed(nn.Module):
    """Learned row + col positional embedding for the BEV grid."""
    def __init__(self, num_feats: int, row_embed: int, col_embed: int):
        super().__init__()
        self.row = nn.Embedding(row_embed, num_feats)
        self.col = nn.Embedding(col_embed, num_feats)

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(H, device=device)
        j = torch.arange(W, device=device)
        # (H, W, 2*num_feats)
        pos = torch.cat([self.col(j).unsqueeze(0).expand(H, -1, -1),
                         self.row(i).unsqueeze(1).expand(-1, W, -1)], dim=-1)
        return pos.permute(2, 0, 1).unsqueeze(0)    # (1, C, H, W)


# ---------------------------------------------------------------------------
# BEV deformable cross-attention for the detection decoder
# ---------------------------------------------------------------------------

def _inv_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


class BEVDeformCrossAttn(nn.Module):
    """
    2-D deformable cross-attention on the BEV feature map.

    Weight shapes match attentions.1.* in the official BEVFormer decoder
    checkpoint (num_heads=8, num_levels=1, num_points=4):
      sampling_offsets  : [64, 256]
      attention_weights : [32, 256]
      value_proj        : [256, 256]
      output_proj       : [256, 256]
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 num_levels: int = 1, num_points: int = 4):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim   = embed_dim // num_heads

        self.sampling_offsets  = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj        = nn.Linear(embed_dim, embed_dim)
        self.output_proj       = nn.Linear(embed_dim, embed_dim)
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias,   0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias,   0.)

    def forward(self, query: torch.Tensor, value: torch.Tensor,
                ref_pts_2d: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        """
        query       : (B, Q, C)
        value       : (B, L, C)   — BEV feature map (L = bev_h * bev_w)
        ref_pts_2d  : (B, Q, 2)   — reference points in [0, 1]
        spatial_shapes: (1, 2) long  [(bev_h, bev_w)]
        level_start_index: (1,) long  — kept for API uniformity with encoder
        """
        B, Q, _ = query.shape
        B, L, _ = value.shape

        v = self.value_proj(value).view(B, L, self.num_heads, self.head_dim)

        # Offsets scaled by inverse spatial dims so one unit = one pixel
        offsets = self.sampling_offsets(query).view(
            B, Q, self.num_heads, self.num_levels, self.num_points, 2)
        hw = spatial_shapes[0].float()                             # [H, W]
        offsets = offsets / hw.flip(0)[None, None, None, None, None, :]   # ÷ W, H

        ref = ref_pts_2d[:, :, None, None, None, :]                # (B,Q,1,1,1,2)
        locs = (ref + offsets).clamp(0.0, 1.0)                    # (B,Q,H,L,P,2)

        attn = self.attention_weights(query).view(
            B, Q, self.num_heads, self.num_levels, self.num_points).softmax(-1)

        out = ms_deform_attn_core(v, spatial_shapes, locs, attn)
        return self.output_proj(out)


class _BEVDecLayer(nn.Module):
    """One BEVFormer decoder layer: self-attn + BEV deformable cross-attn + FFN."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 ffn_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                                dropout=dropout, batch_first=True)
        self.cross_attn = BEVDeformCrossAttn(embed_dim, num_heads)
        self.linear1    = nn.Linear(embed_dim, ffn_dim)
        self.linear2    = nn.Linear(ffn_dim, embed_dim)
        self.norm1      = nn.LayerNorm(embed_dim)
        self.norm2      = nn.LayerNorm(embed_dim)
        self.norm3      = nn.LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, query_pos: torch.Tensor,
                bev_feat: torch.Tensor, ref_pts_2d: torch.Tensor,
                bev_spatial: torch.Tensor, bev_lvl_idx: torch.Tensor) -> torch.Tensor:
        # Post-norm + query_pos added to Q (and K) in self-attn and Q in cross-attn.
        # Matches official: query_pos is added inside each attention module before projection.
        q_p = query + query_pos
        sa_out = self.self_attn(q_p, q_p, query, need_weights=False)[0]   # Q=q+pos, K=q+pos, V=q
        q = self.norm1(query + self.dropout(sa_out))
        ca_out = self.cross_attn(q + query_pos, bev_feat, ref_pts_2d, bev_spatial, bev_lvl_idx)
        q = self.norm2(q + self.dropout(ca_out))
        ffn_out = self.linear2(torch.nn.functional.relu(self.linear1(q)))
        q = self.norm3(q + self.dropout(ffn_out))
        return q


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------

class SimpleDetHead(nn.Module):
    """
    BEVFormer-style detection head with 6-layer decoder.
      - Deformable cross-attention to BEV feature map (weight-compatible with
        the official checkpoint's attentions.1.* keys)
      - One reg_branch per decoder layer for iterative 3-D reference refinement
      - Single cls_branch on the final-layer query output
    """

    NUM_REG     = 10
    NUM_DEC     = 6

    def __init__(self, embed_dim: int = 256, num_queries: int = 900,
                 num_classes: int = 10, num_heads: int = 8,
                 ffn_dim: int = 512, dropout: float = 0.1,
                 bev_h: int = 50, bev_w: int = 50):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_queries = num_queries
        self.bev_h       = bev_h
        self.bev_w       = bev_w

        self.query_embed = nn.Embedding(num_queries, embed_dim * 2)  # content + pos
        self.ref_points  = nn.Linear(embed_dim, 3)                   # pos → initial 3-D ref

        self.decoder_layers = nn.ModuleList([
            _BEVDecLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(self.NUM_DEC)
        ])

        # reg_branches: official layout is
        #   [Linear(D,D), ReLU, Linear(D,D), ReLU, Linear(D,NUM_REG)]
        #   checkpoint indices: 0, (1=ReLU), 2, (3=ReLU), 4
        self.reg_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),    # 0
                nn.ReLU(inplace=True),              # 1
                nn.Linear(embed_dim, embed_dim),    # 2
                nn.ReLU(inplace=True),              # 3
                nn.Linear(embed_dim, self.NUM_REG), # 4
            )
            for _ in range(self.NUM_DEC)
        ])

        # cls_branch: official layout is
        #   [Linear(D,D), LN, ReLU, Linear(D,D), LN, ReLU, Linear(D,C)]
        #   checkpoint indices: 0, 1, (2=ReLU), 3, 4, (5=ReLU), 6
        self.cls_branch = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),    # 0
            nn.LayerNorm(embed_dim),            # 1
            nn.ReLU(inplace=True),              # 2
            nn.Linear(embed_dim, embed_dim),    # 3
            nn.LayerNorm(embed_dim),            # 4
            nn.ReLU(inplace=True),              # 5
            nn.Linear(embed_dim, num_classes),  # 6
        )

    def forward(self, bev_feat: torch.Tensor) -> dict:
        """
        bev_feat : (B, bev_h*bev_w, embed_dim)
        Returns  : dict with
                     'cls_logits'  (B, Q, num_classes)
                     'reg_preds'   (B, Q, 10)  — final layer deltas
                     'ref_pts'     (B, Q, 3)   — final refined reference points [0,1]
        """
        B      = bev_feat.shape[0]
        device = bev_feat.device

        bev_spatial   = torch.tensor([[self.bev_h, self.bev_w]],
                                     dtype=torch.long, device=device)
        bev_lvl_idx   = torch.zeros(1, dtype=torch.long, device=device)

        qw        = self.query_embed.weight                          # (Q, 2*C)
        # Official split: first C = query_pos (position), last C = content query
        query_pos = qw[:, :self.embed_dim].unsqueeze(0).expand(B, -1, -1)   # (B, Q, C)
        query     = qw[:, self.embed_dim:].unsqueeze(0).expand(B, -1, -1)   # (B, Q, C)

        # Initial 3-D reference points from query_pos (matching official reference_points(query_pos))
        ref_pts   = self.ref_points(query_pos).sigmoid()             # (B, Q, 3)
        reg_out   = None

        for layer, reg_branch in zip(self.decoder_layers, self.reg_branches):
            ref_2d  = ref_pts[..., :2]                               # (B, Q, 2)
            query   = layer(query, query_pos, bev_feat, ref_2d, bev_spatial, bev_lvl_idx)
            reg_out = reg_branch(query)                                # (B, Q, 10)

            # Refine reference points — xy from reg[0:2], z from reg[4].
            # Clamp deltas to ±4 logit units so a single layer can't drive ref_pts
            # all the way to 0/1, which would cause _inv_sigmoid → ±Inf next iter.
            # Clamp the final logit to ±6  (sigmoid(±6) ≈ 0.0025 / 0.9975).
            ref_inv = _inv_sigmoid(ref_pts.clone())
            ref_inv[..., :2]  = ref_inv[..., :2]  + reg_out[..., :2].clamp(-4.0, 4.0)
            ref_inv[..., 2:3] = ref_inv[..., 2:3] + reg_out[..., 4:5].clamp(-4.0, 4.0)
            ref_pts = ref_inv.clamp(-6.0, 6.0).sigmoid().detach()

        return {
            'cls_logits': self.cls_branch(query),                      # (B, Q, C)
            'reg_preds':  reg_out,                                     # (B, Q, 10) deltas
            'ref_pts':    ref_pts,                                     # (B, Q, 3) refined [0,1]
        }


# ---------------------------------------------------------------------------
# Full BEVFormer-Tiny model
# ---------------------------------------------------------------------------

# Official img_norm_cfg (bevformer_tiny_fp16.py), to_rgb=True:
NUSCENES_IMG_MEAN = torch.tensor([123.675, 116.280, 103.530])   # RGB order (R, G, B)
NUSCENES_IMG_STD  = torch.tensor([ 58.395,  57.120,  57.375])   # RGB order, ImageNet std

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


class BEVFormerTiny(nn.Module):

    BEV_H       = 50
    BEV_W       = 50
    EMBED_DIM   = 256
    NUM_CAMS    = 6
    NUM_Z       = 4
    NUM_CLASSES = 10
    NUM_QUERIES = 900

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        D = self.EMBED_DIM

        self.device = _get_device()

        # --- Backbone (ResNet-50, C5 → 2048 ch) + FPN neck (2048 → 256) ------
        # Separated so the official checkpoint can be loaded with a simple
        # prefix remap:  img_backbone.* → backbone.*,  img_neck.* → neck.*
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        self.neck     = FPNNeck(in_ch=2048, out_ch=D)

        # Camera and level positional embeddings (mirroring transformer.py)
        self.cams_embeds  = nn.Parameter(torch.empty(self.NUM_CAMS, D))
        self.level_embeds = nn.Parameter(torch.empty(1, D))          # single level
        nn.init.normal_(self.cams_embeds)
        nn.init.normal_(self.level_embeds)

        # --- BEV queries + positional encoding --------------------------------
        self.bev_queries = nn.Parameter(torch.empty(self.BEV_H * self.BEV_W, D))
        nn.init.normal_(self.bev_queries)
        self.bev_pos_enc = LearnedPosEmbed(D // 2, self.BEV_H, self.BEV_W)

        # CAN-bus MLP (18-dim signal -> embed) with LayerNorm
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, D // 2),
            nn.ReLU(inplace=True),
            nn.Linear(D // 2, D),
            nn.ReLU(inplace=True),
            nn.LayerNorm(D),
        )

        # --- BEV encoder ------------------------------------------------------
        self.encoder = BEVFormerEncoder(
            embed_dim=D,
            bev_h=self.BEV_H,
            bev_w=self.BEV_W,
            num_cams=self.NUM_CAMS,
            num_layers=3,
            num_points_in_pillar=self.NUM_Z,
            pc_range=PC_RANGE,
            ffn_dim=D * 2,
            num_heads=8,
            num_points_sca=8,
            num_points_tsa=4,
        )

        # --- Detection head ---------------------------------------------------
        self.det_head = SimpleDetHead(
            embed_dim=D,
            num_queries=self.NUM_QUERIES,
            num_classes=self.NUM_CLASSES,
            bev_h=self.BEV_H,
            bev_w=self.BEV_W,
        )

        # Move to device and pin to float32 in one call.
        # MPS does not support autocast or bfloat16 reliably; explicit fp32 avoids
        # silent NaN propagation from mixed-precision state.
        self.to(device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Image pre-processing (done on-device for MPS efficiency)
    # ------------------------------------------------------------------

    def _normalize(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, num_cams, 3, H, W)  uint8 or float [0,255]"""
        mean = NUSCENES_IMG_MEAN.to(imgs.device)[None, None, :, None, None]
        std  = NUSCENES_IMG_STD .to(imgs.device)[None, None, :, None, None]
        return (imgs.float() - mean) / std

    # ------------------------------------------------------------------
    # Ego-motion BEV shift (from CAN bus data, matching transformer.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_shift(img_metas: list, bev_h: int, bev_w: int,
                       grid_length=(0.512, 0.512)) -> torch.Tensor:  # grid_length=(0.512, 0.512) means each BEV cell covers 0.512 m, so a 50-cell grid spans 25.6 m. Dividing by 0.512 × 50 converts metres → normalised [0,1] BEV coordinates.
        """Returns (bs, 2) shift tensor in normalised BEV coordinates."""
        # The CAN bus provides integrated position deltas and absolute yaw directly, so we can compute the BEV shift in one step without needing to integrate over time or apply a rotation matrix.
        # position delta in global frame (metres)
        dx = np.array([m['can_bus'][0] for m in img_metas])
        dy = np.array([m['can_bus'][1] for m in img_metas])
        # # absolute ego yaw (degrees)
        angle = np.array([m['can_bus'][-2] / np.pi * 180 for m in img_metas])

        length = np.sqrt(dx ** 2 + dy ** 2)  # distance travelled
        tangle = np.arctan2(dy, dx) / np.pi * 180  # direction of travel in global frame
        bev_angle = angle - tangle  # heading relative to direction of travel
        # Project distance into BEV (x=right, y=forward) and normalise by grid cell size
        sy = length * np.cos(bev_angle / 180 * np.pi) / grid_length[0] / bev_h  # [0,1] BEV units, forward axis
        sx = length * np.sin(bev_angle / 180 * np.pi) / grid_length[1] / bev_w  # [0,1] BEV units, lateral axis
        return torch.tensor(np.stack([sx, sy], axis=1), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Rotate prev BEV by ego-rotation angle (from CAN bus)
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_prev_bev(prev_bev: torch.Tensor, img_metas: list,
                         bev_h: int, bev_w: int) -> torch.Tensor:
        """prev_bev: (bs, L, C) -> rotated (bs, L, C)."""
        bs = prev_bev.shape[0]
        out = prev_bev.clone()
        for i in range(bs):
            angle = img_metas[i]['can_bus'][-1]  # delta_yaw in degrees
            tmp = prev_bev[i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
            # TF.rotate does a bilinear rotation of the 50×50 BEV grid in place. This compensates for ego yaw change — features that were "ahead" of the car last frame are rotated to stay in the correct global direction.
            tmp = TF.rotate(tmp, float(angle), center=[bev_h // 2, bev_w // 2])  # spatial 2D rotation on BEV grid
            out[i] = tmp.permute(1, 2, 0).reshape(bev_h * bev_w, -1)  # back to (L, C)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        imgs:      torch.Tensor,   # (B, num_cams, 3, H, W)  float [0,255]
        img_metas: list,           # list[dict] with keys lidar2img, img_shape, can_bus
        prev_bev:  torch.Tensor | None = None,  # (B, L, C) or None
    ) -> dict:

        B, N, _, H, W = imgs.shape
        device = self.device

        # Force fp32 on all inputs — MPS does not support autocast or fp16 reliably.
        imgs = imgs.float().to(device)
        if prev_bev is not None:
            prev_bev = prev_bev.float().to(device)

        imgs = self._normalize(imgs)

        # ---------- Backbone + FPN neck ----------
        # Flatten cameras into batch: (B*N, 3, H, W)
        imgs_flat = imgs.view(B * N, 3, H, W)
        feat = self.neck(self.backbone(imgs_flat))          # (B*N, 256, Hf, Wf)
        _, C, Hf, Wf = feat.shape

        # Add camera and level embeddings then reshape
        feat = feat.view(B, N, C, Hf, Wf)
        feat = feat + self.cams_embeds[None, :, :, None, None]          # (1,N,C,1,1)
        feat = feat + self.level_embeds.view(1, 1, -1, 1, 1)            # (1,1,C,1,1)

        # Flatten spatial dims: (num_cams, H*W, B, C)
        feat_flat = feat.permute(1, 0, 2, 3, 4).flatten(3)  # (N, B, C, Hf*Wf)
        feat_flat = feat_flat.permute(0, 3, 1, 2)            # (N, Hf*Wf, B, C)

        spatial_shapes    = torch.tensor([[Hf, Wf]], dtype=torch.long,  device=device)
        level_start_index = torch.tensor([0],        dtype=torch.long,  device=device)

        # ---------- BEV queries ----------
        bev_q   = self.bev_queries.unsqueeze(1).expand(-1, B, -1)  # (L, B, C)
        bev_pos = self.bev_pos_enc(self.BEV_H, self.BEV_W, device) # (1, C, H, W)
        # (1, C, H*W) → (1, L, C) — batch-first format for broadcasting with (B, L, C)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1)              # (1, L, C)

        # CAN-bus conditioning
        can_bus = torch.from_numpy(
            np.stack([m['can_bus'] for m in img_metas], axis=0).astype(np.float32)
        ).to(device)                                                 # (B, 18)
        bev_q = bev_q + self.can_bus_mlp(can_bus).unsqueeze(0)          # (L, B, C)

        # Rotate and shift prev BEV for temporal alignment
        if prev_bev is not None:
            prev_bev = self._rotate_prev_bev(prev_bev, img_metas, self.BEV_H, self.BEV_W)
            shift = self._compute_shift(img_metas, self.BEV_H, self.BEV_W).to(device)
        else:
            shift = None

        # ---------- BEV Encoder ----------
        bev_feat = self.encoder(
            bev_q,
            feat_flat,
            spatial_shapes,
            level_start_index,
            img_metas,
            prev_bev=prev_bev,
            shift=shift,
            bev_pos=bev_pos,    # (1, L, C) — added to Q inside each TSA and SCA
        )                                                            # (B, L, C)

        # Guard the encoder→decoder boundary.
        # Projection edge-cases (cameras pointing away from BEV anchor grid) or
        # first-frame cold-start can leave isolated NaN/Inf in bev_feat.
        # Zero-filling keeps the detection head stable; the caller's NaN guard
        # on cls_logits/reg_preds/ref_pts will still catch frames where the
        # resulting predictions are garbage.
        if not bev_feat.isfinite().all():
            n_bad = (~bev_feat.isfinite()).sum().item()
            print(f'  [WARN-ENC] bev_feat has {n_bad} non-finite values — zeroing')
            bev_feat = bev_feat.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- Detection head ----------
        preds = self.det_head(bev_feat)

        return {
            'bev_feat':   bev_feat,                                  # (B, L, C)
            'cls_logits': preds['cls_logits'],                       # (B, Q, 10)
            'reg_preds':  preds['reg_preds'],                        # (B, Q, 10)
            'ref_pts':    preds['ref_pts'],                          # (B, Q, 3)
        }
