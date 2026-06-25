"""
Sparse4D-v3 detection modules — pure PyTorch, MPS-compatible.

Differences from v2 (detection3d.py):

SparseBox3DEncoderV3
  - per-component embedding dims [pos=128, size=32, yaw=32, vel=64]
  - mode="cat": output = cat([pos, size, yaw, vel]) = 256 (no output_fc)
  - each component MLP has out_loops=4: 4 × [Linear, ReLU, LayerNorm]

SparseBox3DRefinementModuleV3
  - adds quality_layers → (centerness, yawness) per anchor
  - quality input = instance_feature + anchor_embed (same as regression)
  - forward returns (refined_anchor, cls_logits, quality)

SparseBox3DDecoderV3
  - flatten top-k over (anchor × class) like the reference
  - final score = sigmoid(cls) * sigmoid(centerness), re-sorted

Anchor layout (same as v2): [x,y,z, log_l,log_w,log_h, sin,cos, vx,vy,vz]
Quality layout: [centerness, yawness] (CNS=0, YNS=1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .detection3d import NUM_CLASSES, ANCHOR_DIMS, Scale


def _linear_relu_ln(out_dims: int, in_loops: int, out_loops: int,
                    in_dims: int | None = None) -> list:
    """Reference linear_relu_ln: out_loops × [in_loops × (Linear, ReLU), LN]."""
    if in_dims is None:
        in_dims = out_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(in_dims, out_dims))
            layers.append(nn.ReLU(inplace=True))
            in_dims = out_dims
        layers.append(nn.LayerNorm(out_dims))
    return layers


# ---------------------------------------------------------------------------
# Anchor box encoder (v3: cat mode, per-component dims)
# ---------------------------------------------------------------------------

class SparseBox3DEncoderV3(nn.Module):
    """
    v3 anchor encoder: each box component gets its own embedding subspace and
    the results are CONCATENATED (not summed):
      pos (3) → 128,  size (3) → 32,  yaw (2) → 32,  vel (3) → 64  ⇒ 256

    Checkpoint keys: pos_fc.* / size_fc.* / yaw_fc.* / vel_fc.*
    (Sequential indices 0,3,6,9 = Linear; 2,5,8,11 = LayerNorm; no output_fc)
    """

    def __init__(self, embed_dims: tuple[int, ...] = (128, 32, 32, 64)):
        super().__init__()
        d_pos, d_size, d_yaw, d_vel = embed_dims
        self.pos_fc  = nn.Sequential(*_linear_relu_ln(d_pos,  1, 4, 3))
        self.size_fc = nn.Sequential(*_linear_relu_ln(d_size, 1, 4, 3))
        self.yaw_fc  = nn.Sequential(*_linear_relu_ln(d_yaw,  1, 4, 2))
        self.vel_fc  = nn.Sequential(*_linear_relu_ln(d_vel,  1, 4, 3))

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        """anchor (..., 11) → embedding (..., 256)"""
        pos  = self.pos_fc(anchor[..., 0:3])
        size = self.size_fc(anchor[..., 3:6])
        yaw  = self.yaw_fc(anchor[..., 6:8])
        vel  = self.vel_fc(anchor[..., 8:11])
        # concatenates unequal-width pieces (pos128 + size32 + yaw32 + vel64 = 256) with no output_fc
        return torch.cat([pos, size, yaw, vel], dim=-1)


# ---------------------------------------------------------------------------
# Box refinement with quality estimation
# ---------------------------------------------------------------------------

class SparseBox3DRefinementModuleV3(nn.Module):
    """
    Same regression / classification structure as v2 plus a quality branch:
      quality_layers : linear_relu_ln(D,1,2) + Linear(D, 2) → [centerness, yawness]
    Regression and quality consume (instance_feature + anchor_embed);
    classification consumes instance_feature only (reference behaviour).
    """

    def __init__(self, embed_dims: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        D = embed_dims

        self.layers = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.LayerNorm(D),
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.LayerNorm(D),
            nn.Linear(D, ANCHOR_DIMS),
            Scale([1.0] * ANCHOR_DIMS),
        )
        self.cls_layers = nn.Sequential(
            *_linear_relu_ln(D, 1, 2),
            nn.Linear(D, num_classes),
        )
        self.quality_layers = nn.Sequential(
            *_linear_relu_ln(D, 1, 2),
            nn.Linear(D, 2),                # [centerness, yawness]
        )

    def forward(
        self,
        instance_feature: torch.Tensor,           # (B, N, D)
        anchor:           torch.Tensor,           # (B, N, 11)
        anchor_embed:     torch.Tensor,           # (B, N, D)
        time_interval:    float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Regression and quality both consume instance_feature + anchor_embed; 
        classification(cls_logits) still consumes instance_feature alone
        """
        feature = instance_feature + anchor_embed
        delta   = self.layers(feature)            # (B, N, 11)

        refined = delta.clone()
        refined[..., :8] = delta[..., :8] + anchor[..., :8]   # pos, size, sin/cos

        if not isinstance(time_interval, torch.Tensor):
            time_interval = anchor.new_tensor(time_interval)
        dt = time_interval.clamp(min=1e-3)
        if dt.dim() == 0:
            refined[..., 8:] = delta[..., 8:] / dt + anchor[..., 8:]
        else:
            refined[..., 8:] = delta[..., 8:] / dt[:, None, None] + anchor[..., 8:]

        cls_logits = self.cls_layers(instance_feature)        # (B, N, C)
        quality    = self.quality_layers(feature)             # (B, N, 2)
        return refined, cls_logits, quality


# ---------------------------------------------------------------------------
# Decoder with centerness re-ranking
# ---------------------------------------------------------------------------

class SparseBox3DDecoderV3(nn.Module):
    """
    Reference v3 decoding:
      1. sigmoid(cls) flattened over (anchor × class), top-num_output
      2. score *= sigmoid(centerness), re-sort descending
      3. score_threshold filter
      4. box decode: exp sizes, atan2 yaw
    """

    def __init__(self, num_output: int = 300, score_threshold: float = 0.05):
        super().__init__()
        self.num_output      = num_output
        self.score_threshold = score_threshold

    @torch.no_grad()
    def forward(
        self,
        anchor:      torch.Tensor,             # (B, N, 11)
        cls_logits:  torch.Tensor,             # (B, N, C)
        quality:     torch.Tensor | None = None,  # (B, N, 2) [centerness, yawness]
        instance_id: torch.Tensor | None = None,  # (B, N) long track IDs (tracking)
        trajectories: torch.Tensor | None = None, # (B, N, K, T, 2) motion forecasts
        mode_logits:  torch.Tensor | None = None, # (B, N, K)
    ) -> list[dict]:
        B, N, C = cls_logits.shape
        scores_all = cls_logits.sigmoid()                       # (B, N, C)

        # This lets one anchor contribute under multiple classes
        K = min(self.num_output, N * C)
        flat_scores, indices = scores_all.flatten(1).topk(K, dim=1)  # Flatten the full anchor×class(N×C) score grid and takes the global top-K (B, K)
        cls_ids    = indices % C                                # (B, K)
        anchor_ids = indices // C                               # (B, K)
        
        # Insert the quality/centerness branch: final score = cls_logits.sigmoid() * centerness.sigmoid()
        """
        The intuition: a query can be confident about class but poorly localized; 
        multiplying by centerness down-weights well-classified-but-badly-placed boxes. 
        That's the v3 quality-estimation contribution, and it's why v3's refine emits a third output (quality) that v2's doesn't.
        """
        if quality is not None:
            centerness = quality[..., 0].gather(1, anchor_ids)  # (B, K)
            flat_scores = flat_scores * centerness.sigmoid()
            flat_scores, order = flat_scores.sort(dim=1, descending=True)
            cls_ids    = cls_ids.gather(1, order)
            anchor_ids = anchor_ids.gather(1, order)

        results = []
        for b in range(B):
            scores = flat_scores[b]
            labels = cls_ids[b]
            boxes  = anchor[b, anchor_ids[b]]                   # (K, 11)
            # Track IDs + trajectories follow the same anchor selection as boxes
            tids = instance_id[b, anchor_ids[b]] if instance_id is not None else None
            trajs = trajectories[b, anchor_ids[b]] if trajectories is not None else None
            tmodes = mode_logits[b, anchor_ids[b]] if mode_logits is not None else None

            keep   = scores >= self.score_threshold
            scores, labels, boxes = scores[keep], labels[keep], boxes[keep]
            if tids is not None:
                tids = tids[keep]
            if trajs is not None:
                trajs = trajs[keep]                            # (Kept, K, T, 4) [loc,scale]
                tmodes = tmodes[keep].softmax(dim=-1)          # (Kept, K) mode probs

            if boxes.shape[0] > 0:
                x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
                w = boxes[:, 3].exp()
                l = boxes[:, 4].exp()
                h = boxes[:, 5].exp()
                yaw = torch.atan2(boxes[:, 6], boxes[:, 7])
                vx, vy = boxes[:, 8], boxes[:, 9]
                boxes_3d = torch.stack([x, y, z, w, l, h, yaw, vx, vy], dim=-1)
            else:
                boxes_3d = torch.zeros(0, 9, device=anchor.device)

            res = {
                'boxes_3d':  boxes_3d,
                'scores_3d': scores,
                'labels_3d': labels,
            }
            if tids is not None:
                res['track_ids'] = tids
            if trajs is not None:
                res['trajectories'] = trajs[..., :2]   # (Kept, K, T, 2) lidar-frame disp.
                if trajs.shape[-1] > 2:                 # QCNet-Laplace packs scale too
                    res['traj_scale'] = trajs[..., 2:]
                res['traj_scores']  = tmodes           # (Kept, K) mode probabilities
            results.append(res)
        return results
