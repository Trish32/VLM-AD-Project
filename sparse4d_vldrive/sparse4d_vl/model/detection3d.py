"""
3-D detection-specific modules for Sparse4D — pure PyTorch, MPS-compatible.

SparseBox3DEncoder       : encodes anchor box (11-dim) to embed_dims embedding
SparseBox3DRefinementModule : iterative box refinement MLP + classification head
SparseBox3DDecoder       : post-processing — top-K selection with score threshold

Anchor / box convention  (11 dims):
  [x, y, z,  log_w, log_l, log_h,  sin_yaw, cos_yaw,  vx, vy, vz]

Sizes are stored in **log-space** (matching the reference Sparse4D checkpoint
format).  The decoder applies `.exp()` when producing metric output.

10 nuScenes object classes (consistent with BEVFormer):
  car, truck, construction_vehicle, bus, trailer,
  barrier, motorcycle, bicycle, pedestrian, traffic_cone
"""

from __future__ import annotations

import torch
import torch.nn as nn


CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
NUM_CLASSES = len(CLASS_NAMES)

ANCHOR_DIMS = 11   # x,y,z,log_w,log_l,log_h,sin,cos,vx,vy,vz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_relu_ln_seq(in_dims: int, embed_dims: int) -> list:
    """
    Two-layer MLP matching reference linear_relu_ln(embed_dims, 1, 2, in_dims):
      [Linear(in,D), ReLU, LN(D), Linear(D,D), ReLU, LN(D)]
    Sequential index: 0=Lin, 1=ReLU, 2=LN, 3=Lin, 4=ReLU, 5=LN
    """
    return [
        nn.Linear(in_dims, embed_dims),
        nn.ReLU(inplace=True),
        nn.LayerNorm(embed_dims),
        nn.Linear(embed_dims, embed_dims),
        nn.ReLU(inplace=True),
        nn.LayerNorm(embed_dims),
    ]


class Scale(nn.Module):
    """Per-element learnable scalar multiplier on the raw delta (reference Sparse4D Scale module).
       It lets the network learn how aggressive each coordinate's correction should be.
    """

    def __init__(self, init_val: list | float = 1.0):
        super().__init__()
        # It's a real checkpoint parameter (layers.11.scale), 
        # so it had to exist as a module with the exact key name
        if isinstance(init_val, (list, tuple)):
            self.scale = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        else:
            self.scale = nn.Parameter(torch.tensor([float(init_val)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


# ---------------------------------------------------------------------------
# Anchor box encoder
# ---------------------------------------------------------------------------

class SparseBox3DEncoder(nn.Module):
    """
    Encodes an 11-dim anchor box into an embed_dims positional embedding.

    Each semantic group (position, size, yaw, velocity) is embedded by its
    own two-layer MLP (linear_relu_ln structure), then **summed** and passed
    through a final two-layer MLP (output_fc).

    Key names match the reference checkpoint exactly:
      pos_fc, size_fc, yaw_fc, vel_fc, output_fc — each a 6-element Sequential.
    """

    def __init__(self, embed_dims: int = 256):
        super().__init__()
        D = embed_dims
        # Each semantic group gets its own MLP
        self.pos_fc    = nn.Sequential(*_linear_relu_ln_seq(3, D))
        self.size_fc   = nn.Sequential(*_linear_relu_ln_seq(3, D))
        self.yaw_fc    = nn.Sequential(*_linear_relu_ln_seq(2, D))
        self.vel_fc    = nn.Sequential(*_linear_relu_ln_seq(3, D))
        self.output_fc = nn.Sequential(*_linear_relu_ln_seq(D, D))

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        """
        anchor : (..., 11)  [x,y,z, log_w,log_l,log_h, sin,cos, vx,vy,vz]
        Returns: (..., embed_dims)
        """
        pos = anchor[..., 0:3]    # x, y, z
        siz = anchor[..., 3:6]    # log_w, log_l, log_h
        yaw = anchor[..., 6:8]    # sin_yaw, cos_yaw
        vel = anchor[..., 8:11]   # vx, vy, vz

        # Semantic groups are summed
        feat = (
            self.pos_fc(pos)
            + self.size_fc(siz)
            + self.yaw_fc(yaw)
            + self.vel_fc(vel)
        )
        return self.output_fc(feat)


# ---------------------------------------------------------------------------
# Box refinement module
# ---------------------------------------------------------------------------

class SparseBox3DRefinementModule(nn.Module):
    """
    Iterative box refinement.

    Structure matches the reference checkpoint key names exactly:
      self.layers     = deep MLP → 11 regression targets (+ learnable Scale)
      self.cls_layers = MLP       → num_classes logits

    Anchor format uses log-space sizes: delta is added directly to anchor.
    Yaw delta is handled via angle addition (sin/cos composition).
    """

    def __init__(self, embed_dims: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        D = embed_dims

        # Regression: linear_relu_ln(D, 2, 2) + Linear(11) + Scale(11)
        # indices:  0=Lin, 1=ReLU, 2=Lin, 3=ReLU, 4=LN,
        #           5=Lin, 6=ReLU, 7=Lin, 8=ReLU, 9=LN,
        #           10=Lin(11), 11=Scale(11)
        self.layers = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.LayerNorm(D),
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.LayerNorm(D),
            nn.Linear(D, ANCHOR_DIMS),  # Linear(D, 11)
            Scale([1.0] * ANCHOR_DIMS),  # Learnable Scale(11)
        )

        # Classification: linear_relu_ln(D, 1, 2) + Linear(num_cls)
        # indices: 0=Lin, 1=ReLU, 2=LN, 3=Lin, 4=ReLU, 5=LN, 6=Lin(C)
        self.cls_layers = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.LayerNorm(D),
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.LayerNorm(D),
            nn.Linear(D, num_classes),
        )

    def forward(
        self,
        instance_feature: torch.Tensor,           # (B, N, D)
        anchor:           torch.Tensor,           # (B, N, 11)  current anchor
        anchor_embed:     torch.Tensor | None = None,  # (B, N, D)
        time_interval:    float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        refined_anchor : (B, N, 11)  updated anchor box (log-space sizes)
        cls_logits     : (B, N, num_classes)
        """
        # The box correction is position-aware (it needs to know where the anchor is), but the class identity isn't
        feat  = instance_feature if anchor_embed is None else (instance_feature + anchor_embed)
        # Residual refinement: the network predicts a delta added to the incoming anchor, not the box from scratch
        """
        Scale-gated delta (additive in position/log-size/sin-cos, dt-normalized in velocity) and re-scores it, 
        so boxes converge over stages rather than being predicted once.
        """
        delta = self.layers(feat)              # (B, N, 11) after Scale

        # Position, size, sin/cos: all additive (reference uses simple addition)
        refined = delta.clone()
        refined[..., :6]  = delta[..., :6]  + anchor[..., :6]   # pos(x,y,z) + log_size(log_w,log_l,log_h)
        refined[..., 6:8] = delta[..., 6:8] + anchor[..., 6:8]  # sin_yaw, cos_yaw

        # Velocity: delta is displacement over time_interval → divide to get m/s
        if not isinstance(time_interval, torch.Tensor):
            time_interval = anchor.new_tensor(time_interval)
        # time_interval can be scalar or (B,); clamp to avoid div-by-zero
        dt = time_interval.clamp(min=1e-3)
        # dt can be scalar or per-batch
        if dt.dim() == 0:
            refined[..., 8:] = delta[..., 8:] / dt + anchor[..., 8:]  # velocity
        else:
            refined[..., 8:] = delta[..., 8:] / dt[:, None, None] + anchor[..., 8:]

        cls_logits = self.cls_layers(instance_feature)   # (B, N, C)

        return refined, cls_logits


# ---------------------------------------------------------------------------
# Post-processing decoder
# ---------------------------------------------------------------------------

class SparseBox3DDecoder(nn.Module):
    """
    Converts raw predictions to nuScenes-format 3-D detections.

    Selects top-K anchors by max classification score, converts log-space
    sizes to metric via exp(), and returns a detection dict.
    """

    def __init__(self, num_output: int = 300, score_threshold: float = 0.1):
        super().__init__()
        self.num_output      = num_output
        self.score_threshold = score_threshold

    @torch.no_grad()
    def forward(
        self,
        anchor:     torch.Tensor,     # (B, N, 11)  log-space sizes
        cls_logits: torch.Tensor,     # (B, N, C)
    ) -> list[dict]:
        """
        Returns a list (len=B) of dicts, each with:
          boxes_3d  : (K, 9)  [x, y, z, w, l, h, yaw, vx, vy]  sizes in metres
          scores_3d : (K,)    max class score (sigmoid)
          labels_3d : (K,)    long, class index
        """
        B = anchor.shape[0]
        results = []
        scores_all = cls_logits.sigmoid()  # multi-label-style independent class scoring, matching the focal-loss training.

        for b in range(B):
            scores = scores_all[b]
            max_scores, labels = scores.max(dim=-1)  # 1. max over classes per query → (score, label) per query

            K = min(self.num_output, max_scores.shape[0])
            topk_scores, topk_idx = max_scores.topk(K)  # 2. then topk queries
            topk_labels = labels[topk_idx]
            topk_anchor = anchor[b, topk_idx]   # (K, 11)

            keep = topk_scores >= self.score_threshold
            topk_scores = topk_scores[keep]
            topk_labels = topk_labels[keep]
            topk_anchor = topk_anchor[keep]

            if topk_anchor.shape[0] > 0:
                # Decode the box
                x, y, z   = topk_anchor[:, 0], topk_anchor[:, 1], topk_anchor[:, 2]
                # Sizes: convert from log-space to metric
                w = topk_anchor[:, 3].exp()
                l = topk_anchor[:, 4].exp()
                h = topk_anchor[:, 5].exp()
                sin_y     = topk_anchor[:, 6]
                cos_y     = topk_anchor[:, 7]
                yaw       = torch.atan2(sin_y, cos_y)
                vx, vy    = topk_anchor[:, 8], topk_anchor[:, 9]

                boxes_3d  = torch.stack([x, y, z, w, l, h, yaw, vx, vy], dim=-1)
            else:
                boxes_3d  = torch.zeros(0, 9, device=anchor.device)

            results.append({
                'boxes_3d':  boxes_3d,
                'scores_3d': topk_scores,
                'labels_3d': topk_labels,
            })

        return results
