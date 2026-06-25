"""
Core building blocks for Sparse4D — pure PyTorch, MPS-compatible.

Modules
-------
SparseBox3DKeyPointsGenerator  : 7 fixed + 6 learnable keypoints from a 3-D anchor box
DeformableFeatureAggregation   : multi-view, multi-level, grouped grid_sample
GraphAttention                 : multi-head self-attention over N sparse queries
AsymmetricFFN                  : FFN with optional input-projection (v2 style)
LinearFusionModule             : temporal feature fusion with exponential decay

All coordinate operations use pure NumPy / PyTorch — no mmcv, no C++ extensions.

Keypoint layout (13 points = 7 fixed + 6 learnable)
  Fixed (scale × [W, L, H]):
    0 : centre  (0, 0, 0)
    1 : +W face (0.45, 0, 0)
    2 : −W face (-0.45, 0, 0)
    3 : +L face (0, 0.45, 0)
    4 : −L face (0, -0.45, 0)
    5 : +H face (0, 0, 0.45)
    6 : −H face (0, 0, -0.45)
  Learnable 7-12: predicted by learnable_fc, sigmoid − 0.5 × [W, L, H]

DFA residual_mode="cat": output = cat([proj(agg_feat), instance_feature]) (2D=512).
  The AsymmetricFFN then maps this 512-dim back to 256-dim with an internal
  identity_fc skip connection, matching the reference Sparse4D-v2 design.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed scale values matching the reference Sparse4D-v2 config
_FIX_SCALE_DEFAULT = np.array([
    [0.0,   0.0,   0.0  ],
    [0.45,  0.0,   0.0  ],
    [-0.45, 0.0,   0.0  ],
    [0.0,   0.45,  0.0  ],
    [0.0,  -0.45,  0.0  ],
    [0.0,   0.0,   0.45 ],
    [0.0,   0.0,  -0.45 ],
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Key-point generator
# ---------------------------------------------------------------------------

class SparseBox3DKeyPointsGenerator(nn.Module):
    """
    Generates 13 3-D keypoints per anchor box (7 fixed + 6 learnable).

    anchor           : (B, N, 11)  [x,y,z, log_w,log_l,log_h, sin,cos, vx,vy,vz]
    instance_feature : (B, N, D)   used to predict the 6 learnable offsets
    Returns          : (B, N, 13, 3)  3-D positions in ego/lidar frame
    """

    def __init__(
        self,
        embed_dims:       int   = 256,
        num_learnable_pts: int  = 6,
        fix_scale:        np.ndarray | None = None,
    ):
        super().__init__()
        self.embed_dims        = embed_dims
        self.num_learnable_pts = num_learnable_pts
        fs = _FIX_SCALE_DEFAULT if fix_scale is None else np.array(fix_scale, dtype=np.float32)
        # Register as a buffer so it loads from the checkpoint when present.
        # v2 checkpoints omit it (default kept); v3 trained fix_scale to
        # non-default learned values, so it MUST be loaded from state_dict.
        self.register_buffer('fix_scale', torch.from_numpy(fs.astype(np.float32)))
        self.num_pts           = len(fs) + num_learnable_pts

        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(embed_dims, num_learnable_pts * 3)
            nn.init.zeros_(self.learnable_fc.weight)
            nn.init.zeros_(self.learnable_fc.bias)

    def forward(
        self,
        anchor:           torch.Tensor,        # (B, N, 11)
        instance_feature: torch.Tensor | None = None,  # (B, N, D)
    ) -> torch.Tensor:                          # (B, N, num_pts, 3)
        bs, num_anchor = anchor.shape[:2]

        # Fixed keypoints: scale[None,None] × [W,L,H]  → (B, N, 7, 3)
        fix = self.fix_scale.to(anchor.dtype)               # (7, 3)
        # 1. Scale factors — 7 fixed offsets (fix_scale, e.g. center + 6 face points) 
        # plus 6 learnable offsets the network predicts from instance_feature
        scale = fix.unsqueeze(0).unsqueeze(0).expand(bs, num_anchor, -1, -1)  # (B, N, 7, 3)

        if self.num_learnable_pts > 0 and instance_feature is not None:
            learn = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid() - 0.5  # Gives offsets in (−0.5, 0.5) box-fractions
            )                                                # (B, N, 6, 3)
            scale = torch.cat([scale, learn], dim=-2)        # (B, N, 13, 3)

        # 2. To metric size — multiply by the box's real extent
        # Now each point is an offset in meters relative to box center
        # scale × [metric_W, metric_L, metric_H]  (log-space → exp)
        wlh = anchor[..., 3:6].exp()                         # (B, N, 3)
        key_points = scale * wlh.unsqueeze(-2)               # (B, N, 13, 3)

        # 3. Rotate XY by yaw (rotation around Z)
        sin_yaw = anchor[..., 6]                             # (B, N)
        cos_yaw = anchor[..., 7]
        rot = anchor.new_zeros(bs, num_anchor, 3, 3)
        rot[:, :, 0, 0] =  cos_yaw
        rot[:, :, 0, 1] = -sin_yaw
        rot[:, :, 1, 0] =  sin_yaw
        rot[:, :, 1, 1] =  cos_yaw
        rot[:, :, 2, 2] = 1.0
        # rot: (B, N, 3, 3)  key_points: (B, N, 13, 3) → (B, N, 13, 3, 1)
        key_points = torch.matmul(rot[:, :, None], key_points.unsqueeze(-1)).squeeze(-1)

        # 4. Translate by anchor centre
        center = anchor[..., :3].unsqueeze(-2)               # (B, N, 1, 3)
        return key_points + center                           # Translate — add the box center (B, N, 13, 3)


# ---------------------------------------------------------------------------
# Deformable Feature Aggregation
# ---------------------------------------------------------------------------

class DeformableFeatureAggregation(nn.Module):
    """
    Multi-view, multi-level, grouped deformable feature sampling.

    For each of N anchors:
      1. Generate 7 keypoints in ego frame.
      2. Project keypoints to each camera image via projection_mat.
      3. Sample multi-level FPN features at projected locations (F.grid_sample).
      4. Weighted aggregation across cameras × levels × keypoints, per group.

    When temp_anchor / temp_feature / temp_time_interval are provided (v2
    temporal stages), the same current FPN feature maps are also sampled at
    projected historical anchor locations and fused via LinearFusionModule.

    Groups parameter corresponds to multi-head channels — embed_dims is split
    into num_groups sub-spaces, each with its own attention weights over the
    (cam × level × keypoint) sampling locations.
    """

    def __init__(
        self,
        embed_dims:   int   = 256,
        num_groups:   int   = 8,
        num_levels:   int   = 4,
        num_cams:     int   = 6,
        num_pts:      int   = 7,
        attn_drop:    float = 0.15,
        use_camera_embed: bool = False,
        temporal_fusion: bool = False,
        residual_mode:   str  = "cat",
    ):
        super().__init__()
        assert embed_dims % num_groups == 0
        self.embed_dims      = embed_dims
        self.num_groups      = num_groups
        self.num_levels      = num_levels
        self.num_cams        = num_cams
        self.num_pts         = num_pts
        self.attn_drop       = nn.Dropout(attn_drop)
        self.use_camera_embed = use_camera_embed
        self.residual_mode   = residual_mode

        if use_camera_embed:
            # Camera embed path: weight dim is G*L*P (no num_cams — camera
            # info is added to the feature before weights_fc).
            # Keys match reference: weights_fc, camera_encoder.*
            self.weights_fc = nn.Linear(embed_dims, num_groups * num_levels * num_pts)
            # camera_encoder matches linear_relu_ln(D, 1, 2, 12):
            # [Lin(12,D), ReLU, LN, Lin(D,D), ReLU, LN]
            self.camera_encoder = nn.Sequential(
                nn.Linear(12, embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(embed_dims),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(embed_dims),
            )
        else:
            # No camera embed: weight dim includes num_cams
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * num_pts
            )
            self.camera_encoder = None

        # Optional temporal fusion module (Sparse4D-v2 temporal stages)
        self.linear_fusion = LinearFusionModule(embed_dims) if temporal_fusion else None

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.key_pts_gen = SparseBox3DKeyPointsGenerator(
            embed_dims=embed_dims,
            num_learnable_pts=6,
        )

        nn.init.zeros_(self.weights_fc.weight)
        nn.init.zeros_(self.weights_fc.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # ------------------------------------------------------------------
    # Geometry: project keypoints to all cameras
    # ------------------------------------------------------------------

    @staticmethod
    def project_points(
        keypoints:       torch.Tensor,  # (B, N, P, 3)
        projection_mat:  torch.Tensor,  # (B, N_cam, 4, 4)
        image_wh:        torch.Tensor,  # (B, N_cam, 2)  [W, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        pts_2d : (B, N_cam, N, P, 2)  normalised [0, 1] w.r.t. image_wh
        valid  : (B, N_cam, N, P)     True = keypoint in front of camera and in-frame
        """
        B, N, P, _ = keypoints.shape
        N_cam = projection_mat.shape[1]

        # 1. Homogeneous coords: append a 1 → (B, N, P, 4)
        ones  = torch.ones(*keypoints.shape[:-1], 1, device=keypoints.device,
                           dtype=keypoints.dtype)
        pts_h = torch.cat([keypoints, ones], dim=-1)            # (B, N, P, 4)

        # Expand for cameras: (B, N_cam, N*P, 4)
        pts_h = pts_h.unsqueeze(1).expand(B, N_cam, N, P, 4)
        pts_h = pts_h.reshape(B * N_cam, N * P, 4)

        # Project via (3, 4) sub-matrix of 4×4 that already folds lidar→camera→intrinsics: (B*N_cam, N*P, 3)
        proj = projection_mat[:, :, :3, :].reshape(B * N_cam, 3, 4)
        # 2. Multiply by each camera's projection_mat: bmm gives (u·z, v·z, z) 
        # (B*N_cam, 3, 4) @ (B*N_cam, 4, N*P) → (B*N_cam, 3, N*P)
        pts_3 = torch.bmm(proj, pts_h.transpose(1, 2)).transpose(1, 2)  # (B*N_cam, N*P, 3)

        # 3. Perspective divide by clamped z → pixel (u,v); keep depth=z separately
        z     = pts_3[..., 2:3].clamp(min=1e-5)
        uv    = pts_3[..., :2] / z                               # pixel coords
        depth = pts_3[..., 2]                                    # (B*N_cam, N*P)

        # Reshape
        uv    = uv.reshape(B, N_cam, N, P, 2)
        depth = depth.reshape(B, N_cam, N, P)

        # 4. Normalise to [0, 1] using image dimensions
        wh = image_wh.unsqueeze(2).unsqueeze(3)                  # (B, N_cam, 1, 1, 2)
        pts_2d = uv / wh.clamp(min=1.0)  # (B, N_cam, N, P, 2)

        # A valid mask is also computed but — critically — not used to mask features. It retained for parity and diagnostics
        # grid_sample with padding_mode='zeros' already contributes 0 for out-of-frame points
        valid = (
            (depth  >  0.0) &
            (pts_2d[..., 0] >= 0.0) & (pts_2d[..., 0] <= 1.0) &
            (pts_2d[..., 1] >= 0.0) & (pts_2d[..., 1] <= 1.0)
        )
        return pts_2d, valid

    # ------------------------------------------------------------------
    # Attention weight generation
    # ------------------------------------------------------------------

    def _get_weights(
        self,
        instance_feature: torch.Tensor,          # (B, N, D)
        anchor_embed:     torch.Tensor,          # (B, N, D)
        projection_mat:   torch.Tensor | None = None,  # (B, N_cam, 4, 4)
    ) -> torch.Tensor:
        """
        Predicts the attention weights from instance_feature + anchor_embed
        Returns weights: (B, N, N_cam, N_levels, N_pts, G) normalised.

        Matches the reference layout exactly: the weights_fc output is
        interpreted as (..., level, point, GROUP) — group varies FASTEST —
        and the softmax runs jointly over (cam × level × point) per group.
        No FOV masking (reference relies on grid_sample zero-padding).
        """
        B, N, _ = instance_feature.shape
        G, C, L, P = self.num_groups, self.num_cams, self.num_levels, self.num_pts

        feat = instance_feature + anchor_embed   # (B, N, D)

        if self.camera_encoder is not None and projection_mat is not None:
            # Camera-embed path (v2): per-camera weights, G*L*P per cam.
            # proj[:,:,:3,:] (B,C,3,4) → (B,C,12) → camera_encoder → (B,C,D)
            cam_in  = projection_mat[:, :, :3, :].reshape(B, C, 12)
            cam_emb = self.camera_encoder(cam_in)           # (B, C, D)
            feat_cam = feat.unsqueeze(2) + cam_emb.unsqueeze(1)  # (B, N, C, D)
            raw = self.weights_fc(feat_cam)                 # (B, N, C, L*P*G)
        else:
            # No camera-embed (v1): flat weights over C*L*P*G
            raw = self.weights_fc(feat)                     # (B, N, C*L*P*G)

        w = (
            raw.reshape(B, N, -1, G)        # (B, N, C*L*P, G) — group last/fastest
               .softmax(dim=-2)             # softmax over dim=−2 (jointly across cam×level×point per group)
               .reshape(B, N, C, L, P, G)
        )
        return self.attn_drop(w)

    # ------------------------------------------------------------------
    # Feature sampling via grid_sample
    # ------------------------------------------------------------------

    def _sample_features(
        self,
        feature_maps: list[torch.Tensor],  # [(B*N_cam, D, H_l, W_l)]
        pts_2d:       torch.Tensor,        # (B, N_cam, N, P, 2)  in [0, 1]
    ) -> torch.Tensor:
        """
        Returns sampled : (B, N, N_cam, N_levels, P, G, D_g)
        where D_g = embed_dims // num_groups
        """
        B, N_cam, N, P, _ = pts_2d.shape
        G  = self.num_groups
        D  = self.embed_dims
        Dg = D // G

        # 1. Coord convention: grid_sample expects coords in [-1, 1], we have [0,1]
        grid = 2.0 * pts_2d - 1.0                                # (B, N_cam, N, P, 2)
        grid_flat = grid.reshape(B * N_cam, N, P, 2)             # (B*N_cam, N, P, 2)

        # 2. Grouped sampling — the 256-dim feature is split into 8 groups of 32. 
        sampled_levels = []
        for feat in feature_maps:
            # feat: (B*N_cam, D=256, H, W) → split into groups → (B*N_cam*G, Dg=32, H, W)
            BC, D_, H, W = feat.shape
            feat_g = feat.reshape(BC * G, Dg, H, W)

            # Replicate grid for each group: (B*N_cam*G, N, P, 2)
            grid_g = grid_flat.unsqueeze(1).expand(-1, G, -1, -1, -1)
            grid_g = grid_g.reshape(BC * G, N, P, 2)

            # 3. Each group samples independently (this is the "grouped" in grouped attention, like attention heads): (B*N_cam*G, Dg, N, P)
            sampled = F.grid_sample(
                feat_g, grid_g, mode='bilinear',
                padding_mode='zeros', # No FOV masking: A keypoint projecting outside camera c gets grid coords outside [−1,1], and padding_mode='zeros' makes grid_sample return 0 there
                align_corners=False,  # it changes the pixel-center vs pixel-corner mapping by half a pixel. Get it wrong and every sampled value is subtly off, mAP drops a few points, and nothing crashes
            )

            # Reshape → (B, N_cam, G, Dg, N, P) → permute → (B, N, N_cam, P, G, Dg)
            sampled = sampled.reshape(B, N_cam, G, Dg, N, P)
            sampled = sampled.permute(0, 4, 1, 5, 2, 3).contiguous()  # (B, N, N_cam, P, G, Dg)
            sampled_levels.append(sampled)

        # Loop over the 4 FPN levels, then stack levels: (B, N, N_cam, P, N_levels, G, Dg)
        sampled = torch.stack(sampled_levels, dim=4)               # stack on level dim
        # Permute to (B, N, N_cam, N_levels, P, G, Dg)
        sampled = sampled.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return sampled

    # ------------------------------------------------------------------
    # Grouped weighted aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(
        sampled: torch.Tensor,  # (B, N, N_cam, N_levels, P, G, Dg)
        weights: torch.Tensor,  # (B, N, N_cam, N_levels, P, G)
    ) -> torch.Tensor:
        """Returns (B, N, G*Dg) = (B, N, D)."""
        # Each group keeps its own 32-dim subspace (that's why the sum is over 2,3,4 but not the group axis), then groups concatenate back to 256.
        # Weighted sum over cam, level, point (N_cam, N_levels, P) → (B, N, G, Dg)
        out = (sampled * weights.unsqueeze(-1)).sum(dim=(2, 3, 4))
        B, N, G, Dg = out.shape
        return out.reshape(B, N, G * Dg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        instance_feature:   torch.Tensor,            # (B, N, D)
        anchor:             torch.Tensor,            # (B, N, 11)
        anchor_embed:       torch.Tensor,            # (B, N, D)
        feature_maps:       list[torch.Tensor],      # [(B*N_cam, D, H_l, W_l)]×4
        projection_mat:     torch.Tensor,            # (B, N_cam, 4, 4)
        image_wh:           torch.Tensor,            # (B, N_cam, 2)
        temp_instance_feature: torch.Tensor | None = None,  # (B, N_temp, D)
        temp_anchor:           torch.Tensor | None = None,  # (B, N_temp, 11) in current frame
        temp_time_interval:    torch.Tensor | None = None,  # (B,) seconds
    ) -> torch.Tensor:                               # (B, N, D)

        B, N, D = instance_feature.shape

        # 1. Generate keypoints (instance_feature feeds the learnable pts)
        keypoints = self.key_pts_gen(anchor, instance_feature)  # (B, N, 13, 3)

        # 2. Project to image plane
        pts_2d, _ = self.project_points(keypoints, projection_mat, image_wh)
        # pts_2d: (B, N_cam, N, 13, 2)

        # 3. Sample features from current frame
        sampled = self._sample_features(feature_maps, pts_2d)
        # sampled: (B, N, N_cam, N_levels, 13, G, Dg)

        # 4. Attention weights (pass projection_mat for camera_embed path)
        weights = self._get_weights(instance_feature, anchor_embed, projection_mat)
        # weights: (B, N, N_cam, N_levels, 13, G)

        # 5. Aggregate
        output = self._aggregate(sampled, weights)    # (B, N, D)

        proj = self.output_proj(output)                      # (B, N, D)
        if self.residual_mode == "cat":
            proj = torch.cat([proj, instance_feature], dim=-1)  # cat with the residual instance_feature (B, N, 2D)
        return proj


# ---------------------------------------------------------------------------
# Graph Attention (GNN — instance-to-instance self-attention)
# ---------------------------------------------------------------------------

class GraphAttention(nn.Module):
    """
    Standard multi-head self-attention over N sparse instance queries.
    Used for both 'gnn' (current-only) and 'temp_gnn' (current + temporal)
    operation steps in the Sparse4D decoder.

    For 'temp_gnn': pass the concatenated [current, temporal] features as
    `key_value` while keeping `query` = current features only. The returned
    tensor has shape (B, N_curr, D) — only the current queries are updated.
    """

    def __init__(self, embed_dims: int = 256, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(embed_dims, num_heads,
                                              dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query:     torch.Tensor,            # (B, N_q, D)
        key_value: torch.Tensor | None = None,   # (B, N_kv, D)  or None → self-attn
        query_pos: torch.Tensor | None = None,   # (B, N_q, D)  positional embed
        key_pos:   torch.Tensor | None = None,   # (B, N_kv, D) positional embed
    ) -> torch.Tensor:                      # (B, N_q, D)
        """
        Matches mmcv MultiheadAttention convention: positional embeddings are
        added to query and key (NOT value) before attention.  When key_value
        is None this is self-attention and key_pos defaults to query_pos.
        """
        if key_value is None:
            key_value = query
            if key_pos is None:
                key_pos = query_pos
        q = query if query_pos is None else query + query_pos
        k = key_value if key_pos is None else key_value + key_pos
        out, _ = self.attn(q, k, key_value, need_weights=False)
        return self.dropout(out)


# ---------------------------------------------------------------------------
# AsymmetricFFN  (Sparse4D-v2 style)
# ---------------------------------------------------------------------------

class AsymmetricFFN(nn.Module):
    """
    FFN matching the reference Sparse4D AsymmetricFFN structure.

    Checkpoint key structure:
      pre_norm.*       : LayerNorm(in_dims)
      fc1.*            : Linear(in_dims, ffn_dims)        [layers.0.0 in reference]
      fc2.*            : Linear(ffn_dims, embed_dims)     [layers.1   in reference]
      identity_fc.*    : Linear(in_dims, embed_dims)      (only when in_dims != embed_dims)

    When residual_mode="cat" in the DFA, in_dims = embed_dims * 2 = 512.
    The identity_fc maps the 512-dim input to 256-dim for the skip connection.
    """

    def __init__(
        self,
        embed_dims:  int   = 256,
        in_dims:     int | None = None,   # if None → same as embed_dims
        ffn_dims:    int   = 1024,
        dropout:     float = 0.1,
    ):
        super().__init__()
        in_d = in_dims if in_dims is not None else embed_dims

        self.pre_norm    = nn.LayerNorm(in_d)
        self.fc1         = nn.Linear(in_d, ffn_dims)
        self.fc2         = nn.Linear(ffn_dims, embed_dims)
        self.drop        = nn.Dropout(dropout)
        self.act         = nn.ReLU(inplace=True)
        self.identity_fc = nn.Linear(in_d, embed_dims) if in_d != embed_dims else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.pre_norm(x)
        # Reference AsymmetricFFN: the identity skip is the PRE-NORMED x
        out = self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
        return self.identity_fc(x) + out


# ---------------------------------------------------------------------------
# Linear Fusion Module (temporal feature fusion)
# ---------------------------------------------------------------------------

class LinearFusionModule(nn.Module):
    """
    Fuses current and one set of historical features using an exponential
    time-decay weight and a learnable linear layer.

    w_temp = alpha^(|Δt| × beta),  then output = fc([curr, w_temp * temp]).

    alpha=0.9, beta=10 → for Δt=0.5 s: w ≈ 0.9^5 ≈ 0.59.
    """

    def __init__(self, embed_dims: int = 256, alpha: float = 0.9, beta: float = 10.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.fc    = nn.Linear(embed_dims * 2, embed_dims)

    def forward(
        self,
        curr_feat:   torch.Tensor,   # (B, N, D)
        temp_feat:   torch.Tensor,   # (B, N_temp, D)
        time_interval: torch.Tensor, # (B,)  seconds
    ) -> torch.Tensor:               # (B, N, D)
        # Time-decay weight: scalar per batch element
        decay = self.alpha ** (time_interval.abs() * self.beta)   # (B,)
        decay = decay[:, None, None]                               # (B, 1, 1)

        # If N_temp != N (temporal subset), aggregate to match current N
        if temp_feat.shape[1] != curr_feat.shape[1]:
            # Simple mean-pool temporal features to match current query count
            # In practice, the caller aligns sizes via the instance bank
            temp_feat = temp_feat.mean(dim=1, keepdim=True).expand_as(curr_feat)

        weighted_temp = temp_feat * decay
        fused = torch.cat([curr_feat, weighted_temp], dim=-1)     # (B, N, 2D)
        return self.fc(fused)
