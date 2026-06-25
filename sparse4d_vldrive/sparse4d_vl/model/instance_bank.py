"""
InstanceBank — sparse anchor/query management for Sparse4D.

Matches the reference Sparse4D-v2 temporal flow:

  get()    → returns the 900 FRESH learnable priors (stage 0 always starts
             from the K-means anchors) plus the cached 600 temporal instances
             (projected into the current lidar frame) separately.
  update() → called AFTER the single-frame decoder stage: keeps the top
             (900-600)=300 fresh predictions and concatenates the 600 cached
             temporal instances in front: [cached 600, selected 300].
  cache()  → stores the top-600 most-confident final instances; confidences
             of re-used temporal slots are max(prev*decay, new).
This order matters: the temporal merge happening after stage 0 (not before) 
					is what makes the 600 cached queries carry history while 300 fresh ones discover new objects. 

Anchor layout (11 dims):
  [x, y, z, log_w, log_l, log_h, sin_yaw, cos_yaw, vx, vy, vz]

Temporal anchor projection (prev lidar frame → current lidar frame):
  T = inv(lidar2global_curr) @ lidar2global_prev
  pos  = R @ (pos + vel·Δt) + t
  [cos, sin] = R[:2,:2] @ [cos, sin]       (yaw rotates with the frame)
  vel  = R[:3,:3] @ vel
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Default anchor initialisation
# ---------------------------------------------------------------------------

def _default_anchors(num_anchor: int = 900) -> torch.Tensor:
    """
    Random K-means-style anchor initialisation when no .npy file is provided.
    In practice these are overridden by checkpoint anchors.
    """
    torch.manual_seed(0)
    N = num_anchor

    xyz      = torch.randn(N, 3) * torch.tensor([10.0, 10.0, 0.5])
    log_wlh  = torch.tensor([2.0, 4.5, 1.7]).log().unsqueeze(0).expand(N, -1).clone()
    log_wlh += torch.randn(N, 3) * 0.1
    yaw      = torch.rand(N) * 2 * math.pi
    sin_yaw  = yaw.sin().unsqueeze(-1)
    cos_yaw  = yaw.cos().unsqueeze(-1)
    vel      = torch.zeros(N, 3)

    return torch.cat([xyz, log_wlh, sin_yaw, cos_yaw, vel], dim=-1)   # (N, 11)


# ---------------------------------------------------------------------------
# InstanceBank
# ---------------------------------------------------------------------------

class InstanceBank(nn.Module):
    """
    Args
    ----
    num_anchor        : total number of object queries (default 900)
    embed_dims        : feature dimension
    num_temp_instances: queries recycled from cache each frame (0 = v1 single-frame)
    confidence_decay  : multiplicative decay on cached confidence per frame
    max_time_interval : reset cache when gap between frames exceeds this (s)
    anchor_path       : optional .npy file with K-means anchors (N, 11)
    """

    def __init__(
        self,
        num_anchor:         int   = 900,
        embed_dims:         int   = 256,
        num_temp_instances: int   = 0,
        confidence_decay:   float = 0.6,
        max_time_interval:  float = 2.0,
        default_time_interval: float = 0.5,
        anchor_path:        str | None = None,
        v3_yaw_projection_bug: bool = False,
    ):
        super().__init__()
        self.num_anchor         = num_anchor
        self.embed_dims         = embed_dims
        self.num_temp_instances = min(num_temp_instances, num_anchor)
        self.confidence_decay   = confidence_decay
        self.max_time_interval  = max_time_interval
        self.default_time_interval = default_time_interval
        # Sparse4D-v3 reference has a known bug ("# TODO: Fix bug" in
        # anchor_projection): the rotated [cos, sin] pair is written into
        # slots [6,7] = [sin, cos] — SWAPPED. The v3 checkpoint was trained
        # with this, so faithful reproduction must replicate it.
        self.v3_yaw_projection_bug = v3_yaw_projection_bug

        # Learnable anchor priors and instance features
        if anchor_path is not None:
            import numpy as np
            data = np.load(anchor_path)
            assert data.shape == (num_anchor, 11), \
                f'Expected ({num_anchor}, 11) anchor file, got {data.shape}'
            anchors_init = torch.from_numpy(data.astype('float32'))
        else:
            anchors_init = _default_anchors(num_anchor)

        self.anchors          = nn.Parameter(anchors_init)          # (N, 11)
        self.instance_feature = nn.Parameter(
            torch.zeros(num_anchor, embed_dims)
        )
        nn.init.xavier_uniform_(self.instance_feature.data.view(1, num_anchor, embed_dims))

        # Runtime temporal state (plain attributes, NOT in state_dict)
        self.cached_feature:    torch.Tensor | None = None   # (1, 600, D)
        self.cached_anchor:     torch.Tensor | None = None   # (1, 600, 11)
        self.confidence:        torch.Tensor | None = None   # (1, 600)
        self._prev_lidar2global: torch.Tensor | None = None  # (4, 4)
        self._prev_timestamp:    float | None = None
        self._temporal_merged = False   # True when update() placed cache in slots 0..599

        # Tracking state (Sparse4D-v3 detection-and-tracking; inference only).
        # instance_id is padded to num_anchor: first num_temp_instances are the
        # IDs carried by the cache, the rest are -1 (unassigned).
        self.instance_id:     torch.Tensor | None = None   # (1, num_anchor) long
        self.prev_id:         int = 0                       # next fresh track ID
        self.temp_confidence: torch.Tensor | None = None   # (1, num_anchor) cache-selection conf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_state(self):
        """Call at the start of each new scene."""
        self.cached_feature     = None
        self.cached_anchor      = None
        self.confidence         = None
        self._prev_lidar2global = None
        self._prev_timestamp    = None
        self._temporal_merged   = False
        # tracking
        self.instance_id        = None
        self.prev_id            = 0
        self.temp_confidence    = None

    def get(
        self,
        batch_size:    int,
        ego2global:    torch.Tensor | None = None,  # (B, 4, 4) current ego→global
        lidar2ego:     torch.Tensor | None = None,  # (4, 4) or (B, 4, 4)
        timestamp:     float | None = None,         # current lidar timestamp (s)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None,
               torch.Tensor | None, float]:
        """
        Returns
        -------
        instance_feature : (B, N, D)       fresh learnable priors
        anchor           : (B, N, 11)      fresh K-means anchors
        temp_feature     : (B, 600, D) | None   cached features
        temp_anchor      : (B, 600, 11)| None   cached anchors in CURRENT lidar frame
        time_interval    : float           Δt to previous frame (s)
        """
        B = batch_size
        # 1. Fresh priors: the learnable instance_feature and anchors
        # stage 0 always starts cold from the priors, regardless of history
        instance_feature = self.instance_feature.unsqueeze(0).expand(B, -1, -1)
        anchor           = self.anchors.unsqueeze(0).expand(B, -1, -1)

        # Time interval
        if (timestamp is None or self._prev_timestamp is None):
            time_interval = self.default_time_interval
        else:
            dt = timestamp - self._prev_timestamp  # The time gap to the previous frame
            # clamped — if it's 0 or exceeds max_time_interval (2s, e.g. a scene cut), fall back to the default 0.5s
            if dt == 0 or abs(dt) > self.max_time_interval:
                time_interval = self.default_time_interval
            else:
                time_interval = dt

        # Drop stale cache (scene change / too-long gap)
        if (self.cached_anchor is not None and timestamp is not None
                and self._prev_timestamp is not None
                and abs(timestamp - self._prev_timestamp) > self.max_time_interval):
            self.reset_state()

        # 2. The cached anchors get geometrically projected into the current frame
        temp_feature, temp_anchor = None, None
        if self.cached_anchor is not None:
            # Project cached anchors into the current lidar frame
            lidar2global_curr = self._build_lidar2global(ego2global, lidar2ego)
            if lidar2global_curr is not None and self._prev_lidar2global is not None:
                T = torch.linalg.inv(lidar2global_curr) @ self._prev_lidar2global  # prev-lidar → curr-lidar
                self.cached_anchor = self._project_anchors(
                    self.cached_anchor, T.to(self.cached_anchor.dtype),
                    time_interval,
                    yaw_swap=self.v3_yaw_projection_bug,
                )
                self._prev_lidar2global = lidar2global_curr
            temp_feature = self.cached_feature
            temp_anchor  = self.cached_anchor

        return instance_feature, anchor, temp_feature, temp_anchor, time_interval

    def update(
        self,
        instance_feature: torch.Tensor,  # (B, N, D)   after single-frame stage
        anchor:           torch.Tensor,  # (B, N, 11)
        cls_logits:       torch.Tensor,  # (B, N, C)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Merge cached temporal instances after the single-frame decoder stage:
        keep the top-(N - 600) fresh predictions, place the 600 cached
        instances FIRST: [cached 600, selected 300].
        """
        if self.cached_feature is None:
            self._temporal_merged = False
            return instance_feature, anchor

        N_fresh = self.num_anchor - self.num_temp_instances  # 900 - 600 = 300
        conf = cls_logits.max(dim=-1).values                  # per-query confidence (B, N)
        topk_idx = conf.topk(N_fresh, dim=1).indices          # keep best 300 fresh (B, N_fresh)

        idx_f = topk_idx.unsqueeze(-1).expand(-1, -1, instance_feature.shape[-1])
        idx_a = topk_idx.unsqueeze(-1).expand(-1, -1, anchor.shape[-1])
        sel_feature = instance_feature.gather(1, idx_f)       # (B, N_fresh, D)
        sel_anchor  = anchor.gather(1, idx_a)                 # (B, N_fresh, 11)

        # The cache goes first (slots 0–599), the 300 best fresh detections after. 
        # This ordering is contractual: the temporal stages' temp_gnn attends to slots [:600] as "memory," 
        # so the cache must occupy those slots.
        instance_feature = torch.cat([self.cached_feature, sel_feature], dim=1)  # [600 cached, 300 fresh]
        anchor           = torch.cat([self.cached_anchor,  sel_anchor],  dim=1)
        self._temporal_merged = True  # records that the merge happened, which cache() later needs
        return instance_feature, anchor

    def cache(
        self,
        instance_feature: torch.Tensor,  # (B, N, D)   final decoder output
        anchor:           torch.Tensor,  # (B, N, 11)
        cls_logits:       torch.Tensor,  # (B, N, C)
        ego2global:       torch.Tensor | None = None,  # (B, 4, 4)
        lidar2ego:        torch.Tensor | None = None,  # (4,4) or (B,4,4)
        timestamp:        float | None = None,
    ):
        """
        Store the top-600 most-confident instances for the next frame.
        Temporal slots (0..599) keep max(prev_conf * decay, new_conf).
        """
        if self.num_temp_instances <= 0:
            return

        feat = instance_feature.detach()
        anch = anchor.detach()
        conf = cls_logits.detach().max(dim=-1).values.sigmoid()   # (B, N)

        # a slot that was confident last frame keeps max(prev·0.6, new). 
        # This gives temporal hysteresis — an object briefly occluded (low new_conf this frame) survives on its decayed prior confidence instead of being instantly dropped, 
        # but a slot that stays unconfirmed decays geometrically and falls out of the top-600. 
        if self.confidence is not None and self._temporal_merged:  # temporal slots already had history
            N_temp = self.num_temp_instances
            conf[:, :N_temp] = torch.maximum(
                self.confidence * self.confidence_decay, conf[:, :N_temp]
            )

        # Full-frame confidence used by tracking's update_instance_id so the
        # kept-600 ID selection matches this cache's feature/anchor selection.
        self.temp_confidence = conf                                # (B, N)

        N_temp = self.num_temp_instances
        topk_conf, topk_idx = conf.topk(N_temp, dim=1)            # (B, 600)
        idx_f = topk_idx.unsqueeze(-1).expand(-1, -1, feat.shape[-1])
        idx_a = topk_idx.unsqueeze(-1).expand(-1, -1, anch.shape[-1])

        self.confidence     = topk_conf                            # (B, 600)
        self.cached_feature = feat.gather(1, idx_f)                # (B, 600, D)
        self.cached_anchor  = anch.gather(1, idx_a)                # (B, 600, 11)

        # It stores prev_lidar2global and prev_timestamp so next frame's get() can build T and dt
        l2g = self._build_lidar2global(ego2global, lidar2ego)
        if l2g is not None:
            self._prev_lidar2global = l2g
        self._prev_timestamp = timestamp

    # ------------------------------------------------------------------
    # Tracking: track-ID propagation (Sparse4D-v3, inference only)
    # ------------------------------------------------------------------

    def get_instance_id(
        self,
        cls_logits: torch.Tensor,        # (B, N, C)  final-stage class logits
        threshold:  float | None = None, # only mint IDs for conf >= threshold
    ) -> torch.Tensor:                   # (B, N) long track IDs (-1 = none)
        """
        Assign a persistent track ID to every current query.  Cached temporal
        instances (slots 0..num_temp-1, placed there by update()) INHERIT the ID
        they carried last frame; fresh, confident detections get new sequential
        IDs.  Must be called AFTER cache() (which sets temp_confidence).
        """
        confidence = cls_logits.max(dim=-1).values.sigmoid()      # (B, N)
        instance_id = confidence.new_full(confidence.shape, -1).long()

        # Inherit last frame's IDs into the leading (cached) slots
        if (self.instance_id is not None
                and self.instance_id.shape[0] == instance_id.shape[0]):
            n = self.instance_id.shape[1]
            instance_id[:, :n] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new = int(mask.sum())
        new_ids = torch.arange(num_new, device=instance_id.device) + self.prev_id
        instance_id[mask] = new_ids
        self.prev_id += num_new

        if self.num_temp_instances > 0:
            self._update_instance_id(instance_id)
        return instance_id

    def _update_instance_id(self, instance_id: torch.Tensor):
        """Keep the top-num_temp IDs (by the same conf cache() selected) and pad
        to num_anchor with -1, so next frame's leading slots align with the
        cached features/anchors."""
        temp_conf = self.temp_confidence
        if temp_conf is None:
            return
        _, idx = temp_conf.topk(self.num_temp_instances, dim=1)    # (B, 600)
        kept = instance_id.gather(1, idx)                          # (B, 600)
        pad = instance_id.new_full(
            (instance_id.shape[0], self.num_anchor - self.num_temp_instances), -1)
        self.instance_id = torch.cat([kept, pad], dim=1)          # (B, num_anchor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lidar2global(
        ego2global: torch.Tensor | None,
        lidar2ego:  torch.Tensor | None,
    ) -> torch.Tensor | None:
        if ego2global is None:
            return None
        e2g = ego2global[0] if ego2global.dim() == 3 else ego2global  # (4, 4)
        if lidar2ego is not None:
            l2e = lidar2ego[0] if lidar2ego.dim() == 3 else lidar2ego
            return e2g @ l2e
        return e2g

    @staticmethod
    def _project_anchors(
        anchors:       torch.Tensor,  # (B, N, 11)  in prev lidar frame
        T_src2dst:     torch.Tensor,  # (4, 4)      prev lidar → curr lidar
        time_interval: float,
        yaw_swap:      bool = False,
    ) -> torch.Tensor:
        """
        Reference SparseBox3DKeyPointsGenerator.anchor_projection:
          pos  = R @ (pos + vel·Δt) + t  # position advances along its own velocity (vel·Δt) and is rigid-transformed
          [cos, sin] = R[:2,:2] @ [cos, sin]  # rotate heading by (the R[:2,:2] planar part)
          vel  = R @ vel  # velocity is a direction vector so it rotates but doesn't translate

        yaw_swap=True replicates the v3 reference bug: the rotated
        [cos', sin'] pair is written to slots [6,7] = [sin, cos] (swapped).
        The v3 checkpoint was trained with this behaviour.
        """
        projected = anchors.clone()
        R = T_src2dst[:3, :3]                                     # (3, 3)
        t = T_src2dst[:3, 3]                                      # (3,)

        vel = anchors[..., 8:11]                                  # (B, N, 3)
        pos = anchors[..., 0:3] + vel * time_interval             # velocity comp
        projected[..., 0:3] = pos @ R.T + t

        # Yaw: rotate [cos, sin] pair by the planar rotation R[:2,:2]
        cos_sin = torch.stack([anchors[..., 7], anchors[..., 6]], dim=-1)  # (B,N,2) [cos,sin]
        cos_sin = cos_sin @ R[:2, :2].T
        if yaw_swap:
            projected[..., 6] = cos_sin[..., 0]   # cos → sin slot (v3 ref bug)
            projected[..., 7] = cos_sin[..., 1]   # sin → cos slot
        else:
            projected[..., 7] = cos_sin[..., 0]   # cos
            projected[..., 6] = cos_sin[..., 1]   # sin

        projected[..., 8:11] = vel @ R.T
        return projected
