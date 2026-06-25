"""
Precomputed BEV pooling — the core efficiency innovation of MIT BEVFusion,
reproduced in pure PyTorch (no CUDA kernel).

Two ideas from the paper:
1. INTERVAL REDUCTION: instead of scattering each frustum point independently,
   sort points by their target BEV cell, cumulative-sum the features, and take
   differences at cell boundaries (QuickCumsum). One cumsum + one gather replaces
   a random-access scatter-add.
2. PRECOMPUTATION: the frustum-point -> BEV-cell mapping depends only on the
   (fixed) camera calibration, so the gather order, cell boundaries, and output
   indices are computed ONCE and cached, then reused every frame.

Numerically ~equivalent to a per-cell sum-pool (an index_add_ scatter-sum); this
only changes *how* the per-cell sum is computed.

WHY cumsum here, given index_add_ works on MPS? NOT for speed — benchmarked on
M3 Max (real det-config frame, Nprime~2.0M, C=80), index_add_ is actually FASTER:
  reduction-only (indices precomputed):  cumsum 30.4 ms  vs  index_add_ 21.6 ms
  full pooling from scratch:             cumsum 48.8 ms  vs  index_add_ 29.6 ms
The "avoid atomic contention" win is CUDA-specific; on MPS index_add_ is efficient
and cumsum pays for a 2M-row prefix sum + the argsort. We keep cumsum for (a)
FIDELITY — it is the paper's named QuickCumsum algorithm we set out to reproduce —
and (b) DETERMINISM — it sums in a fixed sorted order, whereas index_add_'s fast
path is run-to-run nondeterministic (our activation-diff validation relied on
determinism). Caveat: deterministic != more accurate — the global prefix sum here
has worse float32 cancellation than per-cell index_add_ (|diff| ~6e-3 on this frame).
PRECOMPUTATION (idea 2) is the real, hardware-agnostic speedup and helps BOTH paths
(it caches the expensive argsort): cumsum 48.8->30.4 ms, index_add_ 29.6->21.6 ms.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BEVPool(nn.Module):
    def __init__(self, dx, bx, nx, precompute=True):
        super().__init__()
        # dx/bx are (3,) tensors, nx is (3,) long  (x, y, z grid)
        self.register_buffer('dx', dx, persistent=False)
        self.register_buffer('bx', bx, persistent=False)
        self.register_buffer('nx', nx, persistent=False)
        self.precompute = precompute
        self._cached = False
        # cached geometry-derived index tensors
        self.gather_idx = None      # (M,) indices into flattened x (kept+sorted)
        self.boundary = None        # (M,) bool: start of each BEV-cell interval
        self.out_flat = None        # (num_cells,) flat output index per interval
        self.shape = None           # (B, nz, nx, ny)

    def _build(self, geom):
        """Precompute the gather order / interval boundaries / output indices from a
        FIXED geometry. Everything here is index bookkeeping (no features), so it can be
        cached and reused every frame. geom: (B, N, D, H, W, 3) metric xyz."""
        B, N, D, H, W, _ = geom.shape
        Nprime = B * N * D * H * W           # total number of frustum points
        nx0, nx1, nx2 = int(self.nx[0]), int(self.nx[1]), int(self.nx[2])   # BEV cells (x, y, z)

        # metric xyz -> integer BEV cell (gx, gy, gz) index: idx = floor((p - (bx - dx/2)) / dx)
        g = ((geom - (self.bx - self.dx / 2.0)) / self.dx).long().view(Nprime, 3)   # (Nprime, 3) [gx,gy,gz]
        # per-point batch id (points are laid out batch-major)
        batch_ix = torch.cat([torch.full((Nprime // B, 1), b, device=geom.device, dtype=torch.long)
                              for b in range(B)])                                    # (Nprime, 1)
        coords = torch.cat([g, batch_ix], 1)  # (Nprime, 4) [gx,gy,gz,b]
        # drop points that fall outside the BEV grid on any axis
        kept = ((coords[:, 0] >= 0) & (coords[:, 0] < nx0) &
                (coords[:, 1] >= 0) & (coords[:, 1] < nx1) &
                (coords[:, 2] >= 0) & (coords[:, 2] < nx2))                          # (Nprime,) bool
        keep_idx = torch.nonzero(kept, as_tuple=False).squeeze(1)   # (M,) rows of in-bounds points
        coords = coords[keep_idx]                                  # (M, 4)
        gx, gy, gz, b = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]   # each (M,)
        # RANK: flatten cell+batch (gx,gy,gz,b) to one int so sorting groups same-cell points contiguously
        # collapse a multi-dim coordinate to one integer so equality/sorting is trivial
        ranks = gx * (nx1 * nx2 * B) + gy * (nx2 * B) + gz * B + b   # (M,)
        order = ranks.argsort()             # (M,) permutation that groups by cell
        # After argsort, every cell's points sit in a contiguous block
        # Two points with the same rank fall into the same BEV cell. Different rank = different cell.
        ranks_s = ranks[order]              # (M,) sorted ranks
        # END-of-interval mask (QuickCumsum): True where the NEXT rank differs,
        # plus the final element. cumsum is read at interval ends, then differenced.
        boundary = torch.ones(ranks_s.shape[0], dtype=torch.bool, device=geom.device)   # (M,)
        boundary[:-1] = ranks_s[1:] != ranks_s[:-1]  # mark interval ENDS
        coords_s = coords[order]            # (M, 4) coords in sorted order
        cb = coords_s[boundary]            # (K, 4) one representative coord per occupied cell (K cells)
        # flat output index per cell, in (b, z, x, y) layout to match the canvas below
        out_flat = ((cb[:, 3] * nx2 + cb[:, 2]) * nx0 + cb[:, 0]) * nx1 + cb[:, 1]   # (K,)

        self.gather_idx = keep_idx[order]   # (M,) map sorted position -> original feature row
        self.boundary = boundary           # (M,) interval-end mask
        self.out_flat = out_flat           # (K,) destination cell per interval
        self.shape = (B, nx2, nx0, nx1)    # canvas dims (B, z, x, y)
        self._cached = True

    def forward(self, geom, x):
        # x: (B, N, D, H, W, C) features to splat; geom: (B, N, D, H, W, 3) where each lands
        C = x.shape[-1]
        x = x.reshape(-1, C)               # (Nprime, C) flatten to one row per frustum point
        if not (self.precompute and self._cached):
            self._build(geom)              # (re)compute index bookkeeping
            if not self.precompute:
                pass  # rebuild every call when precompute disabled
        x_s = x[self.gather_idx]           # (M, C) keep in-bounds rows, in cell-sorted order
        xc = x_s.cumsum(0)                 # (M, C) running totals along the sorted axis (rows), Collapses downward vertically
        xk = xc[self.boundary]             # (K, C) sample that running total at each cell's LAST point
        xk = torch.cat((xk[:1], xk[1:] - xk[:-1]))   # (K, C) difference ends -> per-cell sums
        B, nz, nx0, nx1 = self.shape
        canvas = x.new_zeros((B * nz * nx0 * nx1, C))   # (B*z*x*y, C) flat BEV volume
        canvas[self.out_flat] = xk         # scatter each cell sum to its flat index
        final = canvas.view(B, nz, nx0, nx1, C).permute(0, 4, 1, 2, 3)  # (B, C, nz, nx, ny)
        return torch.cat(final.unbind(dim=2), 1)  # collapse z into channels -> (B, C*nz, nx, ny)
