"""
Hard voxelization (mean-reduce) for MIT BEVFusion, pure PyTorch.
("Hard" = fixed caps, vs. "dynamic" which keeps everything.)
Keeps up to max_num_points per voxel (first-come), then mean-pools features
(voxelize_reduce). Emits coords (b, x, y, z) to match sparse_shape [X,Y,Z].
"""
from __future__ import annotations

import torch


@torch.no_grad()
def voxelize_mean(points, voxel_size, point_cloud_range, max_num_points, max_voxels):
    """points: (N, C) with C=5 = [x, y, z, intensity, Δt].
    Returns feats (M, C) = mean of the (capped) points per occupied voxel, and
    coords (M, 3) = the integer [x, y, z] cell of each voxel. M = #occupied voxels."""
    device = points.device
    vs = torch.as_tensor(voxel_size, device=device, dtype=points.dtype)        # (3,) voxel size
    pcr = torch.as_tensor(point_cloud_range, device=device, dtype=points.dtype)  # (6,) [xmin..zmax]
    grid = ((pcr[3:] - pcr[:3]) / vs).round().long()       # (3,) grid dims (nx, ny, nz)
    nx, ny, nz = int(grid[0]), int(grid[1]), int(grid[2])

    # which voxel each point falls in: floor((p - min) / voxel_size)
    cxyz = ((points[:, :3] - pcr[:3]) / vs).floor().long()  # (N, 3) integer cell [x, y, z]
    # keep only points inside the grid on all three axes
    inrange = ((cxyz[:, 0] >= 0) & (cxyz[:, 0] < nx) &
               (cxyz[:, 1] >= 0) & (cxyz[:, 1] < ny) &
               (cxyz[:, 2] >= 0) & (cxyz[:, 2] < nz))        # (N,) bool
    points = points[inrange]                                # (N', C)
    cxyz = cxyz[inrange]                                    # (N', 3)
    if points.shape[0] == 0:
        return (points.new_zeros((0, points.shape[1])),
                points.new_zeros((0, 3), dtype=torch.long))

    # flatten 3D cell -> one int, then unique to enumerate occupied voxels
    flat = (cxyz[:, 0] * ny + cxyz[:, 1]) * nz + cxyz[:, 2]  # (N',) ravel_multi_index
    uniq, inv = torch.unique(flat, return_inverse=True)     # uniq (M,); inv (N',) point -> voxel id
    M = uniq.numel()                                        # number of occupied voxels

    # --- intra-voxel rank: enforce the max_num_points-per-voxel cap (first-come) ---
    order = torch.argsort(inv, stable=True)    # (N',) group points by voxel; stable keeps arrival order
    inv_s = inv[order]                         # (N',) voxel ids in grouped order
    counts = torch.bincount(inv_s, minlength=M)            # (M,) points per voxel
    # (cumsum - counts) = exclusive prefix sum = start offset of each voxel's block in the sorted array;
    # indexing by inv_s broadcasts that start to every row, so arange - start = position WITHIN the voxel.
    rank = torch.arange(points.shape[0], device=device) - (
        torch.cumsum(counts, 0) - counts)[inv_s]            # (N',) 0,1,2,... within each voxel
    keep = rank < max_num_points               # (N',) keep only the first max_num_points per voxel
    sel = order[keep]                          # original row indices of the kept points
    inv_keep = inv[sel]                        # (#kept,) their voxel ids

    # --- mean-pool the kept points per voxel (this IS the "VFE"(voxel feature encoder): just an average) ---
    C = points.shape[1]
    summ = points.new_zeros((M, C)).index_add_(0, inv_keep, points[sel])   # (M, C) per-voxel sum
    cnt = torch.zeros(M, device=device, dtype=points.dtype).index_add_(
        0, inv_keep, torch.ones(sel.shape[0], device=device, dtype=points.dtype))   # (M,) per-voxel count
    feats = summ / cnt.clamp(min=1).unsqueeze(1)            # (M, C) mean feature per voxel

    # voxel coords (x,y,z): take the cell of the first (min-index) point that hit each voxel
    first = torch.zeros(M, dtype=torch.long, device=device).scatter_reduce(
        0, inv, torch.arange(points.shape[0], device=device), reduce='amin', include_self=False)  # (M,)
    coords = cxyz[first]  # (M, 3) integer [x, y, z] per voxel

    # global cap on number of voxels
    if M > max_voxels:
        feats = feats[:max_voxels]
        coords = coords[:max_voxels]
    return feats, coords                       # (M, C), (M, 3) -> fed to SparseEncoder
