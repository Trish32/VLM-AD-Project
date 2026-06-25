"""
Pure-PyTorch hard voxelization, replacing the CUDA `Voxelization` op.

Mirrors mmdet3d/mmcv `hard_voxelize` semantics:
  - points are scanned in order; each point maps to a voxel grid cell
  - a new voxel is created the first time a cell is hit (up to `max_voxels`)
  - up to `max_num_points` points are kept per voxel (first-come-first-served)
  - points outside `point_cloud_range` are dropped

Returns (voxels, num_points_per_voxel, coords) where coords are (z, y, x)
to match mmdet3d's coordinate ordering used by PointPillarsScatter.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def hard_voxelize(points: torch.Tensor,
                  voxel_size,
                  point_cloud_range,
                  max_num_points: int,
                  max_voxels: int):
    """
    Args:
        points: (N, C) float tensor. points[:, :3] are xyz.
        voxel_size: (3,) x/y/z voxel size.
        point_cloud_range: (6,) [x_min, y_min, z_min, x_max, y_max, z_max].
        max_num_points: max points kept per voxel.
        max_voxels: max number of voxels created.

    Returns:
        voxels: (M, max_num_points, C) float tensor (zero-padded).
        num_points: (M,) int tensor — valid points per voxel.
        coords: (M, 3) int tensor — voxel grid coords in (z, y, x) order.
    """
    device = points.device
    dtype = points.dtype
    vs = torch.as_tensor(voxel_size, device=device, dtype=dtype)
    pcr = torch.as_tensor(point_cloud_range, device=device, dtype=dtype)

    grid_size = ((pcr[3:] - pcr[:3]) / vs).round().long()  # (nx, ny, nz)
    nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])

    # integer grid coords per point (x, y, z)
    coords_xyz = ((points[:, :3] - pcr[:3]) / vs).floor().long()

    # mask out-of-range points
    in_range = (
        (coords_xyz[:, 0] >= 0) & (coords_xyz[:, 0] < nx) &
        (coords_xyz[:, 1] >= 0) & (coords_xyz[:, 1] < ny) &
        (coords_xyz[:, 2] >= 0) & (coords_xyz[:, 2] < nz)
    )
    points = points[in_range]
    coords_xyz = coords_xyz[in_range]
    N = points.shape[0]
    if N == 0:
        C = points.shape[1]
        return (points.new_zeros((0, max_num_points, C)),
                points.new_zeros((0,), dtype=torch.long),
                points.new_zeros((0, 3), dtype=torch.long))

    # unique voxel id, preserving order of first appearance
    flat = (coords_xyz[:, 2] * (ny * nx) +
            coords_xyz[:, 1] * nx +
            coords_xyz[:, 0])
    # torch.unique sorts; we want first-appearance order to mirror the CUDA op.
    uniq, inverse = torch.unique(flat, return_inverse=True)
    # first occurrence index for each unique voxel
    first_idx = torch.full((uniq.numel(),), N, device=device, dtype=torch.long)
    arange = torch.arange(N, device=device)
    first_idx = first_idx.scatter_reduce(0, inverse, arange, reduce='amin',
                                         include_self=True)
    order = torch.argsort(first_idx)                  # voxel order by 1st point
    # remap: old unique idx -> new voxel index (first-appearance order)
    remap = torch.empty_like(order)
    remap[order] = torch.arange(order.numel(), device=device)
    voxel_of_point = remap[inverse]                   # (N,) voxel idx per point

    M = uniq.numel()
    if M > max_voxels:
        M = max_voxels

    C = points.shape[1]
    voxels = points.new_zeros((M, max_num_points, C))
    num_points = points.new_zeros((M,), dtype=torch.long)
    coords = points.new_zeros((M, 3), dtype=torch.long)  # (z, y, x)

    # assign points to voxels in scan order, respecting per-voxel capacity
    # vectorized "rank within voxel" via stable cumulative count
    sort_v, sort_p = torch.sort(voxel_of_point, stable=True)
    # rank of each point inside its voxel
    counts = torch.bincount(sort_v, minlength=uniq.numel())
    # build per-point intra-voxel rank
    rank = torch.arange(N, device=device) - (
        torch.cumsum(counts, 0) - counts)[sort_v]

    keep = (sort_v < M) & (rank < max_num_points)
    vsel = sort_v[keep]
    psel = sort_p[keep]
    rsel = rank[keep]
    voxels[vsel, rsel] = points[psel]

    # num points per voxel (capped)
    np_full = counts[:M].clamp(max=max_num_points)
    num_points[:M] = np_full

    # voxel coords (z, y, x) — take from the first point of each voxel
    vox_coords_xyz = coords_xyz[first_idx[order[:M]]]
    coords[:, 0] = vox_coords_xyz[:, 2]   # z
    coords[:, 1] = vox_coords_xyz[:, 1]   # y
    coords[:, 2] = vox_coords_xyz[:, 0]   # x

    return voxels, num_points, coords
