"""RayIoU evaluation for occupancy, pure-PyTorch / MPS.

Reimplements ``projects/mmdet3d_plugin/core/evaluation/ray_metrics.py`` of
FlashOCC (itself from SparseOcc / 4d-occ-forecasting) WITHOUT the CUDA ``dvr``
renderer. RayIoU scores the first occupied surface each simulated LiDAR ray
hits, instead of every voxel (mIoU) -- it is immune to surface "thickening"
and needs no visibility mask (occlusion is intrinsic to ray casting).

Algorithm per sample:
  1. generate_lidar_rays(): ~15840 fixed ray directions (a LiDAR sweep).
  2. For each of up to 8 ego origins (from the devkit, see loader), cast every
     ray through the predicted AND the GT voxel volume; record (class, depth)
     of the first occupied voxel hit.
  3. Keep rays whose GT hit is non-free. A ray is TP for class c if
     pred_class == gt_class == c and |depth_pred - depth_gt| < threshold.
  4. RayIoU_c = TP / (GT + pred - TP), averaged over thresholds {1,2,4} m and
     the 17 non-free classes.

The only deviation from the CUDA version is the renderer: we march each ray in
fixed metric steps and take the first occupied sample (a dense-sampling
approximation of the exact voxel-DDA). With a half-voxel step this matches the
DDA on contiguous surfaces to well within the 1 m threshold.
"""
import math
import numpy as np
import torch

OCC_CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free']
FREE_ID = 17
PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
VOXEL_SIZE = 0.4
THRESHOLDS = [1, 2, 4]


def generate_lidar_rays():
    """Fixed set of ray directions mimicking a nuScenes LiDAR sweep.

    Identical to the official: a set of pitch rings (denser near horizon) x
    360 azimuth steps. Returns (R, 3) unit-ish direction vectors.
    """
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)
    # extend up to the nuScenes LiDAR top FoV (~0.21 rad) with constant delta
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    rays = []
    for pitch in pitch_angles:
        for az_deg in np.arange(0, 360, 1):
            az = np.deg2rad(az_deg)
            rays.append((np.cos(pitch) * np.cos(az),
                         np.cos(pitch) * np.sin(az),
                         np.sin(pitch)))
    return np.array(rays, dtype=np.float32)        # (R, 3)


@torch.no_grad()
def raycast(sem, origins, rays, device, max_range=100.0, step=0.2,
            ray_chunk=4000):
    """Cast rays through a voxel volume, return first-hit (label, distance).

    Args:
        sem: (200, 200, 16) long tensor, voxel class ids indexed [x, y, z].
        origins: (T, 3) numpy, ray start points in the ego frame (metres).
        rays: (R, 3) tensor of unit ray directions.
        max_range/step: metric marching range and stride (step = half a voxel
            by default; finer = more faithful but slower).
        ray_chunk: rays processed at once (caps peak memory on MPS).
    Returns:
        labels: (T*R,) int  -- class of the first occupied voxel (FREE_ID if
            the ray leaves the grid without hitting anything).
        dists:  (T*R,) float -- distance along the ray to that voxel (metres).
    """
    sem = sem.to(device)
    rays = rays.to(device)
    nx, ny, nz = sem.shape
    lower = torch.tensor(PC_RANGE[:3], device=device)
    t = torch.arange(step, max_range + 1e-6, step, device=device)   # (S,)
    S = t.shape[0]

    all_lab, all_dist = [], []
    for o in origins:
        o = torch.tensor(o, dtype=torch.float32, device=device)
        lab_chunks, dist_chunks = [], []
        for r0 in range(0, rays.shape[0], ray_chunk):
            rc = rays[r0:r0 + ray_chunk]                # (r, 3)
            r = rc.shape[0]
            # sample points along each ray: origin + dir * t
            pts = o.view(1, 1, 3) + rc.view(r, 1, 3) * t.view(1, S, 1)  # (r,S,3)
            vox = ((pts - lower) / VOXEL_SIZE).floor().long()          # (r,S,3)
            inside = ((vox[..., 0] >= 0) & (vox[..., 0] < nx) &
                      (vox[..., 1] >= 0) & (vox[..., 1] < ny) &
                      (vox[..., 2] >= 0) & (vox[..., 2] < nz))
            vx = vox[..., 0].clamp(0, nx - 1)
            vy = vox[..., 1].clamp(0, ny - 1)
            vz = vox[..., 2].clamp(0, nz - 1)
            lab = sem[vx, vy, vz]                       # (r, S)
            lab = torch.where(inside, lab, torch.full_like(lab, FREE_ID))
            occ = lab != FREE_ID                        # occupied samples
            has = occ.any(1)                            # did the ray hit?
            first = occ.float().argmax(1)               # first occupied step
            idx = torch.arange(r, device=device)
            hit_lab = torch.where(has, lab[idx, first],
                                  torch.full_like(first, FREE_ID))
            hit_dist = torch.where(has, t[first],
                                   torch.full((r,), max_range, device=device))
            lab_chunks.append(hit_lab)
            dist_chunks.append(hit_dist)
        all_lab.append(torch.cat(lab_chunks))
        all_dist.append(torch.cat(dist_chunks))
    return (torch.cat(all_lab).cpu().numpy(),
            torch.cat(all_dist).cpu().numpy())


class RayIoUAccumulator:
    """Accumulates ray TP/GT/pred counts across samples, then reports RayIoU."""

    def __init__(self):
        n = len(OCC_CLASS_NAMES)
        self.gt_cnt = np.zeros(n)
        self.pred_cnt = np.zeros(n)
        self.tp_cnt = np.zeros((len(THRESHOLDS), n))
        self.samples = 0

    def add(self, pred_label, pred_dist, gt_label, gt_dist):
        """One sample's per-ray results (already flattened over origins)."""
        # evaluate only on rays whose GT actually hit a surface
        valid = gt_label != FREE_ID
        pl, pd = pred_label[valid], pred_dist[valid]
        gl, gd = gt_label[valid], gt_dist[valid]
        self.samples += 1
        l1 = np.abs(pd - gd)
        for i in range(len(OCC_CLASS_NAMES)):
            self.gt_cnt[i] += (gl == i).sum()
            self.pred_cnt[i] += (pl == i).sum()
            for j, thr in enumerate(THRESHOLDS):
                self.tp_cnt[j][i] += ((gl == i) & (pl == i) & (l1 < thr)).sum()

    def report(self):
        ious = []
        for j in range(len(THRESHOLDS)):
            denom = self.gt_cnt + self.pred_cnt - self.tp_cnt[j]
            ious.append((self.tp_cnt[j] / denom)[:-1])   # drop 'free'
        ious = np.array(ious)                             # (3, 17)

        print(f'\n===> RayIoU of {self.samples} samples:')
        header = f'{"class":22s} {"@1":>7s} {"@2":>7s} {"@4":>7s}'
        print(header)
        for i in range(len(OCC_CLASS_NAMES) - 1):
            print(f'{OCC_CLASS_NAMES[i]:22s} '
                  f'{ious[0][i]*100:7.2f} {ious[1][i]*100:7.2f} '
                  f'{ious[2][i]*100:7.2f}')
        m1, m2, m4 = [round(float(np.nanmean(ious[j])) * 100, 2)
                      for j in range(3)]
        rayiou = round(float(np.nanmean(ious)) * 100, 2)
        print(f'{"MEAN":22s} {m1:7.2f} {m2:7.2f} {m4:7.2f}')
        print(f'===> RayIoU (mean over @1/@2/@4): {rayiou}')
        return {'RayIoU': rayiou, 'RayIoU@1': m1, 'RayIoU@2': m2,
                'RayIoU@4': m4}
