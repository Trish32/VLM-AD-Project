"""
(1) FlashOcc occupancy  ->  drivable / free-space representation.

FlashOcc's head emits (200, 200, 16, 18) class logits — 640k voxels, 11.5M logits.
Handing that to a planner is the wrong interface twice over: it is far larger than the
decision needs, and it forces every downstream consumer to re-derive the same few
facts.  What planning actually wants from occupancy is two rasters and a distance
field:

    traversable : where the ego is *allowed* to be   (road surface)
    obstacle    : where the ego *cannot* be          (anything solid at ego height)
    esdf        : metres of clearance to the nearest obstacle

which is 3 x 200 x 200 floats — about 0.5% of the voxel volume — and is directly
consumable by a cost function.

THE HEIGHT BAND
---------------
Collapsing 3D to 2D is not "is this column occupied anywhere".  A gantry, an
overhanging branch, and a tunnel roof are all occupied voxels the ego drives straight
under; the road surface itself is an occupied voxel it drives straight over.  Only the
slab the vehicle body actually sweeps matters.

With the flashocc-r50 grid (`z: [-1, 5.4]`, 16 bins of 0.4 m), voxel k spans
`[-1 + 0.4k, -1 + 0.4(k+1)]`.  The nuScenes ego origin sits at ground level, so:

    ground  z ~ 0.0 m  -> k = 2.5
    roof    z ~ 2.0 m  -> k = 7.5

giving a default band of k in [3, 7] (z in [0.2, 2.2]) — above the road, below the
gantries.  `FreeSpaceExtractor` derives this from the grid config rather than
hard-coding it, and `height_band_m` lets you retune without recomputing indices.

SEMANTICS BEAT GEOMETRY
-----------------------
This is where Occ3D's 17 classes earn their keep over a raw binary occupancy grid.
Geometry alone cannot separate road from curb — both are "stuff near the ground".
The class labels can: `driveable_surface` is traversable, `sidewalk` and `terrain` are
not, and a `pedestrian` voxel is an obstacle regardless of its height.

UNKNOWN IS NOT FREE
-------------------
Occ3D ships `mask_camera` because voxels no camera observed carry no information.  A
grid that reports "free" for unobserved space invites the planner to drive into
occlusion shadows, which is the exact failure mode occupancy was supposed to fix.  We
carry a third state and let the safety filter decide how conservative to be.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:                                            # SciPy ships in the shared env
    from scipy.ndimage import distance_transform_edt as _edt
except ImportError:                             # pragma: no cover - fallback path
    _edt = None


# ---------------------------------------------------------------------------
# Occ3D-nuScenes taxonomy
# ---------------------------------------------------------------------------

# Index order is load-bearing: it matches the checkpoint's head and
# Occupancy/FlashOcc/tools/visualize_occ.py::OCC_CLASSES exactly.
OCC_CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free',
]
FREE_CLASS = 17

# Surfaces the ego may legally occupy.  `other_flat` is deliberately excluded by
# default — it covers traffic islands and median strips, which are flat but not
# drivable.  Callers who want a permissive road boundary can widen it.
TRAVERSABLE_CLASSES = frozenset({11})                      # driveable_surface

# Ground-ish classes that are not obstacles but not drivable either.  They must be
# excluded from the obstacle mask or the road edge becomes a wall at every curb.
GROUND_CLASSES = frozenset({11, 12, 13, 14})               # +other_flat/sidewalk/terrain


@dataclass(frozen=True)
class GridConfig:
    """Metric extent of the occupancy volume. Defaults match flashocc-r50."""

    x: tuple[float, float, float] = (-40.0, 40.0, 0.4)
    y: tuple[float, float, float] = (-40.0, 40.0, 0.4)
    z: tuple[float, float, float] = (-1.0, 5.4, 0.4)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(round((self.x[1] - self.x[0]) / self.x[2])),
                int(round((self.y[1] - self.y[0]) / self.y[2])),
                int(round((self.z[1] - self.z[0]) / self.z[2])))

    def z_band_indices(self, lo_m: float, hi_m: float) -> tuple[int, int]:
        """Voxel index range [k_lo, k_hi) covering the metric band [lo_m, hi_m)."""
        k_lo = int(np.floor((lo_m - self.z[0]) / self.z[2]))
        k_hi = int(np.ceil((hi_m - self.z[0]) / self.z[2]))
        nz = self.shape[2]
        return max(0, min(k_lo, nz)), max(0, min(k_hi, nz))


# ---------------------------------------------------------------------------
# Free-space raster
# ---------------------------------------------------------------------------


@dataclass
class FreeSpace:
    """Planner-facing view of the scene's geometry.

    All rasters are (nx, ny) and indexed [ix, iy], with ix along +x (forward) and iy
    along +y (left) of the ego frame.  `origin` is the metric coordinate of cell
    (0, 0)'s lower corner and `res` the cell size, so world <-> cell is affine.
    """

    traversable: np.ndarray            # (nx, ny) bool
    obstacle: np.ndarray               # (nx, ny) bool
    unknown: np.ndarray                # (nx, ny) bool
    esdf: np.ndarray                   # (nx, ny) float32, metres to nearest obstacle
    origin: tuple[float, float]
    res: float
    #: (nx, ny) int8 Occ3D class per BEV cell, or None when the source had no
    #: semantics (the synthetic corridor). The planner does not read this -- the
    #: three boolean rasters above are the whole planning interface, and adding
    #: a class id to them would let a consumer start special-casing 'pedestrian'
    #: in a filter that is supposed to be class-agnostic. It is carried for
    #: visualisation and diagnosis, where collapsing 18 classes to
    #: drivable/obstacle/unknown hides what the occupancy branch actually said.
    semantics: np.ndarray | None = None

    @property
    def free_fraction(self) -> float:
        return float(self.traversable.mean())

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        """(..., 2) metres -> (..., 2) int cell indices. May fall outside the grid."""
        xy = np.asarray(xy, dtype=np.float64)
        idx = (xy - np.asarray(self.origin)) / self.res
        return np.floor(idx).astype(np.int64)

    def in_bounds(self, cells: np.ndarray) -> np.ndarray:
        """(..., 2) cells -> (...,) bool."""
        nx, ny = self.traversable.shape
        return ((cells[..., 0] >= 0) & (cells[..., 0] < nx) &
                (cells[..., 1] >= 0) & (cells[..., 1] < ny))

    def clearance_at(self, xy: np.ndarray, outside: float = 0.0) -> np.ndarray:
        """Metres of clearance at world points (..., 2).

        Points off the grid return `outside`, defaulting to 0 — i.e. "no clearance
        guarantee".  Treating off-grid as infinitely free is how a planner learns to
        escape the map, so the conservative default is deliberate.
        """
        cells = self.world_to_cell(xy)
        ok = self.in_bounds(cells)
        out = np.full(cells.shape[:-1], float(outside), dtype=np.float64)
        if ok.any():
            cx = cells[..., 0][ok]
            cy = cells[..., 1][ok]
            out[ok] = self.esdf[cx, cy]
        return out

    def traversable_at(self, xy: np.ndarray, outside: bool = False) -> np.ndarray:
        """Whether world points (..., 2) lie on drivable surface."""
        cells = self.world_to_cell(xy)
        ok = self.in_bounds(cells)
        out = np.full(cells.shape[:-1], bool(outside), dtype=bool)
        if ok.any():
            out[ok] = self.traversable[cells[..., 0][ok], cells[..., 1][ok]]
        return out

    def unknown_at(self, xy: np.ndarray, outside: bool = True) -> np.ndarray:
        """Whether world points (..., 2) are unobserved. Off-grid counts as unknown."""
        cells = self.world_to_cell(xy)
        ok = self.in_bounds(cells)
        out = np.full(cells.shape[:-1], bool(outside), dtype=bool)
        if ok.any():
            out[ok] = self.unknown[cells[..., 0][ok], cells[..., 1][ok]]
        return out


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class FreeSpaceExtractor:
    """Reduce a semantic occupancy volume to a `FreeSpace` raster.

    Parameters
    ----------
    grid : metric layout of the incoming volume.
    height_band_m : (lo, hi) slab the ego body sweeps, metres above the ego origin.
        Defaults to (0.2, 2.2): clear of the road surface, below typical gantries.
    inflate_m : dilate obstacles by this radius before the distance transform.  Set it
        to the ego's half-width to turn the ESDF into a configuration-space clearance
        and let the planner treat itself as a point; leave at 0 to keep the ESDF
        purely geometric and do footprint sweeps explicitly (what `safety_filter`
        does, since a rectangle is poorly approximated by a disc at speed).
    traversable_classes / ground_classes : taxonomy overrides.
    """

    def __init__(self,
                 grid: GridConfig | None = None,
                 height_band_m: tuple[float, float] = (0.2, 2.2),
                 inflate_m: float = 0.0,
                 traversable_classes: frozenset[int] = TRAVERSABLE_CLASSES,
                 ground_classes: frozenset[int] = GROUND_CLASSES) -> None:
        self.grid = grid or GridConfig()
        self.height_band_m = height_band_m
        self.inflate_m = float(inflate_m)
        self.traversable_classes = traversable_classes
        self.ground_classes = ground_classes
        # Validate the *metric* band before converting to indices: floor/ceil would
        # silently widen a degenerate band like (2.0, 2.0) into one voxel, which is a
        # config error the caller should hear about rather than a default to apply.
        if height_band_m[1] <= height_band_m[0]:
            raise ValueError(
                f"height band {height_band_m} m is empty (hi must exceed lo)")
        self.k_lo, self.k_hi = self.grid.z_band_indices(*height_band_m)
        if self.k_hi <= self.k_lo:
            raise ValueError(
                f"height band {height_band_m} m is empty under z={self.grid.z}")

    # -- main entry point ---------------------------------------------------

    def __call__(self, semantics: np.ndarray,
                 mask_camera: np.ndarray | None = None) -> FreeSpace:
        """
        Parameters
        ----------
        semantics : (nx, ny, nz) int, Occ3D class index per voxel (17 == free).
            This is `argmax` over FlashOcc's 18 logits — see
            `Occupancy/FlashOcc/model/bevdet_occ.py::get_occ`.
        mask_camera : (nx, ny, nz) bool, True where a camera actually observed the
            voxel.  When omitted every voxel counts as observed, which is the right
            default for a model's own prediction (it predicts everywhere) but wrong
            for GT-derived grids.

        Returns
        -------
        FreeSpace with matching (nx, ny) rasters.
        """
        sem = np.asarray(semantics)
        if sem.ndim != 3:
            raise ValueError(f"expected (nx, ny, nz) semantics, got {sem.shape}")
        nx, ny, nz = sem.shape
        if (nx, ny, nz) != self.grid.shape:
            raise ValueError(
                f"semantics {sem.shape} does not match grid {self.grid.shape}")

        band = sem[:, :, self.k_lo:self.k_hi]                  # (nx, ny, kb)

        # --- obstacle: any solid, non-ground class inside the swept slab ----
        # `free` is not an obstacle; ground classes are excluded so curbs and road
        # surface do not wall the vehicle in at every lane edge.
        solid = (band != FREE_CLASS)
        for c in self.ground_classes:
            solid &= (band != c)
        obstacle = solid.any(axis=2)                           # (nx, ny)

        # --- traversable: drivable surface anywhere in the column -----------
        # Searched over the full height, not the band: the road is *below* the ego
        # body slab, so restricting to the band would find nothing.
        traversable = np.zeros((nx, ny), dtype=bool)
        for c in self.traversable_classes:
            traversable |= (sem == c).any(axis=2)
        # A cell blocked at body height is not drivable no matter what is underneath.
        traversable &= ~obstacle

        # --- unknown: never observed by any camera in the slab --------------
        if mask_camera is not None:
            observed = np.asarray(mask_camera, dtype=bool)[:, :, self.k_lo:self.k_hi]
            unknown = ~observed.any(axis=2)
        else:
            unknown = np.zeros((nx, ny), dtype=bool)
        # Unobserved space is not drivable — see module docstring.
        traversable &= ~unknown

        esdf = self._esdf(obstacle)

        return FreeSpace(
            traversable=traversable,
            obstacle=obstacle,
            unknown=unknown,
            esdf=esdf,
            origin=(self.grid.x[0], self.grid.y[0]),
            res=float(self.grid.x[2]),
            semantics=self.bev_semantics(sem),
        )

    @staticmethod
    def bev_semantics(sem: np.ndarray) -> np.ndarray:
        """(nx, ny, nz) class volume -> (nx, ny) class per cell, topmost non-free.

        Matches the official FlashOCC `vis_occ.py` reduction rather than
        inventing one: each column takes the class of its highest occupied
        voxel, which is the surface you would see looking straight down. On open
        road that is `driveable_surface`; where a car stands it is `car`.
        """
        occupied = sem != FREE_CLASS
        # highest occupied index per column, 0 where the column is empty
        top = (sem.shape[2] - 1
               - np.argmax(occupied[:, :, ::-1], axis=2))
        out = np.take_along_axis(sem, top[:, :, None], axis=2)[:, :, 0]
        return np.where(occupied.any(axis=2), out, FREE_CLASS).astype(np.int8)

    # -- distance field -----------------------------------------------------

    def _esdf(self, obstacle: np.ndarray) -> np.ndarray:
        """Euclidean distance (metres) from each cell to the nearest obstacle.

        A grid with no obstacles has no finite distance; we return +inf-free large
        values by capping at the grid diagonal so downstream arithmetic stays finite.
        """
        res = float(self.grid.x[2])
        if self.inflate_m > 0:
            obstacle = self._dilate(obstacle, int(round(self.inflate_m / res)))

        if not obstacle.any():
            diag = res * float(np.hypot(*obstacle.shape))
            return np.full(obstacle.shape, diag, dtype=np.float32)

        if _edt is not None:
            dist = _edt(~obstacle, sampling=(res, res))
        else:                                       # pragma: no cover
            dist = _chamfer_edt(~obstacle) * res
        return dist.astype(np.float32)

    @staticmethod
    def _dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
        """Square-structuring-element dilation, implemented by shifting.

        Kept dependency-free and explicit rather than pulling in a morphology op; at
        these radii (a few cells) the shift loop is negligible.
        """
        if radius_cells <= 0:
            return mask
        out = mask.copy()
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                out |= np.roll(np.roll(mask, dx, axis=0), dy, axis=1)
        return out


def _chamfer_edt(free: np.ndarray) -> np.ndarray:    # pragma: no cover
    """Two-pass chamfer distance in cells — SciPy-free fallback.

    Approximates Euclidean distance to ~2% with the (1, sqrt2) kernel, which is well
    inside the tolerance of a safety margin measured in tenths of a metre.
    """
    big = float(free.size)
    d = np.where(free, big, 0.0)
    s2 = np.sqrt(2.0)
    nx, ny = d.shape
    for i in range(nx):                              # forward pass
        for j in range(ny):
            if d[i, j] == 0.0:
                continue
            best = d[i, j]
            if i > 0:
                best = min(best, d[i - 1, j] + 1.0)
                if j > 0:
                    best = min(best, d[i - 1, j - 1] + s2)
                if j < ny - 1:
                    best = min(best, d[i - 1, j + 1] + s2)
            if j > 0:
                best = min(best, d[i, j - 1] + 1.0)
            d[i, j] = best
    for i in range(nx - 1, -1, -1):                  # backward pass
        for j in range(ny - 1, -1, -1):
            best = d[i, j]
            if i < nx - 1:
                best = min(best, d[i + 1, j] + 1.0)
                if j > 0:
                    best = min(best, d[i + 1, j - 1] + s2)
                if j < ny - 1:
                    best = min(best, d[i + 1, j + 1] + s2)
            if j < ny - 1:
                best = min(best, d[i, j + 1] + 1.0)
            d[i, j] = best
    return d
