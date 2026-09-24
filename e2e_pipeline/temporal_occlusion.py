"""Unknown space as "not observed lately", not as "geometrically shadowed".

WHY THE GEOMETRIC DEFINITION FAILED. `ray_occlusion` marks every cell behind an
obstacle on the ray from the ego. That is a correct statement about one frame's
line of sight and a wrong definition of ignorance, for a reason that showed up
as a hard measurement: unknown is then *by construction* on the far side of an
obstacle, so a trajectory can only reach it by passing through the thing casting
the shadow. Measured over 6 scenes, 0 of 572 occlusion-entering candidates were
feasible and 98.8% died on the clearance gate at a median 0.00 m -- they were
not being rejected for entering the unknown, they were being rejected for
driving into a parked car. A prior on that region prices somewhere the ego
cannot go, and it reported exactly zero at every value.

It also never shrinks. The shadow is recomputed from scratch each frame, so
driving past a van and seeing what was behind it changes nothing: the mask is a
function of the current geometry alone and carries no memory. 84.0% of the map
was unknown at every step of every scene.

THE TEMPORAL DEFINITION. A cell is unknown if no sensor has observed it within
the last N frames. Observation accumulates in the WORLD frame, so as the ego
moves, cells that were shadowed become visible and flip to free or occupied, and
the unknown region shrinks toward the genuinely unvisited -- side streets, space
beyond sensor range, the area behind the ego. That region is reachable, which is
the property the geometric one lacked, and it is what a prior should price.

N is a real parameter, not a smoothing constant. It is the assumption "the world
does not change faster than N frames", which is false for moving agents and
roughly true for parked cars and walls. Small N distrusts memory and approaches
the single-frame mask; large N trusts stale observations. The default of 10
frames is 5 s at 2 Hz keyframes.

WHAT IT DOES NOT DO. Memory of free space is not a guarantee of free space --
an agent can walk into a cell observed empty 4 s ago. This tracks OBSERVATION,
and the object branch tracks agents; conflating them would let a stale free
reading suppress a live detection. The two are combined downstream, not here.
"""
from __future__ import annotations

import numpy as np


def visible_mask(obstacle: np.ndarray, origin, res: float,
                 n_rays: int = 720, max_range_m: float | None = None
                 ) -> np.ndarray:
    """Cells the sensor can see from the ego this frame.

    The complement of `ray_occlusion` plus a range limit, sharing its marching so
    the two definitions cannot drift apart. The obstacle cell that terminates a
    ray IS visible -- you can see the van, you cannot see behind it.
    """
    nx, ny = obstacle.shape
    vis = np.zeros((nx, ny), dtype=bool)
    ex = int((0.0 - origin[0]) / res)
    ey = int((0.0 - origin[1]) / res)
    max_r = float(np.hypot(nx, ny)) if max_range_m is None else max_range_m / res
    for ang in np.linspace(-np.pi, np.pi, n_rays, endpoint=False):
        dx, dy = np.cos(ang), np.sin(ang)
        for r in np.arange(0.0, max_r, 0.5):
            ix, iy = int(ex + dx * r), int(ey + dy * r)
            if not (0 <= ix < nx and 0 <= iy < ny):
                break
            vis[ix, iy] = True
            if obstacle[ix, iy]:
                break                      # the obstacle is seen; past it is not
    return vis


class TemporalOcclusionMemory:
    """World-frame record of when each cell was last observed.

    Deliberately world-frame. An ego-frame accumulator would have to resample
    every step, and resampling an observation record is exactly the operation
    that invents observations -- interpolating between "seen" and "unseen" cells
    produces a mask nobody measured. Integer cell indices in a fixed world grid
    keep every update a write to the cell that was actually observed.
    """

    NEVER = -(10 ** 9)

    def __init__(self, bounds, res: float = 0.4, horizon_frames: int = 10,
                 margin_m: float = 60.0) -> None:
        (x0, x1), (y0, y1) = bounds
        self.res = float(res)
        self.horizon = int(horizon_frames)
        self.origin = (float(x0 - margin_m), float(y0 - margin_m))
        nx = int(np.ceil((x1 - x0 + 2 * margin_m) / res))
        ny = int(np.ceil((y1 - y0 + 2 * margin_m) / res))
        self.last_seen = np.full((nx, ny), self.NEVER, dtype=np.int64)
        self.frame = 0

    # -- world <-> cell ----------------------------------------------------

    def _cells(self, xy: np.ndarray) -> np.ndarray:
        idx = (np.asarray(xy, float) - np.asarray(self.origin)) / self.res
        return np.floor(idx).astype(np.int64)

    def _in(self, c: np.ndarray) -> np.ndarray:
        nx, ny = self.last_seen.shape
        return ((c[..., 0] >= 0) & (c[..., 0] < nx) &
                (c[..., 1] >= 0) & (c[..., 1] < ny))

    # -- update ------------------------------------------------------------

    def observe(self, obstacle_ego: np.ndarray, ego_origin, ego_res: float,
                ego_xy, ego_yaw: float, frame: int | None = None,
                max_range_m: float | None = None) -> int:
        """Stamp every cell visible from this pose with the current frame."""
        if frame is not None:
            self.frame = int(frame)
        vis = visible_mask(obstacle_ego, ego_origin, ego_res,
                           max_range_m=max_range_m)
        ix, iy = np.nonzero(vis)
        if not len(ix):
            return 0
        # ego-frame metric centres of the visible cells
        px = ego_origin[0] + (ix + 0.5) * ego_res
        py = ego_origin[1] + (iy + 0.5) * ego_res
        c, s = np.cos(ego_yaw), np.sin(ego_yaw)
        wx = ego_xy[0] + c * px - s * py
        wy = ego_xy[1] + s * px + c * py
        cells = self._cells(np.stack([wx, wy], axis=-1))
        ok = self._in(cells)
        if ok.any():
            self.last_seen[cells[ok, 0], cells[ok, 1]] = self.frame
        return int(ok.sum())

    # -- query -------------------------------------------------------------

    def unknown_ego(self, shape, ego_origin, ego_res: float, ego_xy,
                    ego_yaw: float, frame: int | None = None) -> np.ndarray:
        """(nx, ny) bool: cells of the ego grid not observed within `horizon`."""
        f = self.frame if frame is None else int(frame)
        nx, ny = shape
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        px = ego_origin[0] + (ix + 0.5) * ego_res
        py = ego_origin[1] + (iy + 0.5) * ego_res
        c, s = np.cos(ego_yaw), np.sin(ego_yaw)
        wx = ego_xy[0] + c * px - s * py
        wy = ego_xy[1] + s * px + c * py
        cells = self._cells(np.stack([wx, wy], axis=-1))
        ok = self._in(cells)
        seen = np.full((nx, ny), self.NEVER, dtype=np.int64)
        seen[ok] = self.last_seen[cells[..., 0][ok], cells[..., 1][ok]]
        # off-grid is unknown: the memory has no claim there
        return (f - seen) > self.horizon

    @property
    def observed_fraction(self) -> float:
        return float((self.last_seen != self.NEVER).mean())
