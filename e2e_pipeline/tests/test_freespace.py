"""Free-space extraction (item 1).

The assertions here encode the three decisions that separate a useful free-space
raster from a naive "is this column occupied" collapse: the height band, the
semantic split between ground and obstacle, and unknown-is-not-free.
"""

import numpy as np
import pytest

from e2e_pipeline.freespace import (FREE_CLASS, FreeSpaceExtractor, GridConfig)

ROAD, SIDEWALK, CAR, MANMADE = 11, 13, 4, 15


def _empty(grid: GridConfig) -> np.ndarray:
    """A world that is entirely free space."""
    return np.full(grid.shape, FREE_CLASS, dtype=np.int64)


def _road_world(grid: GridConfig) -> np.ndarray:
    """Free everywhere, with drivable surface laid down on the ground plane."""
    sem = _empty(grid)
    k_ground = grid.z_band_indices(-0.4, 0.2)[0]
    sem[:, :, k_ground] = ROAD
    return sem


@pytest.fixture
def grid() -> GridConfig:
    # A 20 x 20 m patch at the real 0.4 m resolution keeps the tests fast while
    # preserving the z layout the height-band logic depends on.
    return GridConfig(x=(-10.0, 10.0, 0.4), y=(-10.0, 10.0, 0.4), z=(-1.0, 5.4, 0.4))


def test_grid_shape_matches_flashocc_layout():
    """The default config must reproduce the checkpoint's (200, 200, 16) volume."""
    assert GridConfig().shape == (200, 200, 16)


def test_road_is_traversable_not_obstacle(grid):
    """Drivable surface must not wall the vehicle in.

    This is the failure mode of a pure geometric occupancy collapse: the road is an
    occupied voxel, so `any(axis=2)` marks every drivable cell an obstacle.
    """
    fs = FreeSpaceExtractor(grid)(_road_world(grid))
    assert fs.traversable.all()
    assert not fs.obstacle.any()
    assert fs.free_fraction == pytest.approx(1.0)


def test_gantry_above_roof_is_not_an_obstacle(grid):
    """Occupied voxels above the ego body must not block the plan."""
    sem = _road_world(grid)
    k_high = grid.z_band_indices(3.0, 3.4)[0]      # ~4 m up: a gantry or bridge deck
    sem[:, :, k_high] = MANMADE

    fs = FreeSpaceExtractor(grid)(sem)
    assert not fs.obstacle.any(), "structure above the roof line blocked the plan"
    assert fs.traversable.all()


def test_car_in_body_band_is_an_obstacle(grid):
    """A vehicle inside the swept slab blocks, and clears the traversable mask."""
    sem = _road_world(grid)
    k_lo, k_hi = grid.z_band_indices(0.2, 2.2)
    sem[10:15, 10:15, k_lo:k_hi] = CAR

    fs = FreeSpaceExtractor(grid)(sem)
    assert fs.obstacle[10:15, 10:15].all()
    assert not fs.traversable[10:15, 10:15].any()
    # Cells away from the car stay drivable.
    assert fs.traversable[0, 0]


def test_sidewalk_is_neither_drivable_nor_obstacle(grid):
    """Curbs must not read as walls, but must not read as road either.

    Geometry alone cannot make this distinction — both road and sidewalk are "stuff
    near the ground".  The Occ3D class labels are what make it possible.
    """
    sem = _road_world(grid)
    k_ground = grid.z_band_indices(-0.4, 0.2)[0]
    sem[:, 40:, k_ground] = SIDEWALK

    fs = FreeSpaceExtractor(grid)(sem)
    assert not fs.traversable[:, 40:].any(), "sidewalk marked drivable"
    assert not fs.obstacle[:, 40:].any(), "sidewalk marked as a wall"
    assert fs.traversable[:, :40].all()


def test_unknown_space_is_not_drivable(grid):
    """Unobserved voxels must not be reported as free.

    A grid that says "free" where no camera looked invites the planner straight into
    occlusion shadows, which is the exact failure occupancy was meant to fix.
    """
    sem = _road_world(grid)
    mask = np.ones(grid.shape, dtype=bool)
    mask[:, 30:, :] = False                        # right half never observed

    fs = FreeSpaceExtractor(grid)(sem, mask_camera=mask)
    assert fs.unknown[:, 30:].all()
    assert not fs.traversable[:, 30:].any()
    assert fs.traversable[:, :30].all()


def test_esdf_reports_metric_distance(grid):
    """The distance field must be in metres, not cells."""
    sem = _road_world(grid)
    k_lo, k_hi = grid.z_band_indices(0.2, 2.2)
    sem[25, 25, k_lo:k_hi] = CAR                   # single obstacle cell

    fs = FreeSpaceExtractor(grid)(sem)
    assert fs.esdf[25, 25] == pytest.approx(0.0, abs=1e-6)
    # Five cells away at 0.4 m resolution is 2.0 m.
    assert fs.esdf[30, 25] == pytest.approx(2.0, abs=0.05)
    assert fs.esdf[25, 30] == pytest.approx(2.0, abs=0.05)


def test_world_to_cell_roundtrip(grid):
    """World <-> cell must be consistent, or every clearance query is off by a shift."""
    fs = FreeSpaceExtractor(grid)(_road_world(grid))
    # Cell centre for index (i, j) is origin + (i + 0.5) * res.
    for ij in [(0, 0), (12, 37), (49, 49)]:
        xy = np.array([grid.x[0] + (ij[0] + 0.5) * grid.x[2],
                       grid.y[0] + (ij[1] + 0.5) * grid.y[2]])
        assert tuple(fs.world_to_cell(xy)) == ij


def test_off_grid_queries_are_conservative(grid):
    """Points outside the map get zero clearance and count as unknown.

    Treating off-grid as free is how a planner learns to escape the map.
    """
    fs = FreeSpaceExtractor(grid)(_road_world(grid))
    far = np.array([[500.0, 500.0]])
    assert fs.clearance_at(far)[0] == 0.0
    assert fs.unknown_at(far)[0]
    assert not fs.traversable_at(far)[0]


def test_inflation_shrinks_clearance(grid):
    """Inflating obstacles must reduce the measured clearance, not raise it."""
    sem = _road_world(grid)
    k_lo, k_hi = grid.z_band_indices(0.2, 2.2)
    sem[25, 25, k_lo:k_hi] = CAR

    plain = FreeSpaceExtractor(grid, inflate_m=0.0)(sem)
    fat = FreeSpaceExtractor(grid, inflate_m=0.8)(sem)
    assert fat.esdf[30, 25] < plain.esdf[30, 25]


def test_rejects_mismatched_volume(grid):
    """Shape mismatches must fail loudly — a silently transposed grid is unfixable."""
    with pytest.raises(ValueError, match="does not match grid"):
        FreeSpaceExtractor(grid)(np.full((10, 10, 16), FREE_CLASS))


def test_empty_height_band_is_rejected():
    """An inverted or empty band is a config error, not something to paper over."""
    with pytest.raises(ValueError, match="empty"):
        FreeSpaceExtractor(GridConfig(), height_band_m=(2.0, 2.0))
