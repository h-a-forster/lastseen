"""A search scenario: the map planners see (terrain, IPP, base, no-fly areas) plus the hidden truth.

`synthetic()` makes procedurally generated scenarios for fast benchmarking. Real scenarios (OpenStreetMap,
elevation and imagery for a chosen place) are built by `terrain.py` into the same structure.
"""

from dataclasses import dataclass

import numpy as np

from app.sar.grid import Grid, Terrain
from app.sar.prior import RINGS_KM

# Relative ease of walking through each terrain class (0 = impassable). Used by the motion model and the truth walker.
PASSABILITY = {Terrain.OPEN: 0.8, Terrain.SCRUB: 0.5, Terrain.FOREST: 0.4, Terrain.WATER: 0.05, Terrain.BUILT: 0.6}
TRAIL_PASSABILITY = 1.0


@dataclass
class SearchMap:
    """Everything a planner may see. The truth lives on `Scenario`, so planners (including generated ones) can't peek."""

    grid: Grid
    terrain: np.ndarray  # (rows, cols) Terrain values
    trails: np.ndarray  # (rows, cols) bool
    no_fly: np.ndarray  # (rows, cols) bool
    ipp: tuple[int, int]
    base: tuple[int, int]
    category: str

    @property
    def passability(self) -> np.ndarray:
        table = np.array([PASSABILITY[t] for t in Terrain])
        return np.where(self.trails, TRAIL_PASSABILITY, table[self.terrain])

    @property
    def rings_km(self) -> tuple[float, float, float, float]:
        return RINGS_KM[self.category]


@dataclass
class Scenario:
    name: str
    map: SearchMap
    subject_cell: tuple[int, int] | None  # hidden truth at the start; None = outside the area


def synthetic(seed: int, size: int = 120, cell_m: float = 25.0, category: str = "hiker", truth_scale_sd: float = 0.3) -> Scenario:
    """A random wilderness area. The subject is placed by the same ring statistics as the prior, but with each
    distance scaled by a log-normal factor (`truth_scale_sd`) so planners aren't tested on a prior that is exactly right."""
    rng = np.random.default_rng(seed)
    grid = Grid(size, size, cell_m)
    terrain = _terrain(rng, size)
    trails = _trails(rng, size, n=3)
    terrain[trails] = Terrain.OPEN
    no_fly = _power_line(rng, size, width=2)
    ipp = (size // 2 + int(rng.integers(-size // 8, size // 8)), size // 2 + int(rng.integers(-size // 8, size // 8)))
    subject = _place_subject(rng, grid, terrain, ipp, RINGS_KM[category], truth_scale_sd)
    return Scenario(f"synthetic-{seed}", SearchMap(grid, terrain, trails, no_fly, ipp, ipp, category), subject)


def _terrain(rng: np.random.Generator, size: int) -> np.ndarray:
    noise = _smooth_noise(rng, size, blobs=12)
    water = _smooth_noise(rng, size, blobs=4) > 0.82
    terrain = np.select([noise < 0.35, noise < 0.55], [Terrain.OPEN, Terrain.SCRUB], Terrain.FOREST)
    terrain[water] = Terrain.WATER
    return terrain.astype(np.int8)


def _smooth_noise(rng: np.random.Generator, size: int, blobs: int) -> np.ndarray:
    """Sum of random Gaussian blobs, scaled to 0..1: cheap terrain-like patches without extra dependencies."""
    r, c = np.indices((size, size))
    field = np.zeros((size, size))
    for _ in range(blobs):
        cy, cx = rng.uniform(0, size, 2)
        sigma = rng.uniform(size / 12, size / 4)
        field += np.exp(-((r - cy) ** 2 + (c - cx) ** 2) / (2 * sigma**2))
    return (field - field.min()) / (np.ptp(field) or 1)


def _trails(rng: np.random.Generator, size: int, n: int) -> np.ndarray:
    trails = np.zeros((size, size), bool)
    for _ in range(n):
        y, x = rng.uniform(0, size, 2)
        heading = rng.uniform(0, 2 * np.pi)
        for _ in range(size * 2):
            heading += rng.normal(0, 0.15)
            y, x = y + np.sin(heading), x + np.cos(heading)
            if not (0 <= y < size and 0 <= x < size):
                break
            trails[int(y), int(x)] = True
    return trails


def _power_line(rng: np.random.Generator, size: int, width: int) -> np.ndarray:
    no_fly = np.zeros((size, size), bool)
    col = int(rng.integers(size // 5, 4 * size // 5))
    no_fly[:, col : col + width] = True
    return no_fly


def _place_subject(rng, grid: Grid, terrain: np.ndarray, ipp: tuple[int, int], rings_km, scale_sd: float) -> tuple[int, int] | None:
    """Sample a distance from the ring percentiles (piecewise uniform), a random bearing, then step to dry land."""
    edges = (0.0, *rings_km, 2 * rings_km[-1])
    band = rng.choice(5, p=(0.25, 0.25, 0.25, 0.20, 0.05))
    dist_km = rng.uniform(edges[band], edges[band + 1]) * rng.lognormal(0, scale_sd)
    bearing = rng.uniform(0, 2 * np.pi)
    cells = dist_km * 1000 / grid.cell_m
    row, col = int(ipp[0] + cells * np.sin(bearing)), int(ipp[1] + cells * np.cos(bearing))
    if not (0 <= row < grid.rows and 0 <= col < grid.cols):
        return None  # outside the search area, as happens in real searches
    while terrain[row, col] == Terrain.WATER:
        row, col = int(np.clip(row + rng.integers(-1, 2), 0, grid.rows - 1)), int(np.clip(col + rng.integers(-1, 2), 0, grid.cols - 1))
    return row, col
