"""A search scenario: the map planners see (terrain, IPP, base, no-fly areas) plus the hidden truth, the subject's track.

`synthetic()` makes procedurally generated scenarios for fast benchmarking. Real scenarios (OpenStreetMap,
elevation and imagery for a chosen place) are built by `terrain.py` into the same structure.
"""

from dataclasses import dataclass

import numpy as np

from app.sar.grid import Grid, Terrain
from app.sar.walker import RECORD_S, truth_track

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


TRACK_HOURS = 8.0  # truth recorded this long after the search starts; the subject stays put after that


@dataclass
class Scenario:
    name: str
    map: SearchMap
    track: np.ndarray  # hidden truth: (records, 2) subject position in metres, every RECORD_S from the search start

    def subject_at(self, t_s: float) -> tuple[int, int] | None:
        """The subject's cell at search time `t_s`, or None while they are outside the mapped area."""
        y, x = self.track[min(int(t_s // RECORD_S), len(self.track) - 1)]
        grid = self.map.grid
        if not (0 <= y < grid.rows * grid.cell_m and 0 <= x < grid.cols * grid.cell_m):
            return None
        return grid.cell_of(y, x)


def synthetic(seed: int, size: int = 120, cell_m: float = 25.0, category: str = "hiker") -> Scenario:
    """A random wilderness area. The subject is moved by the truth walker (behaviour rules), not drawn from the
    statistics planners use, so no planner is tested on its own prior."""
    rng = np.random.default_rng(seed)
    grid = Grid(size, size, cell_m)
    terrain = _terrain(rng, size)
    trails = _trails(rng, size, n=3)
    terrain[trails] = Terrain.OPEN
    no_fly = _power_line(rng, size, width=2)
    ipp = (size // 2 + int(rng.integers(-size // 8, size // 8)), size // 2 + int(rng.integers(-size // 8, size // 8)))
    search_map = SearchMap(grid, terrain, trails, no_fly, ipp, ipp, category)
    return Scenario(f"synthetic-{seed}", search_map, truth_track(search_map, category, rng, TRACK_HOURS))


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
