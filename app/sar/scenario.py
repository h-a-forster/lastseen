"""A search scenario: the map planners see (terrain, IPP, base, no-fly areas) plus the hidden truth, the subject's track.

`synthetic()` makes procedurally generated scenarios for fast benchmarking. Real scenarios (OpenStreetMap,
elevation and imagery for a chosen place) are built by `terrain.py` into the same structure.
"""

from dataclasses import dataclass

import numpy as np

from app.sar.grid import Grid, Terrain
from app.sar.routing import distance_field
from app.sar.walker import RECORD_S, WalkerParams, truth_track

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
    start_heading: float  # hidden truth: direction they set off in (radians; dy = sin, dx = cos)

    def subject_at(self, t_s: float) -> tuple[int, int] | None:
        """The subject's cell at search time `t_s`, or None while they are outside the mapped area."""
        y, x = self.track[min(int(t_s // RECORD_S), len(self.track) - 1)]
        grid = self.map.grid
        if not (0 <= y < grid.rows * grid.cell_m and 0 <= x < grid.cols * grid.cell_m):
            return None
        return grid.cell_of(y, x)


def synthetic(seed: int, size: int = 120, cell_m: float = 25.0, category: str = "hiker",
              truth: WalkerParams | None = None) -> Scenario:
    """A random wilderness area. The subject is moved by the truth walker (behaviour rules), not drawn from the
    statistics planners use, so no planner is tested on its own prior. `truth` replaces the walker's parameters while
    the map still says `category`: people who behave unlike the statistics planners were given (a stress test)."""
    rng = np.random.default_rng(seed)
    grid = Grid(size, size, cell_m)
    terrain = _terrain(rng, size)
    trails = _trails(rng, size, n=3)
    terrain[trails] = Terrain.OPEN
    ipp = (size // 2 + int(rng.integers(-size // 8, size // 8)), size // 2 + int(rng.integers(-size // 8, size // 8)))
    no_fly = _restricted_zone(rng, grid, ipp)
    search_map = SearchMap(grid, terrain, trails, no_fly, ipp, ipp, category)
    return Scenario(f"synthetic-{seed}", search_map, *truth_track(search_map, category, rng, TRACK_HOURS, truth))


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


def _restricted_zone(rng: np.random.Generator, grid: Grid, ipp: tuple[int, int]) -> np.ndarray:
    """A round no-fly zone of 200-400 m radius (say, a heliport or a nesting site), clear of the IPP and base, and
    never cutting off part of the area that drones then couldn't reach."""
    radius = rng.uniform(200, 400) / grid.cell_m
    r, c = np.indices(grid.shape)
    while True:
        centre = rng.uniform(0, grid.rows), rng.uniform(0, grid.cols)
        if np.hypot(centre[0] - ipp[0], centre[1] - ipp[1]) <= radius + 4:
            continue
        no_fly = np.hypot(r + 0.5 - centre[0], c + 0.5 - centre[1]) <= radius
        if np.isfinite(distance_field(grid, no_fly, ipp)[~no_fly]).all():
            return no_fly
