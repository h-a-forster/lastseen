"""Search-area raster, terrain classes and nadir camera geometry."""

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Terrain(IntEnum):
    OPEN = 0
    SCRUB = 1
    FOREST = 2
    WATER = 3
    BUILT = 4


@dataclass(frozen=True)
class Camera:
    """A nadir (straight-down) camera. Defaults approximate a 4/3-inch, 84° diagonal drone camera."""

    hfov_deg: float = 72.0
    vfov_deg: float = 57.0
    width_px: int = 5280

    def footprint_m(self, alt_m: float) -> tuple[float, float]:
        """Ground (across-track, along-track) size of one photo in metres."""
        return (
            2 * alt_m * math.tan(math.radians(self.hfov_deg) / 2),
            2 * alt_m * math.tan(math.radians(self.vfov_deg) / 2),
        )

    def gsd_cm(self, alt_m: float) -> float:
        """Ground sample distance: centimetres of ground per pixel."""
        return 100 * self.footprint_m(alt_m)[0] / self.width_px

    def tiles_per_photo(self, tile_px: int = 1000) -> int:
        """How many detector tiles one full-resolution photo is cut into (the detector sees ~1000 px squares)."""
        height_px = self.width_px * math.tan(math.radians(self.vfov_deg) / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        return math.ceil(self.width_px / tile_px) * math.ceil(height_px / tile_px)


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int
    cell_m: float = 25.0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def area_km2(self) -> float:
        return self.rows * self.cols * self.cell_m**2 / 1e6

    def cell_of(self, y_m: float, x_m: float) -> tuple[int, int]:
        return (
            min(max(int(y_m // self.cell_m), 0), self.rows - 1),
            min(max(int(x_m // self.cell_m), 0), self.cols - 1),
        )

    def centre_m(self, row: int, col: int) -> tuple[float, float]:
        return ((row + 0.5) * self.cell_m, (col + 0.5) * self.cell_m)

    def footprint(self, y_m: float, x_m: float, width_m: float, height_m: float) -> tuple[slice, slice]:
        """Cells whose centres fall inside an axis-aligned photo footprint centred on (y, x)."""
        rows = self._span(y_m, height_m, self.rows)
        cols = self._span(x_m, width_m, self.cols)
        return rows, cols

    def _span(self, centre_m: float, size_m: float, n: int) -> slice:
        lo = math.ceil((centre_m - size_m / 2) / self.cell_m - 0.5)
        hi = math.floor((centre_m + size_m / 2) / self.cell_m - 0.5)
        return slice(max(lo, 0), max(min(hi + 1, n), 0))

    def distance_km(self, row: int, col: int) -> np.ndarray:
        """Distance of every cell centre from cell (row, col), in km."""
        r, c = np.indices(self.shape)
        return np.hypot(r - row, c - col) * self.cell_m / 1000
