"""Flight-pattern geometry shared by planners."""

import math

import numpy as np

from app.sar.grid import Grid

Point = tuple[float, float]


def lanes(grid: Grid, no_fly: np.ndarray, rows: slice, cols: slice, lane_m: float) -> list[tuple[Point, Point]]:
    """Back-and-forth lanes (north-south) covering a block of cells, split around no-fly cells. Each lane is (start, end)."""
    x0, x1 = cols.start * grid.cell_m, cols.stop * grid.cell_m
    out: list[tuple[Point, Point]] = []
    n = max(1, math.ceil((x1 - x0) / lane_m))
    for i in range(n):
        x = x0 + (i + 0.5) * (x1 - x0) / n
        col = min(int(x // grid.cell_m), grid.cols - 1)
        segments = _free_runs(~no_fly[rows, col], rows.start)
        if i % 2:
            segments = [(b, a) for a, b in reversed(segments)]
        for a, b in segments:
            out.append(((a * grid.cell_m, x), (b * grid.cell_m, x)))
    return out


def _free_runs(free: np.ndarray, offset: int) -> list[tuple[int, int]]:
    """Runs of True as (first_row, last_row + 1), offset to grid rows."""
    runs, start = [], None
    for i, ok in enumerate(free):
        if ok and start is None:
            start = i
        if not ok and start is not None:
            runs.append((offset + start, offset + i))
            start = None
    if start is not None:
        runs.append((offset + start, offset + len(free)))
    return runs


def path_length(points: list[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def flatten(lane_list: list[tuple[Point, Point]]) -> list[Point]:
    return [p for lane in lane_list for p in lane]
