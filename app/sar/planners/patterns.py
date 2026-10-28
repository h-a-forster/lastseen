"""Flight-pattern geometry shared by planners."""

import math
from collections import deque

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


def at_altitude(points: list[Point], alt_m: float) -> list[tuple[float, float, float]]:
    return [(y, x, alt_m) for y, x in points]


def densify(points: list[Point], step_m: float) -> list[Point]:
    """Insert points so no segment is longer than `step_m`, so a path can be cut anywhere into battery-sized sorties."""
    out = points[:1]
    for a, b in zip(points, points[1:]):
        n = max(1, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / step_m))
        out.extend((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1))
    return out


def take_sortie(path: deque[Point], base: Point, speed_mps: float, endurance_s: float) -> list[Point]:
    """Pop the longest prefix of `path` a drone can fly from base and back on one battery (5% margin)."""
    budget_m = 0.95 * endurance_s * speed_mps
    route: list[Point] = []
    flown = 0.0
    while path:
        here, nxt = route[-1] if route else base, path[0]
        leg = math.hypot(nxt[0] - here[0], nxt[1] - here[1])
        if flown + leg + math.hypot(nxt[0] - base[0], nxt[1] - base[1]) > budget_m:
            break
        flown += leg
        route.append(path.popleft())
    return route
