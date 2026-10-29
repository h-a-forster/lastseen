"""Flight-pattern geometry shared by planners."""

import math
from collections import deque
from collections.abc import Callable

import numpy as np

from app.sar.grid import Grid
from app.sar.routing import path_length  # noqa: F401  (re-exported for planners)

Point = tuple[float, float]


def lanes(grid: Grid, no_fly: np.ndarray, rows: slice, cols: slice, lane_m: float) -> list[tuple[Point, Point]]:
    """Back-and-forth lanes (north-south) covering a block of cells, split around no-fly cells. Each lane is (start, end),
    from the centre of its first free cell to the centre of its last, so no waypoint sits in a no-fly cell."""
    x0, x1 = cols.start * grid.cell_m, cols.stop * grid.cell_m
    out: list[tuple[Point, Point]] = []
    n = max(1, math.ceil((x1 - x0) / lane_m))
    for i in range(n):
        x = x0 + (i + 0.5) * (x1 - x0) / n
        col = min(int(x // grid.cell_m), grid.cols - 1)
        segments = [(((a + 0.5) * grid.cell_m, x), ((b - 0.5) * grid.cell_m, x)) for a, b in _free_runs(~no_fly[rows, col], rows.start)]
        if i % 2:
            segments = [(end, start) for start, end in reversed(segments)]
        out.extend(segments)
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


def take_sortie(path: deque[Point], home_m: Callable[[Point], float], budget_m: float,
                start: Point | None = None) -> list[Point]:
    """Pop the longest prefix of `path` a drone can fly out to, along and back from within `budget_m` of flight.
    `home_m(p)` is the flying distance between base and p (around no-fly areas); `start` is where an airborne drone
    is (None: at base). From base, raises if even a full battery can't reach the next point, since every later sortie
    would come back empty and the pattern would stop for good."""
    route: list[Point] = []
    flown = 0.0
    while path:
        nxt = path[0]
        leg = math.dist(route[-1] if route else start, nxt) if route or start else home_m(nxt)
        if flown + leg + home_m(nxt) > budget_m:
            if not route and start is None:
                raise ValueError(f"no sortie can reach ({nxt[0]:.0f}, {nxt[1]:.0f}) and fly home on one battery")
            break
        flown += leg
        route.append(path.popleft())
    return route
