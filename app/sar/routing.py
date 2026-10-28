"""Flying around no-fly areas. The arena's autopilot never enters one: any leg that would cross a no-fly cell is
replaced by the shortest detour around it (A* over free cells, then cut down to as few straight legs as possible),
much as a geofence works in a flight controller. Planners can call the same function to cost a route honestly.
"""

import heapq
import math

import numpy as np

from app.sar.grid import Grid

Point = tuple[float, float]
_MOVES = [(dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def path_length(points: list[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def safe_path(points: list[Point], no_fly: np.ndarray, grid: Grid) -> list[Point]:
    """`points`, with every leg that would cross a no-fly cell replaced by a detour around it."""
    if not no_fly.any():
        return list(points)
    out = list(points[:1])
    for a, b in zip(points, points[1:]):
        out.extend([b] if is_clear(a, b, no_fly, grid) else _detour(a, b, no_fly, grid))
    return out


def is_clear(a: Point, b: Point, no_fly: np.ndarray, grid: Grid) -> bool:
    """True if the straight leg a -> b stays out of no-fly cells (checked every 1/20 of a cell, so only clips of a
    corner smaller than that can slip through)."""
    n = max(1, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / (grid.cell_m / 20)))
    ys, xs = np.linspace(a[0], b[0], n + 1), np.linspace(a[1], b[1], n + 1)
    rows = np.clip((ys // grid.cell_m).astype(int), 0, grid.rows - 1)
    cols = np.clip((xs // grid.cell_m).astype(int), 0, grid.cols - 1)
    return not no_fly[rows, cols].any()


def distance_field(grid: Grid, no_fly: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    """Flying distance in metres from cell `origin` to every cell, going around no-fly cells (Dijkstra over the same
    moves as the detours). Unreachable cells are inf."""
    free = ~no_fly
    dist = np.full(grid.shape, math.inf)
    dist[origin] = 0.0
    frontier = [(0.0, origin)]
    while frontier:
        d, (r, c) = heapq.heappop(frontier)
        if d > dist[r, c]:
            continue
        for nr, nc, cost in _neighbours(r, c, free):
            if d + cost * grid.cell_m < dist[nr, nc]:
                dist[nr, nc] = d + cost * grid.cell_m
                heapq.heappush(frontier, (dist[nr, nc], (nr, nc)))
    return dist


def _neighbours(r: int, c: int, free: np.ndarray):
    """Free 8-neighbours of (r, c) with step cost in cells; diagonals may not cut the corner of a blocked cell."""
    rows, cols = free.shape
    for dr, dc, cost in _MOVES:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols) or not free[nr, nc]:
            continue
        if dr and dc and not (free[r + dr, c] and free[r, c + dc]):
            continue
        yield nr, nc, cost


def _detour(a: Point, b: Point, no_fly: np.ndarray, grid: Grid) -> list[Point]:
    cells = _astar(grid.cell_of(*a), grid.cell_of(*b), ~no_fly)
    if cells is None:
        raise ValueError("no route around the no-fly area")
    points = [a, *(grid.centre_m(*c) for c in cells[1:-1]), b]
    return _shortcut(points, no_fly, grid)[1:]


def _astar(start: tuple[int, int], goal: tuple[int, int], free: np.ndarray) -> list[tuple[int, int]] | None:
    """Shortest 8-connected path over free cells; diagonal moves may not cut the corner of a blocked cell."""
    best = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    frontier = [(0.0, start)]
    while frontier:
        _, cell = heapq.heappop(frontier)
        if cell == goal:
            path = [cell]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            return path[::-1]
        for nr, nc, cost in _neighbours(*cell, free):
            g = best[cell] + cost
            if g < best.get((nr, nc), math.inf):
                best[(nr, nc)] = g
                came_from[(nr, nc)] = cell
                heapq.heappush(frontier, (g + math.dist((nr, nc), goal), (nr, nc)))
    return None


def _shortcut(points: list[Point], no_fly: np.ndarray, grid: Grid) -> list[Point]:
    """Drop every corner that a straight, clear leg can skip."""
    out, i = [points[0]], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not is_clear(points[i], points[j], no_fly, grid):
            j -= 1
        out.append(points[j])
        i = j
    return out
