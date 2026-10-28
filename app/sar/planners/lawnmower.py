"""Baseline: split the area into one strip per drone and sweep each strip in parallel lanes, battery by battery.

This is the standard parallel-track pattern; it ignores the probability map entirely.
"""

from app.sar.planners.patterns import Point, lanes, path_length
from app.sar.sim import Drone, View


class Lawnmower:
    name = "lawnmower"

    def __init__(self) -> None:
        self._todo: dict[int, list[tuple[Point, Point]]] = {}

    def plan(self, view: View, drone: Drone) -> list[Point]:
        if drone.id not in self._todo:
            self._todo[drone.id] = self._strip(view, drone.id)
        todo = self._todo[drone.id]
        base = (drone.y, drone.x)
        route: list[Point] = []
        while todo:
            candidate = route + list(todo[0])
            if path_length([base, *candidate, base]) / view.fleet.speed_mps > 0.95 * view.fleet.endurance_s:
                break
            route = candidate
            todo.pop(0)
        return route

    def _strip(self, view: View, index: int) -> list[tuple[Point, Point]]:
        grid = view.map.grid
        n = view.fleet.drones
        c0, c1 = grid.cols * index // n, grid.cols * (index + 1) // n
        lane_m = view.fleet.footprint_m[0] * (1 - view.fleet.overlap)
        return lanes(grid, view.map.no_fly, slice(0, grid.rows), slice(c0, c1), lane_m)
