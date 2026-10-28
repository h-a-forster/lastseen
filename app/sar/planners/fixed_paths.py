"""Fixed-pattern baselines. They ignore the probability map and fly a pattern decided at the start:

- lawnmower: one strip of the area per drone, swept in parallel lanes (the classic parallel-track search);
- spiral, concentric, sector: centred on the initial planning point (IPP), using only the fact that people are
  usually found near where they were last seen. In the SAREnv benchmark (Bristol, 2025) these found as many
  subjects as probability-greedy search.

Each pattern is built once as a path. Drones fly it in battery-sized pieces, costing the way home around no-fly
areas, and start another pass when it ends, as a real team would re-sweep.
"""

import math
from collections import deque

import numpy as np

from app.sar.detection import DetectionModel
from app.sar.interface import DroneState, FleetSpec, Look, Waypoint
from app.sar.planners.patterns import Point, at_altitude, densify, flatten, lanes, take_sortie
from app.sar.routing import distance_field, safe_path
from app.sar.scenario import SearchMap

STEP_M = 50.0  # path resolution


class PathPlanner:
    """One shared path (drones take the next piece in launch order) or one path per drone."""

    def reset(self, search_map: SearchMap, fleet: FleetSpec, detection: DetectionModel, stats: dict, seed: int) -> None:
        self.fleet, self.grid = fleet, search_map.grid
        self._home_m = distance_field(search_map.grid, search_map.no_fly, search_map.base)
        lane_m = fleet.footprint_m[0] * (1 - fleet.overlap)
        self._full = [self._flyable(search_map, path) for path in self.paths(search_map, lane_m, fleet.drones)]
        self._todo = [deque(path) for path in self._full]

    def paths(self, search_map: SearchMap, lane_m: float, drones: int) -> list[list[Point]]:
        raise NotImplementedError

    @staticmethod
    def _flyable(search_map: SearchMap, path: list[Point]) -> list[Point]:
        """Drop points outside the area or over no-fly cells, then route around no-fly areas, so the battery
        planning counts the detours the autopilot will fly."""
        kept = [p for p in densify(path, STEP_M) if _allowed(search_map, p)]
        return densify(safe_path(kept, search_map.no_fly, search_map.grid), STEP_M)

    def observe(self, event: Look) -> None:
        pass

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        i = drone.id % len(self._todo)
        if not self._todo[i]:
            self._todo[i].extend(self._full[i])
        budget_m = 0.95 * self.fleet.endurance_s * self.fleet.speed_mps
        route = take_sortie(self._todo[i], lambda p: self._home_m[self.grid.cell_of(*p)], budget_m)
        return at_altitude(route, self.fleet.alt_m)

    def probability_map(self):
        return None


class Lawnmower(PathPlanner):
    name = "lawnmower"

    def paths(self, search_map: SearchMap, lane_m: float, drones: int) -> list[list[Point]]:
        grid = search_map.grid
        return [flatten(lanes(grid, search_map.no_fly, slice(0, grid.rows),
                              slice(grid.cols * i // drones, grid.cols * (i + 1) // drones), lane_m))
                for i in range(drones)]


class ExpandingSquare(PathPlanner):
    """The standard IAMSAR expanding-square search: legs of 1, 1, 2, 2, 3, 3 ... lane widths, turning 90 degrees."""

    name = "spiral"

    def paths(self, search_map: SearchMap, lane_m: float, drones: int) -> list[list[Point]]:
        centre, reach_m = _ipp_frame(search_map)
        y, x = centre
        points = [centre]
        headings = ((-1, 0), (0, 1), (1, 0), (0, -1))
        leg = 0
        while (length := (leg // 2 + 1) * lane_m) / 2 <= reach_m + lane_m:
            dy, dx = headings[leg % 4]
            y, x = y + dy * length, x + dx * length
            points.append((y, x))
            leg += 1
        return [points]


class Concentric(PathPlanner):
    """Circles of growing radius, one lane apart, each joined to the next by a short radial step."""

    name = "concentric"

    def paths(self, search_map: SearchMap, lane_m: float, drones: int) -> list[list[Point]]:
        centre, reach_m = _ipp_frame(search_map)
        points = [centre]
        radius = lane_m / 2
        while radius <= reach_m:
            points.extend(_arc(centre, radius, 0.0, 2 * math.pi))
            radius += lane_m
        return [points]


class Sector(PathPlanner):
    """One wedge per drone ("pizza" slices), each swept outward in arcs that alternate direction."""

    name = "sector"

    def paths(self, search_map: SearchMap, lane_m: float, drones: int) -> list[list[Point]]:
        centre, reach_m = _ipp_frame(search_map)
        wedge = 2 * math.pi / drones
        out = []
        for i in range(drones):
            a0, a1 = i * wedge, (i + 1) * wedge
            points, radius, outward = [centre], lane_m / 2, True
            while radius <= reach_m:
                points.extend(_arc(centre, radius, a0, a1) if outward else _arc(centre, radius, a1, a0))
                radius, outward = radius + lane_m, not outward
            out.append(points)
        return out


def _ipp_frame(search_map: SearchMap) -> tuple[Point, float]:
    """The IPP in metres, and the distance from it to the farthest corner of the area."""
    grid = search_map.grid
    height, width = grid.rows * grid.cell_m, grid.cols * grid.cell_m
    centre = grid.centre_m(*search_map.ipp)
    return centre, max(math.hypot(y - centre[0], x - centre[1]) for y in (0, height) for x in (0, width))


def _allowed(search_map: SearchMap, p: Point) -> bool:
    """Inside the area and not over a no-fly cell."""
    grid = search_map.grid
    inside = 0 <= p[0] <= grid.rows * grid.cell_m and 0 <= p[1] <= grid.cols * grid.cell_m
    return inside and not search_map.no_fly[grid.cell_of(*p)]


def _arc(centre: Point, radius: float, a0: float, a1: float) -> list[Point]:
    n = max(2, math.ceil(abs(a1 - a0) * radius / STEP_M) + 1)
    angles = np.linspace(a0, a1, n)
    return [(centre[0] + radius * math.sin(a), centre[1] + radius * math.cos(a)) for a in angles]
