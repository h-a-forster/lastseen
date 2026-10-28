"""Baselines centred on the initial planning point (IPP): expanding square, concentric circles and sectors.

They use no probability map, only the fact that people are usually found near where they were last seen, and they
are strong: in the SAREnv benchmark (Bristol, 2025) they found as many subjects as probability-greedy search.
Each builds fixed paths once; drones fly them in battery-sized pieces and start another pass when one ends.
"""

import math
from collections import deque

from app.sar.detection import DetectionModel
from app.sar.interface import DroneState, FleetSpec, Look, Waypoint
from app.sar.planners.patterns import Point, at_altitude, densify, take_sortie
from app.sar.scenario import SearchMap

STEP_M = 50.0  # path resolution


class _PathPlanner:
    """One shared path (drones take the next piece in launch order) or one path per drone."""

    def reset(self, search_map: SearchMap, fleet: FleetSpec, detection: DetectionModel, stats: dict, seed: int) -> None:
        self.fleet = fleet
        grid = search_map.grid
        height, width = grid.rows * grid.cell_m, grid.cols * grid.cell_m
        centre = grid.centre_m(*search_map.ipp)
        reach = max(math.hypot(y - centre[0], x - centre[1]) for y in (0, height) for x in (0, width))
        lane_m = fleet.footprint_m[0] * (1 - fleet.overlap)
        self._full = [
            [p for p in densify(path, STEP_M) if 0 <= p[0] <= height and 0 <= p[1] <= width]
            for path in self.paths(centre, lane_m, reach, fleet.drones)
        ]
        self._paths = [deque(path) for path in self._full]

    def paths(self, centre: Point, lane_m: float, reach_m: float, drones: int) -> list[list[Point]]:
        raise NotImplementedError

    def observe(self, event: Look) -> None:
        pass

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        i = drone.id % len(self._paths)
        if not self._paths[i]:
            self._paths[i].extend(self._full[i])
        path = self._paths[i]
        route = take_sortie(path, (drone.y_m, drone.x_m), self.fleet.speed_mps, self.fleet.endurance_s)
        return at_altitude(route, self.fleet.alt_m)

    def probability_map(self):
        return None


class ExpandingSquare(_PathPlanner):
    """The standard IAMSAR expanding-square search: legs of 1, 1, 2, 2, 3, 3 ... lane widths, turning 90 degrees."""

    name = "spiral"

    def paths(self, centre: Point, lane_m: float, reach_m: float, drones: int) -> list[list[Point]]:
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


class Concentric(_PathPlanner):
    """Circles of growing radius, one lane apart, each joined to the next by a short radial step."""

    name = "concentric"

    def paths(self, centre: Point, lane_m: float, reach_m: float, drones: int) -> list[list[Point]]:
        points = [centre]
        radius = lane_m / 2
        while radius <= reach_m:
            points.extend(_arc(centre, radius, 0.0, 2 * math.pi))
            radius += lane_m
        return [points]


class Sector(_PathPlanner):
    """One wedge per drone ("pizza" slices), each swept outward in arcs that alternate direction."""

    name = "sector"

    def paths(self, centre: Point, lane_m: float, reach_m: float, drones: int) -> list[list[Point]]:
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


def _arc(centre: Point, radius: float, a0: float, a1: float) -> list[Point]:
    n = max(2, math.ceil(abs(a1 - a0) * radius / STEP_M) + 1)
    return [(centre[0] + radius * math.sin(a0 + (a1 - a0) * k / (n - 1)),
             centre[1] + radius * math.cos(a0 + (a1 - a0) * k / (n - 1))) for k in range(n)]
