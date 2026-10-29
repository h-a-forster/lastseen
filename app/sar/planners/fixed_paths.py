"""Fixed-pattern baselines. They ignore the probability map and fly a pattern decided at the start:

- lawnmower: one strip of the area per drone, swept in parallel lanes (the classic parallel-track search);
- spiral, concentric, sector: centred on a datum, at first the initial planning point (IPP), using only the fact
  that people are usually found near where they were last seen. In the SAREnv benchmark (Bristol, 2025) these found
  as many subjects as probability-greedy search.

Each pattern is built once as a path. Drones fly it in battery-sized pieces, costing the way home around no-fly
areas, and start another pass when it ends, as a real team would re-sweep.

Wrapped in the Spark advisor (e.g. "spiral+spark"), a pattern centred on a datum moves it whenever a new reading of
the reports points somewhere new, as a team restarts its search on a new datum after a sighting: the centre becomes
the block one battery can sweep that holds the most probability under the reading, and the pattern starts over from
there, forgetting where it has already searched. It is the fixed-pattern way to use Spark's readings, to set against
planners that fold the same readings into a map they keep.

When the pattern restarts, drones in the air break off and fly the new pattern's next pieces with the battery they
have left, as a team would radio its pilots.

When a drone is lost, the unflown rest of its sortie goes back on its path, and a drone whose own path is done takes
over a lost drone's path before starting another pass, as a team would hand on a lost drone's sector.
"""

import math
from collections import deque

import numpy as np

from app.sar.detection import DetectionModel
from app.sar.interface import DroneLost, DroneState, FleetSpec, Hypotheses, Observation, Waypoint
from app.sar.planners.patterns import Point, at_altitude, densify, flatten, lanes, take_sortie
from app.sar.prior import ring_prior
from app.sar.routing import distance_field, safe_path
from app.sar.scenario import SearchMap

STEP_M = 50.0  # path resolution


class PathPlanner:
    """One shared path (drones take the next piece in launch order) or one path per drone."""

    centred = False  # laid out around a datum, which a new reading of the reports can move

    def reset(self, search_map: SearchMap, fleet: FleetSpec, detection: DetectionModel, stats: dict, seed: int) -> None:
        self.map, self.fleet, self.grid = search_map, fleet, search_map.grid
        self._home_m = distance_field(search_map.grid, search_map.no_fly, search_map.base)
        self._lane_m = fleet.footprint_m[0] * (1 - fleet.overlap)
        self._budget_m = 0.95 * fleet.endurance_s * fleet.speed_mps
        self._centre = search_map.grid.centre_m(*search_map.ipp)
        if self.centred:
            rings = stats["categories"][search_map.category]["rings_km"]
            self._base_poa = ring_prior(search_map.grid, search_map.ipp, rings, weight=search_map.passability)[0]
            self._block_m = math.sqrt(self._budget_m * self._lane_m)  # the square one battery can sweep
            self._base_block = most_likely_block(self._base_poa, self.grid.cell_m, self._block_m)
        self._lost: list[int] = []
        self._notes: list[dict] = []
        self._diverted: set[int] | None = None  # after a restart: the airborne drones already sent to the new pattern
        self._build()

    def _build(self) -> None:
        self._full = [self._flyable(path) for path in self.paths(self.map, self._lane_m, self.fleet.drones)]
        self._todo = [deque(path) for path in self._full]

    def paths(self, search_map: SearchMap, lane_m: float, drones: int) -> list[list[Point]]:
        raise NotImplementedError

    def _flyable(self, path: list[Point]) -> list[Point]:
        """Drop points outside the area or over no-fly cells, then route around no-fly areas, so the battery
        planning counts the detours the autopilot will fly. Points no sortie can reach are dropped too: a detour
        corner that falls in a no-fly cell (the autopilot detours on every leg anyway), or one too far out to fly to
        and back on one battery."""
        kept = [p for p in densify(path, STEP_M) if self._reachable(p)]
        return [p for p in densify(safe_path(kept, self.map.no_fly, self.grid), STEP_M) if self._reachable(p)]

    def _reachable(self, p: Point) -> bool:
        return _allowed(self.map, p) and 2 * self._home_m[self.grid.cell_of(*p)] <= self._budget_m

    def observe(self, event: Observation) -> None:
        if isinstance(event, DroneLost):
            self._todo[event.drone % len(self._todo)].extendleft(reversed([(y, x) for y, x, _ in event.unflown]))
            self._lost.append(event.drone)
        elif isinstance(event, Hypotheses) and self.centred:
            self._move_datum(event)

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        queue = self._todo[self._path_for(drone.id)]
        route = take_sortie(queue, lambda p: self._home_m[self.grid.cell_of(*p)], self._budget_m)
        return at_altitude(route, self.fleet.alt_m)

    def retask(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint] | None:
        """After a restart, each airborne drone once breaks off for the new pattern's next piece."""
        if self._diverted is None or drone.id in self._diverted:
            return None
        self._diverted.add(drone.id)
        queue = self._todo[self._path_for(drone.id)]
        budget_m = 0.95 * drone.battery_s * self.fleet.speed_mps
        route = take_sortie(queue, lambda p: self._home_m[self.grid.cell_of(*p)], budget_m, (drone.y_m, drone.x_m))
        return at_altitude(route, self.fleet.alt_m)

    def _path_for(self, drone_id: int) -> int:
        """The drone's own path; once that is done, a lost drone's unfinished one; else its own again, from the start."""
        own = drone_id % len(self._todo)
        if self._todo[own]:
            return own
        for lost in self._lost:
            if self._todo[lost % len(self._todo)]:
                return lost % len(self._todo)
        self._todo[own].extend(self._full[own])
        return own

    def _move_datum(self, reading: Hypotheses) -> None:
        """Restart the pattern on the block one battery can sweep that the reading makes most likely. If that is the
        statistics' own most likely block (within a lane), the reading says nothing about where to look and the datum
        is the IPP again. Nothing restarts unless the datum moves by more than a lane."""
        house = self.grid.centre_m(*self.map.ipp)
        datum = most_likely_block(reading.likelihood(self._base_poa) * self._base_poa, self.grid.cell_m, self._block_m)
        if math.dist(datum, self._base_block) < self._lane_m:
            datum = house
        if math.dist(datum, self._centre) < self._lane_m:
            return
        self._centre = datum
        self._build()
        self._diverted = set()
        self._notes.append({"kind": "restart", "text": f"Restarted the pattern on Spark's newest reading, centred "
                                                       f"{math.dist(datum, house) / 1000:.1f} km from the house"})

    def drain_notes(self) -> list[dict]:
        notes, self._notes = self._notes, []
        return notes

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
    centred = True

    def paths(self, search_map: SearchMap, lane_m: float, drones: int) -> list[list[Point]]:
        centre, reach_m = _frame(search_map, self._centre)
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
    centred = True

    def paths(self, search_map: SearchMap, lane_m: float, drones: int) -> list[list[Point]]:
        centre, reach_m = _frame(search_map, self._centre)
        points = [centre]
        radius = lane_m / 2
        while radius <= reach_m:
            points.extend(_arc(centre, radius, 0.0, 2 * math.pi))
            radius += lane_m
        return [points]


class Sector(PathPlanner):
    """One wedge per drone ("pizza" slices), each swept outward in arcs that alternate direction."""

    name = "sector"
    centred = True

    def paths(self, search_map: SearchMap, lane_m: float, drones: int) -> list[list[Point]]:
        centre, reach_m = _frame(search_map, self._centre)
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


def _frame(search_map: SearchMap, centre: Point) -> tuple[Point, float]:
    """The datum, and the distance from it to the farthest corner of the area."""
    height, width = search_map.grid.rows * search_map.grid.cell_m, search_map.grid.cols * search_map.grid.cell_m
    return centre, max(math.hypot(y - centre[0], x - centre[1]) for y in (0, height) for x in (0, width))


def most_likely_block(poa: np.ndarray, cell_m: float, side_m: float) -> Point:
    """Centre, in metres, of the square of about `side_m` (clipped to the map) holding the most probability."""
    side = max(1, min(round(side_m / cell_m), *poa.shape))
    total = np.pad(poa.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = total[side:, side:] - total[:-side, side:] - total[side:, :-side] + total[:-side, :-side]
    r, c = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return (r + side / 2) * cell_m, (c + side / 2) * cell_m


def _allowed(search_map: SearchMap, p: Point) -> bool:
    """Inside the area and not over a no-fly cell."""
    grid = search_map.grid
    inside = 0 <= p[0] <= grid.rows * grid.cell_m and 0 <= p[1] <= grid.cols * grid.cell_m
    return inside and not search_map.no_fly[grid.cell_of(*p)]


def _arc(centre: Point, radius: float, a0: float, a1: float) -> list[Point]:
    n = max(2, math.ceil(abs(a1 - a0) * radius / STEP_M) + 1)
    angles = np.linspace(a0, a1, n)
    return [(centre[0] + radius * math.sin(a), centre[1] + radius * math.cos(a)) for a in angles]
