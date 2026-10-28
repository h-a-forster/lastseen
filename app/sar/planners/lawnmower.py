"""Baseline: split the area into one strip per drone and sweep each strip in parallel lanes, battery by battery.

This is the standard parallel-track pattern; it ignores the probability map entirely.
"""

from app.sar.detection import DetectionModel
from app.sar.interface import DroneState, FleetSpec, Look, Waypoint
from app.sar.planners.patterns import Point, at_altitude, lanes, path_length
from app.sar.scenario import SearchMap


class Lawnmower:
    name = "lawnmower"

    def reset(self, search_map: SearchMap, fleet: FleetSpec, detection: DetectionModel, stats: dict, seed: int) -> None:
        self.fleet = fleet
        grid = search_map.grid
        lane_m = fleet.footprint_m[0] * (1 - fleet.overlap)
        self._todo = [
            lanes(grid, search_map.no_fly, slice(0, grid.rows), slice(grid.cols * i // fleet.drones, grid.cols * (i + 1) // fleet.drones), lane_m)
            for i in range(fleet.drones)
        ]

    def observe(self, event: Look) -> None:
        pass

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        todo = self._todo[drone.id]
        base = (drone.y_m, drone.x_m)
        route: list[Point] = []
        while todo:
            candidate = route + list(todo[0])
            if path_length([base, *candidate, base]) / self.fleet.speed_mps > 0.95 * self.fleet.endurance_s:
                break
            route = candidate
            todo.pop(0)
        return at_altitude(route, self.fleet.alt_m)

    def probability_map(self):
        return None
