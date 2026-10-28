"""Baseline: probability-greedy. Each sortie sweeps the block with the most probability of success per second of
flight that fits in one battery, skipping blocks other airborne drones are already sweeping.
"""

from app.sar.planners.patterns import Point, flatten, lanes, path_length
from app.sar.sim import Drone, View

BLOCK_LANES = 4  # block width in lanes; blocks are square


class ProbabilityGreedy:
    name = "greedy"

    def __init__(self) -> None:
        self._claimed: dict[int, tuple[slice, slice]] = {}

    def plan(self, view: View, drone: Drone) -> list[Point]:
        self._claimed.pop(drone.id, None)
        grid = view.map.grid
        lane_m = view.fleet.footprint_m[0] * (1 - view.fleet.overlap)
        side = max(1, round(BLOCK_LANES * lane_m / grid.cell_m))
        value = view.belief.poa * view.detection.pod(view.fleet.gsd_cm, view.map.terrain)
        for rows, cols in self._claimed.values():
            value[rows, cols] = 0.0
        base = (drone.y, drone.x)
        best, best_route = 0.0, []
        for r0 in range(0, grid.rows, side):
            for c0 in range(0, grid.cols, side):
                block = (slice(r0, min(r0 + side, grid.rows)), slice(c0, min(c0 + side, grid.cols)))
                gain = float(value[block].sum())
                if gain <= best * 1e-3:
                    continue
                route = flatten(lanes(grid, view.map.no_fly, *block, lane_m))
                seconds = path_length([base, *route, base]) / view.fleet.speed_mps
                if route and seconds <= 0.95 * view.fleet.endurance_s and gain / seconds > best:
                    best, best_route, chosen = gain / seconds, route, block
        if best_route:
            self._claimed[drone.id] = chosen
        return best_route

