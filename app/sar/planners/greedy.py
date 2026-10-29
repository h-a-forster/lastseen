"""Reference informed baseline: probability-greedy. It keeps its own map (distance-ring prior, Bayes update after each
miss, diffusion as time passes), and each sortie sweeps the block with the most probability of success per second of
flight that fits in one battery, skipping blocks other airborne drones are already sweeping.

When a new reading arrives mid-sortie, a drone in the air switches to another block if, under the new map, that block
is a better use of the battery it has left (`retask`); with no new reading it flies its block to the end.

Hypotheses from a model are a separate factor: the map is multiplied by how much more likely the latest reading makes
each cell, raised to the power TRUST (1 = take it at face value, 0 = ignore it). A newer reading replaces the older
one exactly, so the brief is never counted twice. The reading doesn't spread over time like the map does: each one
already describes where the person may be by then, and every new report brings a new reading.
"""

import numpy as np

from app.sar.belief import Belief, mobility
from app.sar.detection import DetectionModel
from app.sar.interface import DroneLost, DroneState, FleetSpec, Hypotheses, Look, Observation, Waypoint
from app.sar.planners.patterns import at_altitude, flatten, lanes, path_length
from app.sar.prior import ring_prior
from app.sar.scenario import SearchMap

BLOCK_LANES = 4  # block width in lanes; blocks are square
PREDICT_EVERY_S = 60.0
TRUST = 1.0


class ProbabilityGreedy:
    name = "greedy"

    def reset(self, search_map: SearchMap, fleet: FleetSpec, detection: DetectionModel, stats: dict, seed: int) -> None:
        self.map, self.fleet = search_map, fleet
        rings = stats["categories"][search_map.category]["rings_km"]
        self.belief = Belief(*ring_prior(search_map.grid, search_map.ipp, rings, weight=search_map.passability))
        self.pod = detection.pod(fleet.gsd_cm, search_map.terrain)
        self.passability = search_map.passability
        self._prior = self.belief.poa.copy()
        self._likelihood = np.ones(search_map.grid.shape)  # the latest reading of the reports; all 1 before any
        self._claimed: dict[int, tuple[slice, slice]] = {}
        self._readings = 0  # readings received so far
        self._planned_with: dict[int, int] = {}  # per drone, how many readings its block was chosen with
        self._home = search_map.grid.centre_m(*search_map.base)
        self._t_s = 0.0
        self._notes: list[dict] = []

    def observe(self, event: Observation) -> None:
        self._advance(event.t_s)
        if isinstance(event, Look):
            self.belief.observe_miss(event.cells, event.pod)
        elif isinstance(event, Hypotheses):
            self._likelihood = event.likelihood(self._prior) ** TRUST
            self._readings += 1
        elif isinstance(event, DroneLost):  # its block is free again; the map still shows what it didn't photograph
            self._claimed.pop(event.drone, None)

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        self._advance(t_s)
        self._claimed.pop(drone.id, None)
        rate, block, route = self._best_block((drone.y_m, drone.x_m), 0.95 * self.fleet.endurance_s)
        if route:
            self._claim(drone.id, block)
            self._notes.append({"kind": "sortie", "drone": drone.id,
                                "text": f"Drone {drone.id + 1}: block with {rate * 60:.1%} chance of finding per minute of flight"})
        return at_altitude(route, self.fleet.alt_m)

    def retask(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint] | None:
        if self._planned_with.get(drone.id) == self._readings:  # nothing new since this block was chosen
            return None
        self._advance(t_s)
        current = self._claimed.pop(drone.id, None)
        rate, block, route = self._best_block((drone.y_m, drone.x_m), 0.95 * drone.battery_s)
        if not route or block == current:
            if current is not None:
                self._claim(drone.id, current)
            return None
        self._claim(drone.id, block)
        self._notes.append({"kind": "sortie", "drone": drone.id,
                            "text": f"Drone {drone.id + 1} retasked mid-flight: the new reading makes another block "
                                    f"worth {rate * 60:.1%} per minute"})
        return at_altitude(route, self.fleet.alt_m)

    def _claim(self, drone_id: int, block: tuple[slice, slice]) -> None:
        self._claimed[drone_id] = block
        self._planned_with[drone_id] = self._readings

    def _best_block(self, start: tuple[float, float], budget_s: float) -> tuple[float, tuple[slice, slice] | None, list]:
        """The unclaimed block with the most chance of finding per second of flight from `start`, along the block and
        home, within `budget_s`."""
        grid = self.map.grid
        lane_m = self.fleet.footprint_m[0] * (1 - self.fleet.overlap)
        side = max(1, round(BLOCK_LANES * lane_m / grid.cell_m))
        value = self.probability_map() * self.pod
        for rows, cols in self._claimed.values():
            value[rows, cols] = 0.0
        best, best_block, best_route = 0.0, None, []
        for r0 in range(0, grid.rows, side):
            for c0 in range(0, grid.cols, side):
                block = (slice(r0, min(r0 + side, grid.rows)), slice(c0, min(c0 + side, grid.cols)))
                gain = float(value[block].sum())
                if gain <= best * 1e-3:
                    continue
                route = flatten(lanes(grid, self.map.no_fly, *block, lane_m))
                seconds = path_length([start, *route, self._home]) / self.fleet.speed_mps
                if route and seconds <= budget_s and gain / seconds > best:
                    best, best_block, best_route = gain / seconds, block, route
        return best, best_block, best_route

    def probability_map(self):
        """The map with the latest reading applied, keeping the probability inside the area unchanged."""
        poa = self.belief.poa * self._likelihood
        return poa * (self.belief.poa.sum() / poa.sum())

    def drain_notes(self) -> list[dict]:
        notes, self._notes = self._notes, []
        return notes

    def _advance(self, t_s: float) -> None:
        while self._t_s + PREDICT_EVERY_S <= t_s:
            self._t_s += PREDICT_EVERY_S
            self.belief.predict(mobility(self._t_s), self.passability)
