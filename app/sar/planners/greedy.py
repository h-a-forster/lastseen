"""Reference informed baseline: probability-greedy. It keeps its own map (distance-ring prior, Bayes update after each
miss, diffusion as time passes), and each sortie sweeps the block with the most probability of success per second of
flight that fits in one battery, skipping blocks other airborne drones are already sweeping.

Hypotheses from a model are folded in as a likelihood: the map is multiplied by how much more likely the hypotheses
make each cell, raised to the power TRUST (1 = take them at face value, 0 = ignore them). A newer reading replaces
the older one, so the brief is never counted twice.
"""

import numpy as np

from app.sar.belief import Belief, mobility
from app.sar.detection import DetectionModel
from app.sar.interface import DroneState, FleetSpec, Hypotheses, Look, Observation, Waypoint
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
        self._applied = np.ones(search_map.grid.shape)  # the hypotheses' likelihood currently folded into the map
        self._claimed: dict[int, tuple[slice, slice]] = {}
        self._t_s = 0.0

    def observe(self, event: Observation) -> None:
        self._advance(event.t_s)
        if isinstance(event, Look):
            self.belief.observe_miss(event.cells, event.pod)
        elif isinstance(event, Hypotheses):
            self._fold_in(event.likelihood(self._prior) ** TRUST)

    def _fold_in(self, likelihood: np.ndarray) -> None:
        """Swap the previous reading's likelihood for the new one, keeping the probability inside the area unchanged."""
        inside = self.belief.poa.sum()
        poa = self.belief.poa / self._applied * likelihood
        self.belief.poa = poa * inside / poa.sum()
        self._applied = likelihood

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        self._advance(t_s)
        self._claimed.pop(drone.id, None)
        grid = self.map.grid
        lane_m = self.fleet.footprint_m[0] * (1 - self.fleet.overlap)
        side = max(1, round(BLOCK_LANES * lane_m / grid.cell_m))
        value = self.belief.poa * self.pod
        for rows, cols in self._claimed.values():
            value[rows, cols] = 0.0
        base = (drone.y_m, drone.x_m)
        best, best_route = 0.0, []
        for r0 in range(0, grid.rows, side):
            for c0 in range(0, grid.cols, side):
                block = (slice(r0, min(r0 + side, grid.rows)), slice(c0, min(c0 + side, grid.cols)))
                gain = float(value[block].sum())
                if gain <= best * 1e-3:
                    continue
                route = flatten(lanes(grid, self.map.no_fly, *block, lane_m))
                seconds = path_length([base, *route, base]) / self.fleet.speed_mps
                if route and seconds <= 0.95 * self.fleet.endurance_s and gain / seconds > best:
                    best, best_route, chosen = gain / seconds, route, block
        if best_route:
            self._claimed[drone.id] = chosen
        return at_altitude(best_route, self.fleet.alt_m)

    def probability_map(self):
        return self.belief.poa

    def _advance(self, t_s: float) -> None:
        while self._t_s + PREDICT_EVERY_S <= t_s:
            self._t_s += PREDICT_EVERY_S
            self.belief.predict(mobility(self._t_s), self.passability)
