"""The map planner `ours`: a probability map, and sorties that tour several promising blocks.

Same map stack as `greedy` (ISRID prior, Bayes misses, time diffusion, Spark likelihood), but each sortie is a short
tour of up to MAX_TOUR adjacent high-yield blocks, so one battery does more searching before the turnaround that
costs greedy so much air time. The team's planner (`hedge.py`) uses it for the drone that follows a sighting.
Benchmark with:

    uv run python -m app.sar.bench --planners ours,greedy,spiral,lawnmower --seeds 40 --workers 4
"""

from dataclasses import dataclass

import numpy as np

from app.sar.belief import Belief, mobility
from app.sar.detection import DetectionModel
from app.sar.interface import DroneLost, DroneState, FleetSpec, Hypotheses, Look, Observation, Waypoint
from app.sar.planners.patterns import Point, at_altitude, flatten, lanes, path_length
from app.sar.prior import ring_prior
from app.sar.scenario import SearchMap

BLOCK_LANES = 4  # block width in lanes, the same square tiling as greedy
MAX_TOUR = 4  # blocks per sortie
SEEDS = 8  # single-block tours the search grows from
PREDICT_EVERY_S = 60.0
RETASK_GAIN = 1.05  # a new reading retasks a drone only if its best tour finds people this much faster

Block = tuple[slice, slice]  # (rows, cols) of the grid
BlockKey = tuple[int, int]  # (block row, block col) in the tiling


@dataclass(frozen=True)
class Candidate:
    key: BlockKey
    block: Block
    gain: float  # search value inside the block
    route: list[Point]  # lane endpoints in flying order


class SearchPlanner:
    name = "ours"

    def reset(self, search_map: SearchMap, fleet: FleetSpec, detection: DetectionModel, stats: dict, seed: int) -> None:
        self.map, self.fleet = search_map, fleet
        rings = stats["categories"][search_map.category]["rings_km"]
        self.belief = Belief(*ring_prior(search_map.grid, search_map.ipp, rings, weight=search_map.passability))
        self.pod = detection.pod(fleet.gsd_cm, search_map.terrain)
        self.passability = search_map.passability
        self._prior = self.belief.poa.copy()
        self._likelihood = np.ones(search_map.grid.shape)  # the latest reading of the reports; all 1 before any
        self._claimed: dict[int, list[Block]] = {}
        self._readings = 0  # readings received so far
        self._planned_with: dict[int, int] = {}  # per drone, how many readings its tour was chosen with
        self._tour_rate: dict[int, float] = {}  # per drone, the rate of the tour it is flying
        self._home = search_map.grid.centre_m(*search_map.base)
        self._t_s = 0.0
        self._notes: list[dict] = []

    def observe(self, event: Observation) -> None:
        self._advance(event.t_s)
        if isinstance(event, Look):
            self.belief.observe_miss(event.cells, event.pod)
        elif isinstance(event, Hypotheses):
            # The less weight a reading leaves on the map, the more it is trusted.
            trust = max(0.2, 2.0 * (1.0 - event.rest_weight))
            self._likelihood = event.likelihood(self._prior) ** trust
            self._readings += 1
        elif isinstance(event, DroneLost):  # its blocks are free again
            self._claimed.pop(event.drone, None)

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        self._advance(t_s)
        self._claimed.pop(drone.id, None)
        rate, blocks, route = self._best_tour((drone.y_m, drone.x_m), 0.95 * self.fleet.endurance_s)
        if route:
            self._claim(drone.id, blocks)
            self._tour_rate[drone.id] = rate
            self._notes.append({
                "kind": "sortie",
                "drone": drone.id,
                "text": f"Drone {drone.id + 1} searches the {_likeliest(len(blocks))} it can reach on this battery",
            })
        return at_altitude(route, self.fleet.alt_m)

    def retask(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint] | None:
        if self._planned_with.get(drone.id) == self._readings:  # nothing new since this tour was chosen
            return None
        self._advance(t_s)
        current = self._claimed.pop(drone.id, None)
        rate, blocks, route = self._best_tour((drone.y_m, drone.x_m), 0.95 * drone.battery_s)

        if not route or blocks == current:
            if current is not None:
                self._claim(drone.id, current)
            return None
        # The old rate was taken when the tour was chosen, usually from base on a full battery.
        old_rate = self._tour_rate.get(drone.id, 0.0)
        if old_rate > 0 and rate < old_rate * RETASK_GAIN:
            if current is not None:
                self._claim(drone.id, current)
            return None
        self._claim(drone.id, blocks)
        self._tour_rate[drone.id] = rate

        self._notes.append({
            "kind": "sortie",
            "drone": drone.id,
            "text": f"Drone {drone.id + 1} changes course mid-flight to the {_likeliest(len(blocks))} after the new reading",
        })
        return at_altitude(route, self.fleet.alt_m)

    def probability_map(self):
        """The map with the latest reading applied, keeping the probability inside the area unchanged."""
        poa = self.belief.poa * self._likelihood
        return poa * (self.belief.poa.sum() / poa.sum())

    def drain_notes(self) -> list[dict]:
        notes, self._notes = self._notes, []
        return notes

    def _claim(self, drone_id: int, blocks: list[Block]) -> None:
        self._claimed[drone_id] = blocks
        self._planned_with[drone_id] = self._readings

    def _best_tour(self, start: Point, budget_s: float) -> tuple[float, list[Block], list[Point]]:
        """The tour of adjacent unclaimed blocks with the most chance of finding per second of flight from `start`,
        through the blocks and home, within `budget_s`."""
        candidates = self._candidates()
        by_key = {candidate.key: candidate for candidate in candidates}
        # Prefer seeds that are already strong as single-block sorties (amortised base transit).
        single = {candidate.key: self._score(start, [candidate], budget_s)[0] for candidate in candidates}
        ranked = sorted(candidates, key=lambda candidate: single[candidate.key], reverse=True)
        seeds = [candidate for candidate in ranked if single[candidate.key] > 0][:SEEDS]
        best_rate, best_blocks, best_route, best_s = 0.0, [], [], 0.0
        for seed in seeds:
            tour = [seed]
            # Pack adjacent high-mass blocks up to MAX_TOUR / battery: fill the sortie first (closes the
            # air-time gap), then compare packed tours by yield per second.
            while len(tour) < MAX_TOUR:
                in_tour = {candidate.key for candidate in tour}
                pick, pick_rate, pick_s = None, 0.0, 0.0
                for key in _neighbours(tour, by_key):
                    if key in in_tour:
                        continue
                    rate, route, seconds = self._score(start, [*tour, by_key[key]], budget_s)
                    if not route:
                        continue
                    # Prefer the extension with the best packed-tour score; tie-break on more flight time.
                    if pick is None or rate > pick_rate or (rate == pick_rate and seconds > pick_s):
                        pick, pick_rate, pick_s = by_key[key], rate, seconds
                if pick is None:
                    break
                tour.append(pick)
            rate, route, seconds = self._score(start, tour, budget_s)
            if rate > best_rate or (rate == best_rate and seconds > best_s):
                best_rate, best_blocks, best_route, best_s = rate, [c.block for c in tour], route, seconds
        return best_rate, best_blocks, best_route

    def _score(self, start: Point, tour: list[Candidate], budget_s: float) -> tuple[float, list[Point], float]:
        """Search value per second of flying `tour` from `start` and home, its route and seconds; an empty route if
        it doesn't fit in `budget_s`."""
        route = [point for candidate in tour for point in candidate.route]
        seconds = path_length([start, *route, self._home]) / self.fleet.speed_mps
        if seconds > budget_s:
            return 0.0, [], 0.0
        return sum(candidate.gain for candidate in tour) / seconds, route, seconds

    def _candidates(self) -> list[Candidate]:
        """Every unclaimed block worth searching, in row-major tiling order."""
        grid = self.map.grid
        lane_m = self.fleet.footprint_m[0] * (1 - self.fleet.overlap)
        side = max(1, round(BLOCK_LANES * lane_m / grid.cell_m))
        value = self.probability_map() * self.pod
        for blocks in self._claimed.values():
            for rows, cols in blocks:
                value[rows, cols] = 0.0
        candidates = []
        for br, r0 in enumerate(range(0, grid.rows, side)):
            for bc, c0 in enumerate(range(0, grid.cols, side)):
                block = (slice(r0, min(r0 + side, grid.rows)), slice(c0, min(c0 + side, grid.cols)))
                gain = float(value[block].sum())
                if gain <= 0.0:
                    continue
                route = flatten(lanes(grid, self.map.no_fly, *block, lane_m))
                if route:
                    candidates.append(Candidate((br, bc), block, gain, route))
        return candidates

    def _advance(self, t_s: float) -> None:
        while self._t_s + PREDICT_EVERY_S <= t_s:
            self._t_s += PREDICT_EVERY_S
            self.belief.predict(mobility(self._t_s), self.passability)


def _neighbours(tour: list[Candidate], by_key: dict[BlockKey, Candidate]) -> set[BlockKey]:
    """Block-grid 8-neighbours of any block already in the tour (adjacent chain constraint)."""
    keys = set()
    for candidate in tour:
        br, bc = candidate.key
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) != (0, 0) and (br + dr, bc + dc) in by_key:
                    keys.add((br + dr, bc + dc))
    return keys


def _likeliest(blocks: int) -> str:
    return "likeliest block" if blocks == 1 else f"{blocks} likeliest blocks"
