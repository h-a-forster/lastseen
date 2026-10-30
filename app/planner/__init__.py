"""The team's decision system: prior, belief updates and multi-drone multi-block tasking.

Same map stack as `greedy` (ISRID prior, Bayes misses, time diffusion, Spark likelihood), but each sortie is a
TSP-lite tour of up to MAX_TOUR adjacent high-yield blocks so one battery does more searching before the turnaround
that costs greedy so much air time. Benchmark with:

    uv run python -m app.sar.bench --planners ours,greedy,spiral,lawnmower --seeds 40 --workers 4
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.sar.belief import Belief, mobility
from app.sar.detection import DetectionModel
from app.sar.interface import DroneLost, DroneState, FleetSpec, Hypotheses, Look, Observation, Waypoint
from app.sar.planners.patterns import Point, at_altitude, flatten, lanes, path_length
from app.sar.prior import ring_prior
from app.sar.scenario import SearchMap

BLOCK_LANES = 4  # same coarse tiling as greedy
MAX_TOUR = 4  # blocks per sortie (TSP-lite)
SEED_K = 8  # single-block seeds from which tours are grown
PREDICT_EVERY_S = 60.0
TRUST = 1.0

Block = tuple[slice, slice]
BlockKey = tuple[int, int]  # (block-row, block-col) in the coarse tiling


@dataclass(frozen=True)
class _Cand:
    key: BlockKey
    block: Block
    gain: float
    route: list[Point]


class SearchPlanner:
    name = "ours"

    def reset(self, search_map: SearchMap, fleet: FleetSpec, detection: DetectionModel, stats: dict, seed: int) -> None:
        self.map, self.fleet = search_map, fleet
        rings = stats["categories"][search_map.category]["rings_km"]
        self.belief = Belief(*ring_prior(search_map.grid, search_map.ipp, rings, weight=search_map.passability))
        self.pod = detection.pod(fleet.gsd_cm, search_map.terrain)
        self.passability = search_map.passability
        self._prior = self.belief.poa.copy()
        self._likelihood = np.ones(search_map.grid.shape)
        self._claimed: dict[int, list[Block]] = {}
        self._current_rate: dict[int, float] = {}
        self._readings = 0
        self._planned_with: dict[int, int] = {}
        self._home = search_map.grid.centre_m(*search_map.base)
        self._t_s = 0.0
        self._notes: list[dict] = []

    def observe(self, event: Observation) -> None:
        self._advance(event.t_s)
        if isinstance(event, Look):
            self.belief.observe_miss(event.cells, event.pod)
        elif isinstance(event, Hypotheses):
            raw_likelihood = event.likelihood(self._prior)
            rest_w = getattr(event, 'rest_weight', 0.5)
            dynamic_trust = max(0.2, 2.0 * (1.0 - rest_w))
            self._likelihood = raw_likelihood ** dynamic_trust
            self._readings += 1
        elif isinstance(event, DroneLost):
            self._claimed.pop(event.drone, None)

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        self._advance(t_s)
        self._claimed.pop(drone.id, None)
        rate, blocks, route = self._best_tour((drone.y_m, drone.x_m), 0.95 * self.fleet.endurance_s)
        if route:
            self._claim(drone.id, blocks)

            self._current_rate[drone.id] = rate
            
            self._notes.append({
                "kind": "sortie", "drone": drone.id,
                "text": (f"Drone {drone.id + 1}: {len(blocks)}-block tour with "
                         f"{rate * 60:.1%} chance of finding per minute of flight"),
            })
        return at_altitude(route, self.fleet.alt_m)

    def retask(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint] | None:
        if self._planned_with.get(drone.id) == self._readings:
            return None
        self._advance(t_s)
        current = self._claimed.pop(drone.id, None)

        rate, blocks, route = self._best_tour((drone.y_m, drone.x_m), 0.95 * drone.battery_s)
        
        if not route or _same_blocks(blocks, current):
            if current is not None:
                self._claim(drone.id, current)
            return None
            

        old_rate = self._current_rate.get(drone.id, 0.0)
        improvement_threshold = 1.05  
        
        if old_rate > 0 and rate < old_rate * improvement_threshold:
            if current is not None:
                self._claim(drone.id, current)
            return None

        self._claim(drone.id, blocks)
        self._current_rate[drone.id] = rate
        
        self._notes.append({
            "kind": "sortie", "drone": drone.id,
            "text": (f"Drone {drone.id + 1} retasked mid-flight: new reading -> {len(blocks)}-block tour "
                     f"worth {rate * 60:.1%} per minute"),
        })
        return at_altitude(route, self.fleet.alt_m)

    def probability_map(self):
        poa = self.belief.poa * self._likelihood
        return poa * (self.belief.poa.sum() / poa.sum())

    def drain_notes(self) -> list[dict]:
        notes, self._notes = self._notes, []
        return notes

    def _claim(self, drone_id: int, blocks: list[Block]) -> None:
        self._claimed[drone_id] = list(blocks)
        self._planned_with[drone_id] = self._readings

    def _best_tour(self, start: Point, budget_s: float) -> tuple[float, list[Block], list[Point]]:
        """Pack a tour of up to MAX_TOUR adjacent blocks that fits in `budget_s`, pick the best by detect-mass / time."""
        cands = self._candidates()
        if not cands:
            return 0.0, [], []

        by_key = {c.key: c for c in cands}
        # Prefer seeds that are already strong as single-block sorties (amortised base transit).
        ranked = sorted(cands, key=lambda c: self._tour_score(start, [c], budget_s)[0], reverse=True)
        seeds = [c for c in ranked if self._tour_score(start, [c], budget_s)[0] > 0][:SEED_K]
        if not seeds:
            return 0.0, [], []

        best_rate, best_blocks, best_route, best_seconds = 0.0, [], [], 0.0
        for seed in seeds:
            # Pack adjacent high-mass blocks up to MAX_TOUR / battery: fill the sortie first (closes the
            # air-time gap), then compare packed tours by yield per second.
            tour = [seed]
            while len(tour) < MAX_TOUR:
                used = {c.key for c in tour}
                nbrs = [by_key[k] for k in _neighbour_keys(tour, by_key) if k not in used]
                grown = None
                for nxt in nbrs:
                    rate, route, seconds = self._tour_score(start, tour + [nxt], budget_s)
                    if not route:
                        continue
                    # Prefer the extension with the best packed-tour score; tie-break on more flight time.
                    if grown is None or rate > grown[0] or (rate == grown[0] and seconds > grown[3]):
                        grown = (rate, tour + [nxt], route, seconds)
                if grown is None:
                    break
                tour = grown[1]

            rate, route, seconds = self._tour_score(start, tour, budget_s)
            if route and (rate > best_rate or (rate == best_rate and seconds > best_seconds)):
                best_rate, best_blocks, best_route, best_seconds = rate, [c.block for c in tour], route, seconds
        return best_rate, best_blocks, best_route

    def _tour_score(self, start: Point, tour: list[_Cand], budget_s: float) -> tuple[float, list[Point], float]:
        if not tour:
            return 0.0, [], 0.0
        route: list[Point] = []
        for c in tour:
            route.extend(c.route)
        seconds = path_length([start, *route, self._home]) / self.fleet.speed_mps
        if not route or seconds > budget_s:
            return 0.0, [], 0.0
        gain = sum(c.gain for c in tour)
        return gain / seconds, route, seconds

    def _candidates(self) -> list[_Cand]:
        grid = self.map.grid
        lane_m = self.fleet.footprint_m[0] * (1 - self.fleet.overlap)
        side = max(1, round(BLOCK_LANES * lane_m / grid.cell_m))
        value = self.probability_map() * self.pod
        for blocks in self._claimed.values():
            for rows, cols in blocks:
                value[rows, cols] = 0.0
        out: list[_Cand] = []
        for br, r0 in enumerate(range(0, grid.rows, side)):
            for bc, c0 in enumerate(range(0, grid.cols, side)):
                block = (slice(r0, min(r0 + side, grid.rows)), slice(c0, min(c0 + side, grid.cols)))
                gain = float(value[block].sum())
                if gain <= 0.0:
                    continue
                route = flatten(lanes(grid, self.map.no_fly, *block, lane_m))
                if route:
                    out.append(_Cand((br, bc), block, gain, route))
        return out

    def _advance(self, t_s: float) -> None:
        while self._t_s + PREDICT_EVERY_S <= t_s:
            self._t_s += PREDICT_EVERY_S
            self.belief.predict(mobility(self._t_s), self.passability)


def _neighbour_keys(tour: list[_Cand], by_key: dict[BlockKey, _Cand]) -> set[BlockKey]:
    """Block-grid 8-neighbours of any block already in the tour (adjacent chain constraint)."""
    keys: set[BlockKey] = set()
    for c in tour:
        br, bc = c.key
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                k = (br + dr, bc + dc)
                if k in by_key:
                    keys.add(k)
    return keys


def _same_blocks(a: list[Block], b: list[Block] | None) -> bool:
    if b is None or len(a) != len(b):
        return False
    return [(s.start, s.stop) for block in a for s in block] == [(s.start, s.stop) for block in b for s in block]
