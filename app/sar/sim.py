"""Time-stepped search simulation: drones fly planner routes, take nadir photos, and the map updates after every look.

Fast mode (here) samples each detection from the calibrated POD table, so thousands of runs cost no model calls.
Imagery mode swaps the sampler for real aerial photos run through the perception pipeline.
"""

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.sar.belief import Belief
from app.sar.detection import DetectionModel
from app.sar.grid import Camera
from app.sar.prior import ring_prior
from app.sar.scenario import Scenario, SearchMap

PREDICT_EVERY_S = 60.0


@dataclass(frozen=True)
class FleetSpec:
    drones: int = 3
    speed_mps: float = 10.0
    alt_m: float = 60.0  # above ground; low enough that a person is 20-65 px for the detector
    endurance_s: float = 25 * 60  # usable flight time per battery, landing reserve already removed
    swap_s: float = 180.0  # battery swap and relaunch at base
    overlap: float = 0.1  # along-track photo overlap
    camera: Camera = Camera()

    @property
    def footprint_m(self) -> tuple[float, float]:
        return self.camera.footprint_m(self.alt_m)

    @property
    def gsd_cm(self) -> float:
        return self.camera.gsd_cm(self.alt_m)


@dataclass
class Drone:
    id: int
    y: float
    x: float
    battery_s: float
    route: list[tuple[float, float]] = field(default_factory=list)
    returning: bool = False
    ready_at_s: float = 0.0
    since_photo_m: float = math.inf
    seen: np.ndarray | None = None  # cells already photographed this sortie; overlap is not a second independent look


@dataclass(frozen=True)
class View:
    """What a planner sees when a drone at base asks for its next sortie."""

    t_s: float
    map: SearchMap
    belief: Belief
    fleet: FleetSpec
    detection: DetectionModel
    drones: tuple[Drone, ...]


class Planner(Protocol):
    name: str

    def plan(self, view: View, drone: Drone) -> list[tuple[float, float]]:
        """Waypoints (y, x in metres) for one sortie. The sim adds the flight home."""


@dataclass
class RunResult:
    planner: str
    scenario: str
    found_s: float | None
    cumulative_pos: list[tuple[float, float]]  # (t_s, P(found by t) under the planner's own map)
    photos: int
    flight_s: float
    forced_returns: int  # battery safety overrides: the planner sent a drone too far
    no_fly_s: float  # drone-seconds spent over no-fly cells; must be 0
    calibrated: bool


def mobility(t_s: float, m0: float = 0.3, tau_s: float = 2 * 3600) -> float:
    """Fraction of probability mass that moves to a neighbouring cell per minute; decays as the subject tires or shelters."""
    return m0 * math.exp(-t_s / tau_s)


class Sim:
    def __init__(self, scenario: Scenario, planner: Planner, fleet: FleetSpec = FleetSpec(),
                 detection: DetectionModel = DetectionModel(), seed: int = 0, max_s: float = 4 * 3600, dt_s: float = 5.0):
        self.scenario, self.planner, self.fleet, self.detection = scenario, planner, fleet, detection
        self.map = scenario.map
        self.rng = np.random.default_rng(seed)
        self.max_s, self.dt_s = max_s, dt_s
        poa, row = ring_prior(self.map.grid, self.map.ipp, self.map.rings_km, weight=self.map.passability)
        self.belief = Belief(poa, row)
        self.subject = scenario.subject_cell
        by, bx = self.map.grid.centre_m(*self.map.base)
        self.base_m = (by, bx)
        self.drones = [Drone(i, by, bx, fleet.endurance_s) for i in range(fleet.drones)]
        self.pod_map = detection.pod(fleet.gsd_cm, self.map.terrain)
        self.passability = self.map.passability

    def run(self) -> RunResult:
        t, found_s, photos, flight_s, forced, no_fly_s = 0.0, None, 0, 0.0, 0, 0.0
        miss_product = 1.0
        curve = [(0.0, 0.0)]
        next_predict = PREDICT_EVERY_S
        while t < self.max_s and found_s is None:
            for d in self.drones:
                if self._at_base(d) and not d.route and t >= d.ready_at_s:
                    d.route = list(self.planner.plan(self._view(t), d))
                    d.returning = False
                    d.seen = np.zeros(self.map.grid.shape, bool)
                if not d.route:
                    continue
                flight_s += self.dt_s
                d.battery_s -= self.dt_s
                if not d.returning and d.battery_s < self._time_home(d) + 2 * self.dt_s:
                    d.route, d.returning = [self.base_m], True
                    forced += 1
                for photo_at in self._move(d):
                    photos += 1
                    hit, pos = self._look(d, *photo_at)
                    miss_product *= 1 - pos
                    if hit:
                        found_s = t
                        break
                if self.map.no_fly[self.map.grid.cell_of(d.y, d.x)]:
                    no_fly_s += self.dt_s
                if not d.route and not d.returning:
                    d.route, d.returning = [self.base_m], True
                elif not d.route and d.returning:
                    d.battery_s, d.ready_at_s = self.fleet.endurance_s, t + self.fleet.swap_s
                if found_s is not None:
                    break
            if t >= next_predict:
                self._time_passes(t)
                next_predict += PREDICT_EVERY_S
                curve.append((t, 1 - miss_product))
            t += self.dt_s
        return RunResult(self.planner.name, self.scenario.name, found_s, curve, photos, flight_s, forced, no_fly_s,
                         self.detection.calibrated)

    def _view(self, t: float) -> View:
        """A snapshot planners can't alter: the probability map is a read-only copy."""
        poa = self.belief.poa.copy()
        poa.flags.writeable = False
        return View(t, self.map, Belief(poa, self.belief.row), self.fleet, self.detection, tuple(self.drones))

    def _at_base(self, d: Drone) -> bool:
        return math.hypot(d.y - self.base_m[0], d.x - self.base_m[1]) < 1.0

    def _time_home(self, d: Drone) -> float:
        return math.hypot(d.y - self.base_m[0], d.x - self.base_m[1]) / self.fleet.speed_mps

    def _move(self, d: Drone) -> list[tuple[float, float]]:
        """Advance along the route for one step; return where photos were taken (none on the way home)."""
        budget = self.fleet.speed_mps * self.dt_s
        spacing = self.fleet.footprint_m[1] * (1 - self.fleet.overlap)
        photos = []
        while budget > 0 and d.route:
            ty, tx = d.route[0]
            dist = math.hypot(ty - d.y, tx - d.x)
            step = min(dist, budget, max(spacing - d.since_photo_m, 0.0) if not d.returning else budget)
            if dist > 0:
                d.y += (ty - d.y) * step / dist
                d.x += (tx - d.x) * step / dist
            budget -= step
            d.since_photo_m += step
            if not d.returning and d.since_photo_m >= spacing:
                photos.append((d.y, d.x))
                d.since_photo_m = 0.0
            if step >= dist:
                d.route.pop(0)
        return photos

    def _look(self, d: Drone, y: float, x: float) -> tuple[bool, float]:
        """One nadir photo: did it reveal the subject? If not, update the map. Returns (hit, POS of the look)."""
        cells = self.map.grid.footprint(y, x, *self.fleet.footprint_m)
        pod = np.where(d.seen[cells], 0.0, self.pod_map[cells])
        d.seen[cells] = True
        if self.subject is not None:
            r, c = self.subject
            if cells[0].start <= r < cells[0].stop and cells[1].start <= c < cells[1].stop:
                if self.rng.random() < pod[r - cells[0].start, c - cells[1].start]:
                    return True, float((self.belief.poa[cells] * pod).sum())
        return False, self.belief.observe_miss(cells, pod)

    def _time_passes(self, t: float) -> None:
        m = mobility(t)
        self.belief.predict(m, self.passability)
        if self.subject is not None and self.rng.random() < m:
            self.subject = self._wander(self.subject)

    def _wander(self, cell: tuple[int, int]) -> tuple[int, int] | None:
        r, c = cell
        options = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        rows, cols = self.map.grid.shape
        weights = np.array([self.passability[o] if 0 <= o[0] < rows and 0 <= o[1] < cols else 1.0 for o in options])
        nr, nc = options[self.rng.choice(4, p=weights / weights.sum())]
        return (nr, nc) if 0 <= nr < rows and 0 <= nc < cols else None
