"""The arena: a time-stepped search simulation. Drones fly a planner's routes and take nadir photos, the hidden subject
moves along its precomputed track, and the arena decides what each photo reveals. The planner hears only about photos
that found nobody.

Fast mode samples each detection from the POD table, so thousands of runs cost no model calls. Imagery mode
(`photos=PhotoPool`) decides each photo of the person with a real recorded answer of SAM 3.1 + Muse Spark on a real
person of the same pixel size (`app.perception.pool`), also free. In both modes each photo of empty ground raises
false alarms at the measured rate; a reviewer has to reject them, which is what the alarm count measures.

A drone can be lost during the search (`losses`, see `drone_losses`): it stops where it is, keeps the photos it took,
and the planner is told what it had left to fly.
"""

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np

from app.sar.detection import DetectionModel
from app.sar.grid import Terrain
from app.sar.cues import Report
from app.sar.interface import MAX_ALT_M, MIN_ALT_M, DroneLost, DroneState, FleetSpec, Look, Planner, Waypoint
from app.sar.recorder import Recorder
from app.sar.routing import distance_field, path_length, safe_path
from app.sar.scenario import Scenario
from app.sar.stats import lost_person_stats

IDLE_RETRY_S = 60.0
DETOUR_MARGIN_S = 120.0  # below this much spare battery, the time home is measured along the detour
LOSS_WINDOW_S = (15 * 60.0, 2 * 3600.0)  # when a drone fails, if one does: within the first two hours


@dataclass
class Drone:
    id: int
    y: float
    x: float
    alt_m: float
    battery_s: float
    route: list[Waypoint] = field(default_factory=list)
    returning: bool = False
    ready_at_s: float = 0.0
    since_photo_m: float = math.inf
    east_west: bool = False
    seen: np.ndarray | None = None  # cells already photographed this sortie
    lost: bool = False

    def state(self, airborne: bool) -> DroneState:
        return DroneState(self.id, self.y, self.x, self.alt_m, self.battery_s, airborne, tuple(self.route))


@dataclass
class RunResult:
    planner: str
    scenario: str
    found_s: float | None
    photos: int
    flight_s: float
    forced_returns: int  # battery safety overrides: the planner sent a drone too far
    no_fly_s: float  # drone-seconds over no-fly cells: 0, bar legs that clip a corner by under 1/20 of a cell
    plan_ms_max: float
    calibrated: bool
    alarms: int = 0  # false alarms raised on empty ground, each one for a reviewer to reject
    in_frame: int = 0  # fresh photos that had the person in them (a miss among these is the detector's)
    drone_s: float = 0.0  # drone-seconds of working (not lost) drones; flight_s / drone_s is the share spent flying


def drone_losses(seed: int, drones: int) -> dict[int, float]:
    """One drone, chosen by the seed, fails at a time chosen by the seed: the same failure for every planner."""
    rng = np.random.default_rng([seed, 5])
    return {int(rng.integers(drones)): float(rng.uniform(*LOSS_WINDOW_S))}


def planner_seed(seed: int) -> int:
    """The seed a planner gets: derived from the scenario's, so it can't regenerate the scenario and its truth."""
    return int(np.random.SeedSequence([seed, 3]).generate_state(1)[0])


class Sim:
    def __init__(self, scenario: Scenario, planner: Planner, fleet: FleetSpec = FleetSpec(),
                 detection: DetectionModel | None = None, seed: int = 0, max_s: float = 4 * 3600, dt_s: float = 5.0,
                 reports: list[Report] | tuple[Report, ...] = (), recorder: Recorder | None = None, photos=None,
                 losses: dict[int, float] | None = None):
        if any(not 0 <= d < fleet.drones for d in losses or {}):
            raise ValueError(f"lost drones {sorted(losses)} are not all in a fleet of {fleet.drones}")
        detection = detection or DetectionModel.load()
        self.scenario, self.planner, self.fleet, self.detection = scenario, planner, fleet, detection
        self.rec = recorder  # None in benchmarks: recording costs time
        self.photos = photos  # a PhotoPool for imagery mode; None for fast mode
        self.pending_reports = sorted(reports, key=lambda r: r.t_s)
        self.pending_losses = sorted((t_s, d) for d, t_s in (losses or {}).items())
        self.map = scenario.map
        self.detect_rng = np.random.default_rng([seed, 1])  # its own stream, apart from the scenario's
        self.alarm_rng = np.random.default_rng([seed, 4])  # and false alarms another, so they never shift detections
        self.alarms_per_photo = detection.alarms_per_tile * fleet.camera.tiles_per_photo()
        self.alarms, self.in_frame = 0, 0
        self.max_s, self.dt_s = max_s, dt_s
        self.base_m = self.map.grid.centre_m(*self.map.base)
        self.drones = [Drone(i, *self.base_m, fleet.alt_m, fleet.endurance_s) for i in range(fleet.drones)]
        self._pod_at: dict[float, np.ndarray] = {}
        self._home_m = distance_field(self.map.grid, self.map.no_fly, self.map.base)
        for layer in (self.map.terrain, self.map.trails, self.map.no_fly):
            layer.flags.writeable = False  # planners read the map; they can't change it for the arena or the next run
        planner.reset(self.map, fleet, detection, copy.deepcopy(lost_person_stats()), planner_seed(seed))

    def run(self) -> RunResult:
        t, found_s, photos, flight_s, forced, no_fly_s, plan_ms_max, drone_s = 0.0, None, 0, 0.0, 0, 0.0, 0.0, 0.0
        while t < self.max_s and found_s is None:
            while self.pending_reports and self.pending_reports[0].t_s <= t:
                report = self.pending_reports.pop(0)
                if self.rec:
                    self.rec.add("report", t, text=report.text)
                self.planner.observe(report)
            while self.pending_losses and self.pending_losses[0][0] <= t:
                self._lose(self.drones[self.pending_losses.pop(0)[1]], t)
            for d in self.drones:
                if d.lost:
                    continue
                drone_s += self.dt_s
                if self._at_base(d) and not d.route and t >= d.ready_at_s:
                    plan_ms_max = max(plan_ms_max, self._launch(d, t))
                if not d.route:
                    continue
                flight_s += self.dt_s
                d.battery_s -= self.dt_s
                if not d.returning and d.battery_s < self._time_home(d) + 2 * self.dt_s:
                    self._send_home(d)
                    forced += 1
                    if self.rec:
                        self.rec.add("forced_return", t, drone=d.id)
                for photo_at in self._move(d):
                    photos += 1
                    if self._look(d, t, *photo_at):
                        found_s = t
                        if self.rec:
                            self.rec.add("found", t, drone=d.id, y_m=round(d.y, 1), x_m=round(d.x, 1))
                        break
                if self.map.no_fly[self.map.grid.cell_of(d.y, d.x)]:
                    no_fly_s += self.dt_s
                if not d.route and not d.returning:
                    self._send_home(d)
                elif not d.route and d.returning:
                    d.battery_s, d.ready_at_s = self.fleet.endurance_s, t + self.fleet.swap_s
                if found_s is not None:
                    break
            if self.rec:
                self.rec.tick(t, self.drones, self.planner)
            t += self.dt_s
        return RunResult(self.planner.name, self.scenario.name, found_s, photos, flight_s, forced, no_fly_s,
                         plan_ms_max, self.detection.calibrated, self.alarms, self.in_frame, drone_s)

    def _lose(self, d: Drone, t: float) -> None:
        unflown = () if d.returning else tuple(d.route)
        d.route, d.lost = [], True
        if self.rec:
            self.rec.add("drone_lost", t, drone=d.id, y_m=round(d.y, 1), x_m=round(d.x, 1))
        self.planner.observe(DroneLost(t, d.id, unflown))

    def _launch(self, d: Drone, t: float) -> float:
        """Ask the planner for a sortie; returns how long it took in ms. The fleet it sees leaves out lost drones."""
        fleet = tuple(o.state(airborne=bool(o.route)) for o in self.drones if not o.lost)
        started = time.perf_counter()
        route = list(self.planner.plan(t, d.state(airborne=False), fleet))
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._check(route)
        if route:
            d.route, d.returning = self._safe((d.y, d.x), route), False
            d.seen = np.zeros(self.map.grid.shape, bool)
        else:
            d.ready_at_s = t + IDLE_RETRY_S
        return elapsed_ms

    def _check(self, route: list[Waypoint]) -> None:
        height, width = self.map.grid.rows * self.map.grid.cell_m, self.map.grid.cols * self.map.grid.cell_m
        for y, x, alt in route:
            if not (0 <= y <= height and 0 <= x <= width):
                raise ValueError(f"{self.planner.name}: waypoint ({y:.0f}, {x:.0f}) is outside the search area")
            if self.map.no_fly[self.map.grid.cell_of(y, x)]:
                raise ValueError(f"{self.planner.name}: waypoint ({y:.0f}, {x:.0f}) is in a no-fly area")
            if not MIN_ALT_M <= alt <= MAX_ALT_M:
                raise ValueError(f"{self.planner.name}: altitude {alt} m is outside {MIN_ALT_M}-{MAX_ALT_M} m")

    def _send_home(self, d: Drone) -> None:
        d.route, d.returning = self._safe((d.y, d.x), [(*self.base_m, d.alt_m)]), True

    def _safe(self, start: tuple[float, float], route: list[Waypoint]) -> list[Waypoint]:
        """The autopilot's version of a route: legs that would cross a no-fly area become detours around it, flown
        at the altitude of the leg they replace."""
        out, here = [], start
        for y, x, alt in route:
            out.extend((py, px, alt) for py, px in safe_path([here, (y, x)], self.map.no_fly, self.map.grid)[1:])
            here = (y, x)
        return out

    def _at_base(self, d: Drone) -> bool:
        return math.hypot(d.y - self.base_m[0], d.x - self.base_m[1]) < 1.0

    def _time_home(self, d: Drone) -> float:
        """Flying time home around no-fly areas. Far from the battery limit an upper bound is enough (the grid
        distance plus two cells for the ends of the route); near it, the autopilot's actual detour is measured."""
        bound = (self._home_m[self.map.grid.cell_of(d.y, d.x)] + 2 * self.map.grid.cell_m) / self.fleet.speed_mps
        if d.battery_s > bound + DETOUR_MARGIN_S:
            return bound
        return path_length(safe_path([(d.y, d.x), self.base_m], self.map.no_fly, self.map.grid)) / self.fleet.speed_mps

    def _move(self, d: Drone) -> list[tuple[float, float, bool]]:
        """Advance along the route for one step; return where photos were taken (none on the way home), each with
        whether the drone was heading mostly east-west."""
        budget = self.fleet.speed_mps * self.dt_s
        photos = []
        while budget > 0 and d.route:
            ty, tx, d.alt_m = d.route[0]
            spacing = self.fleet.camera.footprint_m(d.alt_m)[1] * (1 - self.fleet.overlap)
            dist = math.hypot(ty - d.y, tx - d.x)
            step = min(dist, budget) if d.returning else min(dist, budget, max(spacing - d.since_photo_m, 0.0))
            if dist > 0:
                d.east_west = abs(tx - d.x) > abs(ty - d.y)
                d.y += (ty - d.y) * step / dist
                d.x += (tx - d.x) * step / dist
            budget -= step
            d.since_photo_m += step
            if not d.returning and d.since_photo_m >= spacing:
                photos.append((d.y, d.x, d.east_west))
                d.since_photo_m = 0.0
            if step >= dist:
                d.route.pop(0)
        return photos

    def _look(self, d: Drone, t: float, y: float, x: float, east_west: bool) -> bool:
        """One nadir photo. True if it revealed the subject; otherwise the planner is told about the miss.
        The camera's wide side is across the direction of flight; footprints snap to the nearer grid axis."""
        across, along = self.fleet.camera.footprint_m(d.alt_m)
        cells = self.map.grid.footprint(y, x, *((along, across) if east_west else (across, along)))
        pod = np.where(d.seen[cells], 0.0, self._pod(d.alt_m)[cells])
        d.seen[cells] = True
        alarms = int(self.alarm_rng.poisson(self.alarms_per_photo)) if self.alarms_per_photo > 0 else 0
        self.alarms += alarms
        if self.rec:
            self.rec.add("photo", t, drone=d.id, cells=[cells[0].start, cells[0].stop, cells[1].start, cells[1].stop],
                         **({"alarms": alarms} if alarms else {}))
        subject = self.scenario.subject_at(t)
        if subject is not None:
            r, c = subject
            if cells[0].start <= r < cells[0].stop and cells[1].start <= c < cells[1].stop:
                p = pod[r - cells[0].start, c - cells[1].start]
                self.in_frame += bool(p > 0)
                if self._detect(d, t, r, c, p):
                    return True
        pod.flags.writeable = False
        self.planner.observe(Look(t, d.id, y, x, d.alt_m, cells, pod))
        return False

    def _detect(self, d: Drone, t: float, r: int, c: int, pod: float) -> bool:
        """Whether this photo of the person finds them: a roll against the table (fast mode), or the pipeline's
        recorded answer on a real person of the same size, with canopy hiding them at the table's forest rate."""
        if self.photos is None:
            return self.detect_rng.random() < pod  # rolled even when pod is 0, so the stream matches earlier results
        if pod <= 0:  # this sortie already photographed this spot: overlap isn't a second look
            return False
        gsd = self.fleet.camera.gsd_cm(d.alt_m)
        shot = self.photos.shot(gsd, self.detect_rng)
        factor = self.detection.terrain_factor(gsd, Terrain(int(self.map.terrain[r, c])))
        hidden = factor < 1 and self.detect_rng.random() >= factor
        found = shot.detected and not hidden
        if self.rec:
            self.rec.add("sighting", t, drone=d.id, tile=shot.tile, box=[round(v, 1) for v in shot.box],
                         size_px=round(shot.size_px), detected=shot.detected, sam=shot.sam, spark=shot.spark,
                         canopy=hidden, found=found)
        return found

    def _pod(self, alt_m: float) -> np.ndarray:
        if alt_m not in self._pod_at:
            self._pod_at[alt_m] = self.detection.pod(self.fleet.camera.gsd_cm(alt_m), self.map.terrain)
        return self._pod_at[alt_m]
