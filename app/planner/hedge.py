"""The team's planner: the expanding square, until a witness sighting sends one drone after it.

Every drone flies the expanding square from the house, the strongest fixed pattern with three drones, until a
reading of a report made after the brief arrives (a sighting). Then one drone breaks off at once and hunts with the
map planner (`ours`), which folds Spark's reading into its probability map; the others keep sweeping. A wrong
sighting costs one drone's share of the sweep; a right one sends a drone where the person was seen, often an hour or
more before the pattern would get there.

The brief alone redirects no one: the pattern already starts where the statistics say, and on development seeds a
hunter launched on the brief found fewer people within the first hour than one held back for a sighting.
"""

import dataclasses

from app.planner import SearchPlanner
from app.sar.detection import DetectionModel
from app.sar.interface import DroneLost, DroneState, FleetSpec, Hypotheses, Observation, Waypoint
from app.sar.planners.fixed_paths import ExpandingSquare
from app.sar.scenario import SearchMap


class Hedge:
    name = "hedge"

    def reset(self, search_map: SearchMap, fleet: FleetSpec, detection: DetectionModel, stats: dict, seed: int) -> None:
        self.sweep, self.hunt = ExpandingSquare(), SearchPlanner()
        for planner in (self.sweep, self.hunt):
            planner.reset(search_map, fleet, detection, stats, seed)
        self._alive = list(range(fleet.drones))
        self._on_pattern: set[int] = set()  # drones whose current sortie is a piece of the pattern
        self._hunting = False
        self._notes: list[dict] = []

    @property
    def hunter(self) -> int | None:
        """The drone that hunts once a sighting is in: the last one still flying."""
        return self._alive[-1] if self._hunting and self._alive else None

    def observe(self, event: Observation) -> None:
        if isinstance(event, Hypotheses):
            self.hunt.observe(event)  # the pattern stays on the house; only the hunter follows readings
            if event.t_s > 0 and not self._hunting:
                self._hunting = True
                if self.hunter is not None:
                    others = "; the others keep sweeping" if len(self._alive) > 1 else ""
                    self._notes.append({"kind": "hunt", "drone": self.hunter,
                                        "text": f"Drone {self.hunter + 1} leaves the pattern to search where the new "
                                                f"report points{others}"})
            return
        if isinstance(event, DroneLost):
            on_pattern = event.drone in self._on_pattern
            self.sweep.observe(event if on_pattern else dataclasses.replace(event, unflown=()))
            self._alive.remove(event.drone)
            self._on_pattern.discard(event.drone)
        else:
            self.sweep.observe(event)
        self.hunt.observe(event)

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        if drone.id == self.hunter:
            self._on_pattern.discard(drone.id)
            return self.hunt.plan(t_s, drone, fleet)
        self._on_pattern.add(drone.id)
        return self.sweep.plan(t_s, drone, fleet)

    def retask(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint] | None:
        if drone.id != self.hunter:
            return None
        route = self.hunt.retask(t_s, drone, fleet)
        if route is not None:
            self._on_pattern.discard(drone.id)
        return route

    def probability_map(self):
        return self.hunt.probability_map()

    def drain_notes(self) -> list[dict]:
        notes, self._notes = self._notes, []
        return notes + self.sweep.drain_notes() + self.hunt.drain_notes()
