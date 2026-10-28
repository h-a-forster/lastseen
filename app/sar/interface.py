"""The contract between the arena and a planner plug-in.

A planner keeps its own probability map and learns only what a real search team would: the map, the fleet, the
detection rates and published statistics at the start, then one event per photo that found nobody. The arena never
passes the truth, and keeps the safety rules (forced return on low battery) to itself.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.sar.detection import DetectionModel
from app.sar.grid import Camera
from app.sar.scenario import SearchMap

Waypoint = tuple[float, float, float]  # (y_m, x_m, alt_m above ground)
MIN_ALT_M, MAX_ALT_M = 40.0, 120.0


@dataclass(frozen=True)
class FleetSpec:
    drones: int = 3
    speed_mps: float = 10.0
    alt_m: float = 60.0  # default altitude; low enough that a person is 20-65 px for the detector
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


@dataclass(frozen=True)
class DroneState:
    id: int
    y_m: float
    x_m: float
    alt_m: float
    battery_s: float
    airborne: bool
    route: tuple[Waypoint, ...]  # what is left of its current sortie


@dataclass(frozen=True)
class Look:
    """A photo that showed nobody. `pod` covers `cells` and is 0 where this sortie had already photographed a cell:
    overlapping photos are not independent looks."""

    t_s: float
    drone: int
    y_m: float
    x_m: float
    alt_m: float
    cells: tuple[slice, slice]
    pod: np.ndarray  # read-only


# Grows as the arena learns more: Candidate, Hypotheses, Clue, Zone, Constraint, Role, DroneLost, CommsSync.
Observation = Look


class Planner(Protocol):
    name: str

    def reset(self, search_map: SearchMap, fleet: FleetSpec, detection: DetectionModel, stats: dict, seed: int) -> None:
        """A new search. Use only `seed` for randomness, so runs replay exactly."""

    def observe(self, event: Observation) -> None:
        """Something the search team learned, delivered in time order."""

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        """The next sortie for `drone`, which is at base. The arena adds the flight home. An empty list keeps it on the
        ground for a minute. Should take under ~50 ms on a 160x160 grid."""

    def probability_map(self) -> np.ndarray | None:
        """The planner's current probability of area per cell, for display and map scoring. Optional."""
