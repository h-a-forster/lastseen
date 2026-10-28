"""The team's decision system: prior, belief updates and multi-drone tasking.

It plugs into the arena through `app.sar.interface.Planner`. Benchmark it against the baselines with
`uv run python -m app.sar.bench --planners ours,spiral,greedy`; `ours+spark` adds Spark's reading of the witness
reports. Until it is written, this keeps the drones on the ground. `app/sar/planners/greedy.py` is a complete example.
"""

from app.sar.detection import DetectionModel
from app.sar.interface import DroneState, FleetSpec, Observation, Waypoint
from app.sar.scenario import SearchMap


class SearchPlanner:
    name = "ours"

    def reset(self, search_map: SearchMap, fleet: FleetSpec, detection: DetectionModel, stats: dict, seed: int) -> None:
        pass

    def observe(self, event: Observation) -> None:
        pass

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        return []

    def probability_map(self):
        return None
