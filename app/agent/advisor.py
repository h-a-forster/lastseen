"""Put Muse Spark between the witness reports and any planner.

`SparkAdvisor(planner)` is itself a planner. It passes everything through, and when a report arrives it asks Spark to
read all reports so far and forwards the resulting `Hypotheses` event, so the inner planner never calls a model and
stays deterministic. If Spark fails (network, spend cap, cache miss in replay mode), the search carries on without
the hypotheses and the failure is recorded: the drones never wait on the model.
"""

from app.agent.interpret import interpret
from app.meta_api import ModelCallError
from app.sar.detection import DetectionModel
from app.sar.interface import DroneState, FleetSpec, Observation, Planner, Report, Waypoint
from app.sar.prior import ring_prior
from app.sar.scenario import SearchMap


def base_map(search_map: SearchMap, stats: dict):
    """The statistics-only map the hypotheses are drawn on (distance rings weighted by ease of walking)."""
    rings = stats["categories"][search_map.category]["rings_km"]
    return ring_prior(search_map.grid, search_map.ipp, rings, weight=search_map.passability)[0]


def read_all(search_map: SearchMap, reports: list[Report], stats: dict) -> list:
    """Spark's reading after each report, as the advisor would request them (same requests, so they share the cache)."""
    base = base_map(search_map, stats)
    return [interpret(search_map, reports[:k], base) for k in range(1, len(reports) + 1)]


class SparkAdvisor:
    def __init__(self, inner: Planner) -> None:
        self.inner = inner
        self.name = f"{inner.name}+spark"
        self.readings: list = []  # every HypothesisSet received, for the log and the UI
        self.failures: list[str] = []
        self._notes: list[dict] = []

    def reset(self, search_map: SearchMap, fleet: FleetSpec, detection: DetectionModel, stats: dict, seed: int) -> None:
        self.map = search_map
        self.base_poa = base_map(search_map, stats)
        self.reports: list[Report] = []
        self.readings, self.failures, self._notes = [], [], []
        self.inner.reset(search_map, fleet, detection, stats, seed)

    def observe(self, event: Observation) -> None:
        self.inner.observe(event)
        if isinstance(event, Report):
            self.reports.append(event)
            try:
                reading = interpret(self.map, self.reports, self.base_poa)
            except ModelCallError as e:
                self.failures.append(f"{event.t_s:.0f} s: {e}")
                self._notes.append({"kind": "spark_failed", "text": "Spark unavailable; searching on statistics alone"})
                return
            self.readings.append(reading)
            self._notes.append({"kind": "hypotheses", "rest_weight": round(reading.rest_weight, 2),
                                "hypotheses": [{"name": h.name, "weight": round(h.weight, 2), "reason": h.reason}
                                               for h in reading.hypotheses],
                                "dropped": list(reading.dropped), "cached": reading.cached})
            self.inner.observe(reading.to_event())

    def plan(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint]:
        return self.inner.plan(t_s, drone, fleet)

    def retask(self, t_s: float, drone: DroneState, fleet: tuple[DroneState, ...]) -> list[Waypoint] | None:
        retask = getattr(self.inner, "retask", None)
        return retask(t_s, drone, fleet) if retask else None

    def drain_notes(self) -> list[dict]:
        """Notes for the event log: Spark's readings, then whatever the inner planner explains."""
        notes, self._notes = self._notes, []
        inner = getattr(self.inner, "drain_notes", None)
        return notes + (inner() if inner else [])

    def probability_map(self):
        return self.inner.probability_map()
