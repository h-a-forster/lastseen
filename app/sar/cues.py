"""What witnesses and family tell the search team, generated from the hidden truth.

Real searches run on reports: which way the person set off, how fit they are, a sighting. To test whether a model can
use such reports, each scenario gets a brief at the start and, often, a sighting report during the search. Every cue
is true with a fixed reliability; otherwise it comes from a mistaken witness and points somewhere wrong, so a system
that trusts reports blindly is punished.

These rules and numbers were fixed before any model read the text. Planners and models only ever see `Report` text;
the `Cue` details stay with the arena, for scoring how well the text was understood.
"""

import math
from dataclasses import dataclass

import numpy as np

from app.sar.grid import Terrain
from app.sar.scenario import Scenario
from app.sar.stats import lost_person_stats
from app.sar.walker import RECORD_S

RELIABILITY = {"heading": 0.7, "fitness": 0.8, "sighting": 0.75}
SIGHTING_SHARE = 0.6  # share of scenarios with a sighting report during the search
SIGHTING_AT_S = (1800.0, 5400.0)  # the report arrives 30-90 minutes into the search
SIGHTING_AGE_S = (600.0, 2400.0)  # ...about where the person was 10-40 minutes before that
COMPASS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
_NAMES = {"dementia": ("Margaret", 81), "hiker": ("Daniel", 34)}


@dataclass(frozen=True)
class Report:
    """Text the search team receives at search time `t_s` (0 = the brief)."""

    t_s: float
    text: str


@dataclass(frozen=True)
class Cue:
    """One piece of information inside a report, with the hidden facts behind it."""

    kind: str  # heading | fitness | sighting
    true: bool
    t_s: float
    detail: dict  # what the text claims, in numbers: bearing_deg, distance_km, minutes_ago, fit


def reports_for(scenario: Scenario, seed: int) -> tuple[list[Report], list[Cue]]:
    """The brief and any later reports for a scenario. Fixed by the seed, independent of the planner."""
    rng = np.random.default_rng([seed, 2])
    name, age = _NAMES.get(scenario.map.category, ("Alex", 50))
    heading = _heading_cue(scenario, rng)
    fitness = _fitness_cue(scenario, rng)
    brief = (f"{name}, {age}, {_situation(scenario.map.category)} "
             f"A neighbour saw {name} set off towards the {_compass(heading.detail['bearing_deg'])}. "
             f"Family say {name} {'is fit and walks a lot' if fitness.detail['fit'] else 'walks slowly and tires quickly'}.")
    reports, cues = [Report(0.0, brief)], [heading, fitness]
    if rng.random() < SIGHTING_SHARE:
        sighting = _sighting_cue(scenario, rng)
        d = sighting.detail
        where = ("close to the house" if d["distance_km"] < 0.2
                 else f"roughly {d['distance_km']:.1f} km {_compass(d['bearing_deg'])} of the house")
        reports.append(Report(sighting.t_s, f"A dog walker reports seeing someone matching {name}'s description about "
                                            f"{d['minutes_ago']} minutes ago, {where}, {d['place']}."))
        cues.append(sighting)
    return reports, cues


def _situation(category: str) -> str:
    if category == "dementia":
        return "has dementia and left home on foot this afternoon without telling anyone."
    return "did not come back from a day walk; the car is still parked at the house where the walk started."


def _heading_cue(scenario: Scenario, rng: np.random.Generator) -> Cue:
    true_bearing = _bearing(math.sin(scenario.start_heading), math.cos(scenario.start_heading))
    true = rng.random() < RELIABILITY["heading"]
    bearing = true_bearing if true else (true_bearing + rng.uniform(90, 270)) % 360  # a mistaken witness
    return Cue("heading", true, 0.0, {"bearing_deg": round(bearing / 45) * 45 % 360})


def _fitness_cue(scenario: Scenario, rng: np.random.Generator) -> Cue:
    """'Fit' is true when the person ends up farther away than the median for their category."""
    start = np.array(scenario.map.grid.centre_m(*scenario.map.ipp))
    went_km = float(np.hypot(*(scenario.track[-1] - start))) / 1000
    median_km = lost_person_stats()["categories"][scenario.map.category]["rings_km"][1]
    true = rng.random() < RELIABILITY["fitness"]
    fit = (went_km > median_km) == true
    return Cue("fitness", true, 0.0, {"fit": fit})


def _sighting_cue(scenario: Scenario, rng: np.random.Generator) -> Cue:
    t_s = rng.uniform(*SIGHTING_AT_S)
    age_s = min(rng.uniform(*SIGHTING_AGE_S), t_s)  # the track starts with the search; no earlier record exists
    start = np.array(scenario.map.grid.centre_m(*scenario.map.ipp))
    true = rng.random() < RELIABILITY["sighting"]
    if true:
        y, x = scenario.track[int((t_s - age_s) // RECORD_S)]
    else:  # someone else: a plausible spot 0.3-2 km from the house
        bearing, dist = rng.uniform(0, 2 * np.pi), rng.uniform(300, 2000)
        y, x = start[0] - dist * math.cos(bearing), start[1] + dist * math.sin(bearing)
    dy, dx = y - start[0], x - start[1]
    detail = {"bearing_deg": round(_bearing(dy, dx) / 45) * 45 % 360, "distance_km": round(math.hypot(dy, dx) / 1000, 1),
              "minutes_ago": int(round(age_s / 300) * 5), "place": _place(scenario, y, x)}
    return Cue("sighting", true, t_s, detail)


def _place(scenario: Scenario, y: float, x: float) -> str:
    """How a witness would describe the spot."""
    m = scenario.map
    if not (0 <= y < m.grid.rows * m.grid.cell_m and 0 <= x < m.grid.cols * m.grid.cell_m):
        return "out in the countryside"
    r, c = m.grid.cell_of(y, x)
    near = (slice(max(r - 2, 0), r + 3), slice(max(c - 2, 0), c + 3))
    if m.trails[near].any():
        return "on a footpath"
    if (m.terrain[near] == Terrain.WATER).any():
        return "near a stream"
    return {Terrain.FOREST: "in the woods", Terrain.SCRUB: "in rough scrubland"}.get(Terrain(m.terrain[r, c]), "in open fields")


def _bearing(dy: float, dx: float) -> float:
    """Compass bearing in degrees (0 = north, clockwise) of a map offset; rows grow southwards."""
    return math.degrees(math.atan2(dx, -dy)) % 360


def _compass(bearing_deg: float) -> str:
    return COMPASS[int(round(bearing_deg / 45)) % 8]
