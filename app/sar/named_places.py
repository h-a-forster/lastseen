"""Named-place test: witnesses who say where they saw her by a place's name, not by distance and direction.

The simulator's sighting report says "roughly 1.2 km north-east of the house", which rules written for that template
read as well as Muse Spark does (docs/evaluation.md, Known limits). Real witnesses say "by the chapel" instead. On the
Saint-Ursanne map, whose named places come from OpenStreetMap and are given to Spark with the map, this test rewrites
each sighting that happened near a named place so that it names the place instead, and nothing else.

Fixed before any reading was paid for (dev seeds only; the held-out set stays unseen):
- Scenarios: Saint-Ursanne, dementia, one drone, photo mode, 4 h, dev seeds 0-999. Kept: those whose sighting spot
  (right or mistaken) is at least 0.3 km from the house and within 400 m of a named place at least 0.2 km from it.
- Primary: `hedge+spark` with named places against the expanding square alone, found within 2 h; passes if the 95%
  interval is above 0.
- Secondary: `hedge+spark` with named places against the same scenarios worded as the template, within 5 points.
- Reported alongside: `spiral+witness` (rules for the template) and `spiral+named`, a volunteer who knows every named
  place on the map and restarts the square there.

    uv run python -m app.sar.named_places --prefetch --usd-cap 4   # Spark's readings, live and cached
    MODEL_MODE=replay uv run python -m app.sar.named_places        # the comparison from the cache, $0
"""

import argparse
import dataclasses
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from app.agent.layers import Shape, layer
from app.meta_api import spent_usd
from app.sar import bench, cues, metrics
from app.sar.interface import FleetSpec, Hypotheses, Observation, Report
from app.sar.planners.fixed_paths import ExpandingSquare
from app.sar.scenario import Scenario

PLACE = "saint-ursanne"
SEEDS = range(1000)
SPOT_MIN_KM = 0.3  # closer than this, the template already says "close to the house"
PLACE_MIN_KM = 0.2  # the old town's gates, bridge and church are closer than this: naming one says "by the house"
NEAR_M = 400.0  # a witness names a place only if the spot was this close to it
HOURS = 4.0
NAMED_PLANNERS = ["spiral", "spiral+witness", "spiral+named", "hedge+spark"]
TEMPLATE_PLANNERS = ["spiral+witness", "hedge+spark"]
RESULTS = Path(__file__).resolve().parents[2] / "docs" / "results" / "named-places.json"


@dataclasses.dataclass(frozen=True)
class NamedSighting:
    seed: int
    t_s: float
    place: str  # the landmark's name as on the map, e.g. "Chapelle Sainte-Anne"
    kind: str  # chapel, hamlet, bridge, ...
    spot_to_place_m: float


def sighting_spot(scenario: Scenario, seed: int) -> tuple[float, float] | None:
    """Where the witness in `cues.reports_for` really saw someone (her, or the mistaken someone else), in map metres:
    the same draws in the same order, checked against the report's own numbers."""
    rng = np.random.default_rng([seed, 2])
    cues._heading_cue(scenario, rng)
    cues._fitness_cue(scenario, rng)
    if rng.random() >= cues.SIGHTING_SHARE:
        return None
    t_s = rng.uniform(*cues.SIGHTING_AT_S)
    age_s = min(rng.uniform(*cues.SIGHTING_AGE_S), t_s)
    if rng.random() < cues.RELIABILITY["sighting"]:
        y, x = scenario.track[int((t_s - age_s) // cues.RECORD_S)]
    else:
        start = scenario.map.grid.centre_m(*scenario.map.ipp)
        bearing, dist = rng.uniform(0, 2 * np.pi), rng.uniform(300, 2000)
        y, x = start[0] - dist * math.cos(bearing), start[1] + dist * math.sin(bearing)
    detail = next(c.detail for c in cues.reports_for(scenario, seed)[1] if c.kind == "sighting")
    if detail != cues._sighting_detail(scenario, y, x, age_s):
        raise RuntimeError(f"seed {seed}: replayed sighting draws don't match cues.reports_for")
    return float(y), float(x)


def named_sighting(scenario: Scenario, seed: int) -> NamedSighting | None:
    """The named place a witness would give for this scenario's sighting, if it qualifies (module docstring)."""
    spot = sighting_spot(scenario, seed)
    if spot is None:
        return None
    m = scenario.map
    home = m.grid.centre_m(*m.ipp)
    if math.dist(spot, home) < SPOT_MIN_KM * 1000:
        return None
    far = [(label, y, x) for label, y, x in m.landmarks if math.dist((y, x), home) >= PLACE_MIN_KM * 1000]
    label, y, x = min(far, key=lambda lm: math.dist((lm[1], lm[2]), spot))
    if math.dist((y, x), spot) > NEAR_M:
        return None
    name, _, kind = label.rpartition(" (")
    t_s = cues.reports_for(scenario, seed)[0][1].t_s
    return NamedSighting(seed, t_s, name, kind.rstrip(")"), round(math.dist((y, x), spot)))


def where(named: NamedSighting) -> str:
    """The place as a witness would say it."""
    if named.kind == "bridge":
        return f"by the bridge on the {named.place}"
    if named.kind in ("hamlet", "village"):
        return f"just outside {named.place}"
    return f"near {named.place}"


def named_text(text: str, named: NamedSighting) -> str:
    """The template sighting with its distance and direction replaced by the named place."""
    said = re.search(r"roughly [\d.]+ km [a-z-]+ of the house", text)
    if said is None:
        raise ValueError(f"seed {named.seed}: no distance and direction in {text!r}")
    return text.replace(said[0], where(named))


def named_reports_for(scenario: Scenario, seed: int, subject: str | None = None):
    reports, cs = _template_reports_for(scenario, seed, subject)
    named = named_sighting(scenario, seed)
    if named is None:
        return reports, cs
    return [reports[0], Report(reports[1].t_s, named_text(reports[1].text, named))], cs


class NamedPlaceSquare(ExpandingSquare):
    """A volunteer who knows every named place on the map: when a report names one, restart the square there."""

    name = "spiral+named"

    def observe(self, event: Observation) -> None:
        super().observe(event)
        if not isinstance(event, Report) or event.t_s == 0:
            return
        home_y, home_x = self.map.grid.centre_m(*self.map.ipp)
        for label, y, x in self.map.landmarks:
            if label.rpartition(" (")[0] in event.text:
                bearing = math.degrees(math.atan2(x - home_x, home_y - y)) % 360
                spot = layer(self.map, self._base_poa, Shape(spot_bearing_deg=bearing,
                                                             spot_distance_km=math.dist((y, x), (home_y, home_x)) / 1000))
                if spot is not None:
                    super().observe(Hypotheses(event.t_s, ("the named place",), ("",), (1.0,), spot[None], 0.0))
                return


_template_reports_for = cues.reports_for


def use_named_places(named: bool) -> None:
    """Point the bench at the named or the template reports (in this process and in each worker)."""
    bench.reports_for = named_reports_for if named else _template_reports_for
    bench.PLANNERS[NamedPlaceSquare.name] = NamedPlaceSquare


def kept_seeds() -> list[NamedSighting]:
    return [n for seed in SEEDS if (n := named_sighting(bench.build(seed, PLACE, "dementia", 120), seed)) is not None]


def run(seeds: list[int], names: list[str], named: bool, workers: int) -> list[bench.SeedResult]:
    use_named_places(named)
    job = partial(bench.run_seed, planner_names=names, fleet=FleetSpec(drones=1), max_s=HOURS * 3600,
                  category="dementia", size=120, imagery=True, place=PLACE)
    with ProcessPoolExecutor(workers, initializer=use_named_places, initargs=(named,)) as pool:
        return list(pool.map(job, seeds, chunksize=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefetch", action="store_true", help="fetch Spark's readings first (paid where not cached)")
    parser.add_argument("--usd-cap", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    os.environ["USD_CAP"] = str(args.usd_cap)
    kept = kept_seeds()
    seeds = [n.seed for n in kept]
    print(f"{len(kept)} of {len(SEEDS)} scenarios have a sighting near a named place")
    if args.prefetch:
        for named in (True, False):
            use_named_places(named)
            done, errors = bench.prefetch_readings(seeds, "dementia", 120, place=PLACE)
            print(f"{'named' if named else 'template'} readings: {done}, errors {len(errors)}, spent ${spent_usd():.2f}")
            for e in errors:
                print("  ", e)
    named_runs = run(seeds, NAMED_PLANNERS, named=True, workers=args.workers)
    template_runs = run(seeds, TEMPLATE_PLANNERS, named=False, workers=args.workers)
    report(kept, named_runs, template_runs)


def report(kept: list[NamedSighting], named_runs: list, template_runs: list) -> None:
    summary = bench.summarise(named_runs, NAMED_PLANNERS, "spiral", HOURS * 3600)
    print("\nNamed places:")
    bench.print_table(summary, "spiral")
    two_h = 2 * 3600
    lastseen_named = [r.runs["hedge+spark"].found_s for r in named_runs]
    lastseen_template = [r.runs["hedge+spark"].found_s for r in template_runs]
    primary = summary["planners"]["hedge+spark"]["vs_reference"]["2h"]
    secondary = metrics.paired_difference(lastseen_named, lastseen_template, two_h, np.random.default_rng(0))
    rules = metrics.paired_difference([r.runs["spiral+witness"].found_s for r in template_runs],
                                      [r.runs["spiral+witness"].found_s for r in named_runs], two_h,
                                      np.random.default_rng(0))
    print(f"\nPrimary, hedge+spark named vs the square, found within 2 h: {_points(primary)}"
          f" -> {'PASS' if primary[1] > 0 else 'FAIL'}")
    print(f"Secondary, hedge+spark named vs template wording: {_points(secondary)}"
          f" -> {'PASS' if secondary[0] >= -0.05 else 'FAIL'}")
    print(f"Rules for the template (spiral+witness), template vs named wording: {_points(rules)}")
    print(f"Spark failures: {sum(r.spark_failures for r in named_runs + template_runs)}")
    RESULTS.write_text(json.dumps({
        "design": __doc__.split("    uv run")[0].strip(),
        "scenarios": [dataclasses.asdict(n) for n in kept],
        "named": summary,
        "template": bench.summarise(template_runs, TEMPLATE_PLANNERS, "spiral+witness", HOURS * 3600),
        "per_seed": [{"seed": a.seed, "cues": a.cues,
                      "found_s": {**{k: v.found_s for k, v in a.runs.items()},
                                  **{f"{k} (template)": v.found_s for k, v in b.runs.items()}}}
                     for a, b in zip(named_runs, template_runs)],
    }, indent=1), encoding="utf-8")
    print(f"wrote {RESULTS}")


def _points(diff: tuple[float, float, float]) -> str:
    d, lo, hi = (100 * v for v in diff)
    return f"{d:+.1f} points (95% interval {lo:+.1f} to {hi:+.1f})"


if __name__ == "__main__":
    main()
