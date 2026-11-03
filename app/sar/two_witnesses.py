"""Two-witness test: a second dog walker calls in a different spot, and only one of the two is right.

The simulator's reports give at most one sighting. Real searches get several, and they disagree. Here every scenario
with a sighting gets a second one 10-20 minutes later, worded as the same template so that rules written for it read
both exactly (the test is about whom to trust, not about reading). Which of the two is right is a coin toss; the
other one names a spot 1-2 km from where she really was.

Fixed before any reading was paid for (dev seeds only; the held-out set stays unseen):
- Scenarios: synthetic maps, dementia, one drone, photo mode, 4 h, dev seeds 0-199 with a sighting.
- Primary: `hedge+spark` against the expanding square alone, found within 2 h; passes if the 95% interval is above 0.
- Reported alongside: `spiral+witness`, a careful volunteer who restarts the square on each witness's spot as it
  comes in (so on the latest one).

    uv run python -m app.sar.two_witnesses --prefetch --usd-cap 4   # Spark's readings, live and cached
    MODEL_MODE=replay uv run python -m app.sar.two_witnesses        # the comparison from the cache, $0
"""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from app.meta_api import spent_usd
from app.sar import bench, cues
from app.sar.interface import FleetSpec, Report
from app.sar.scenario import Scenario

SEEDS = range(200)
SIZE = 120
SECOND_AFTER_S = (600.0, 1200.0)  # the second call comes 10-20 minutes after the first
WRONG_KM = (1.0, 2.0)  # how far the mistaken witness's spot is from where she really was
HOURS = 4.0
PLANNERS = ["spiral", "spiral+witness", "hedge+spark"]
RESULTS = Path(__file__).resolve().parents[2] / "docs" / "results" / "two-witnesses.json"

_template_reports_for = cues.reports_for


def two_witness_reports_for(scenario: Scenario, seed: int, subject: str | None = None):
    """The brief and two sighting reports, exactly one of them right; the first keeps the template's time."""
    reports, cs = _template_reports_for(scenario, seed, subject)
    if len(reports) < 2:
        return reports, cs
    rng = np.random.default_rng([seed, 5])
    first_s = reports[1].t_s
    second_s = first_s + rng.uniform(*SECOND_AFTER_S)
    right_first = rng.random() < 0.5
    calls = []
    for t_s, right in ((first_s, right_first), (second_s, not right_first)):
        age_s = min(rng.uniform(*cues.SIGHTING_AGE_S), t_s)
        y, x = scenario.track[int((t_s - age_s) // cues.RECORD_S)]
        if not right:
            bearing, dist = rng.uniform(0, 2 * np.pi), rng.uniform(*WRONG_KM) * 1000
            y, x = y - dist * math.cos(bearing), x + dist * math.sin(bearing)
        detail = cues._sighting_detail(scenario, y, x, age_s)
        calls.append((Report(t_s, cues._sighting_text(_name(reports[0].text), detail)),
                      cues.Cue("sighting", right, t_s, detail)))
    first, second = calls
    second = (Report(second[0].t_s, second[0].text.replace("A dog walker reports", "A second dog walker reports")),
              second[1])
    kept = [c for c in cs if c.kind != "sighting"]
    return [reports[0], first[0], second[0]], [*kept, first[1], second[1]]


def _name(brief: str) -> str:
    return brief.split(",", 1)[0]


def use_two_witnesses() -> None:
    bench.reports_for = two_witness_reports_for


def with_sighting() -> list[int]:
    return [s for s in SEEDS if len(_template_reports_for(bench.build(s, None, "dementia", SIZE), s)[0]) > 1]


def run(seeds: list[int], workers: int) -> list[bench.SeedResult]:
    use_two_witnesses()
    job = partial(bench.run_seed, planner_names=PLANNERS, fleet=FleetSpec(drones=1), max_s=HOURS * 3600,
                  category="dementia", size=SIZE, imagery=True)
    with ProcessPoolExecutor(workers, initializer=use_two_witnesses) as pool:
        return list(pool.map(job, seeds, chunksize=4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefetch", action="store_true", help="fetch Spark's readings first (paid where not cached)")
    parser.add_argument("--usd-cap", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    os.environ["USD_CAP"] = str(args.usd_cap)
    seeds = with_sighting()
    print(f"{len(seeds)} of {len(SEEDS)} scenarios have a sighting")
    if args.prefetch:
        use_two_witnesses()
        done, errors = bench.prefetch_readings(seeds, "dementia", SIZE)
        print(f"readings: {done}, errors {len(errors)}, spent ${spent_usd():.2f}")
        for e in errors:
            print("  ", e)
    results = run(seeds, args.workers)
    summary = bench.summarise(results, PLANNERS, "spiral", HOURS * 3600)
    bench.print_table(summary, "spiral")
    primary = summary["planners"]["hedge+spark"]["vs_reference"]["2h"]
    d, lo, hi = (100 * v for v in primary)
    print(f"\nPrimary, hedge+spark vs the square, found within 2 h: {d:+.1f} points (95% interval {lo:+.1f} to "
          f"{hi:+.1f}) -> {'PASS' if lo > 0 else 'FAIL'}")
    print(f"Spark failures: {sum(r.spark_failures for r in results)}")
    RESULTS.write_text(json.dumps({
        "design": __doc__.split("    uv run")[0].strip(), "summary": summary,
        "per_seed": [{"seed": r.seed, "cues": r.cues, "found_s": {k: v.found_s for k, v in r.runs.items()}}
                     for r in results],
    }, indent=1), encoding="utf-8")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
