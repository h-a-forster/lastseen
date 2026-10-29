"""Benchmark planners on the same scenarios (paired seeds): synthetic maps, or a real place with --place. Fast mode:
no model calls.

    uv run python -m app.sar.bench --seeds 200 --category dementia --workers 8
    uv run python -m app.sar.bench --set heldout --planners ours,spiral --json results.json
    uv run python -m app.sar.bench --planners spiral,greedy,greedy+spark --seeds 30 --usd-cap 0.45
    uv run python -m app.sar.bench --imagery             # every photo of the person decided by a real SAM 3.1 + Spark answer
    uv run python -m app.sar.bench --truth-moving 1.5    # people who walk 50% longer than the statistics say
    uv run python -m app.sar.bench --place saint-ursanne # hidden walks on a real valley mapped from OpenStreetMap
    uv run python -m app.sar.bench --lose-drone          # one drone fails in the first two hours of every search
    uv run python -m app.sar.bench --planners spiral,spiral@120   # the same planner flown at 120 m instead of 100 m

Every planner receives the same witness reports. A name ending in "@<metres>" flies that planner at that height, so
heights are compared on the same scenarios. A name ending in "+spark" wraps that planner in the Spark advisor:
Spark's readings are fetched first, in this process and under --usd-cap, then the simulations replay them from the
cache, so the parallel workers can't spend anything.

Tune on the `dev` seeds only. The `heldout` seeds are for final numbers, so a result is never fitted to its own test.
"""

import argparse
import copy
import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import numpy as np

from app.agent.advisor import SparkAdvisor, read_all
from app.meta_api import ModelCallError, spent_usd
from app.perception import pool as photo_pool
from app.planner import SearchPlanner
from app.sar import metrics
from app.sar.detection import DetectionModel
from app.sar.grid import Terrain
from app.sar.interface import MAX_ALT_M, MIN_ALT_M, FleetSpec
from app.sar.planners import BASELINES
from app.sar.cues import reports_for
from app.sar.terrain import PLACES, build
from app.sar.sim import RunResult, Sim, drone_losses
from app.sar.stats import lost_person_stats
from app.sar.walker import PARAMS, WalkerParams

PLANNERS = {**BASELINES, SearchPlanner.name: SearchPlanner}
SPARK = "+spark"
ALT = "@"
SEED_SETS = {"dev": range(0, 100_000), "heldout": range(900_000, 1_000_000)}
CHECKPOINTS_H = (1.0, 2.0)
PRIMARY_H = 2.0  # the primary endpoint in docs/evaluation.md


@dataclasses.dataclass
class SeedResult:
    seed: int
    in_area_at_start: bool  # findable at all at t = 0; the rest start outside the mapped area
    runs: dict[str, RunResult]
    spark_failures: int  # model readings that failed; those runs carried on without hypotheses
    cues: dict[str, bool]  # which report details were true, by kind (a kind is absent if no report gave it)


def make_planner(name: str):
    name = name.partition(ALT)[0]
    if name.endswith(SPARK):
        return SparkAdvisor(PLANNERS[name.removesuffix(SPARK)]())
    return PLANNERS[name]()


def fleet_for(name: str, fleet: FleetSpec) -> FleetSpec:
    """The fleet a planner name flies: the shared one, or at the height after "@" (e.g. "spiral@120")."""
    alt = name.partition(ALT)[2]
    if not alt:
        return fleet
    if not MIN_ALT_M <= float(alt) <= MAX_ALT_M:
        raise ValueError(f"{name}: height must be {MIN_ALT_M:g}-{MAX_ALT_M:g} m")
    return dataclasses.replace(fleet, alt_m=float(alt))


def run_seed(seed: int, planner_names: list[str], fleet: FleetSpec, max_s: float, category: str, size: int,
             truth: WalkerParams | None = None, imagery: bool = False,
             detection: DetectionModel | None = None, place: str | None = None, lose_drone: bool = False) -> SeedResult:
    scenario = build(seed, place, category, size, truth)
    reports, cues = reports_for(scenario, seed)
    photos = photo_pool.shared() if imagery else None
    if imagery and photos is None:
        raise RuntimeError("imagery mode needs gate 1 results (python -m app.perception.gate)")
    losses = drone_losses(seed, fleet.drones) if lose_drone else None
    runs, failures = {}, 0
    for name in planner_names:
        planner = make_planner(name)
        runs[name] = Sim(scenario, planner, fleet_for(name, fleet), detection, seed=seed, max_s=max_s, reports=reports,
                         photos=photos, losses=losses).run()
        failures += len(getattr(planner, "failures", ()))
    return SeedResult(seed, scenario.subject_at(0) is not None, runs, failures, {c.kind: c.true for c in cues})


def prefetch_readings(seeds: list[int], category: str, size: int, truth: WalkerParams | None = None,
                      place: str | None = None, threads: int = 4) -> tuple[int, list[str]]:
    """Fetch Spark's reading of every report prefix for these scenarios (cached ones are free). Returns how many
    readings are available and any errors; stops cleanly at the spend cap."""
    stats = lost_person_stats()

    def one(seed: int) -> int:
        scenario = build(seed, place, category, size, truth)
        return len(read_all(scenario.map, reports_for(scenario, seed)[0], copy.deepcopy(stats)))

    done, errors = 0, []
    with ThreadPoolExecutor(threads) as pool:
        for seed, future in zip(seeds, [pool.submit(one, s) for s in seeds]):
            try:
                done += future.result()
            except ModelCallError as e:
                errors.append(f"seed {seed}: {e}")
    return done, errors


def run(seeds: list[int], planner_names: list[str], fleet: FleetSpec, max_s: float, category: str, size: int,
        workers: int, truth: WalkerParams | None = None, imagery: bool = False,
        detection: DetectionModel | None = None, place: str | None = None, lose_drone: bool = False) -> list[SeedResult]:
    job = partial(run_seed, planner_names=planner_names, fleet=fleet, max_s=max_s, category=category, size=size,
                  truth=truth, imagery=imagery, detection=detection, place=place, lose_drone=lose_drone)
    if workers <= 1:
        return [job(s) for s in seeds]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(job, seeds, chunksize=4))


def summarise(results: list[SeedResult], planner_names: list[str], reference: str, max_s: float) -> dict:
    rng = np.random.default_rng(0)
    n = len(results)
    checkpoints = [h * 3600 for h in CHECKPOINTS_H if h * 3600 < max_s] + [max_s]
    summary = {"scenarios": n, "in_area_at_start": sum(r.in_area_at_start for r in results) / n, "planners": {}}
    for name in planner_names:
        times = [r.runs[name].found_s for r in results]
        entry = {
            "found_by": {f"{t / 3600:g}h": [found / n, *metrics.wilson(found, n)] for t in checkpoints
                         for found in [metrics.found_by(times, t)]},
            "median_min": None if (m := metrics.median_time(times)) is None else m / 60,
            "forced_returns": sum(r.runs[name].forced_returns for r in results),
            "no_fly_s": sum(r.runs[name].no_fly_s for r in results),
            "plan_ms_max": max(r.runs[name].plan_ms_max for r in results),
            # Share of the working drones' time spent flying; the rest is turnarounds at base or a planner idling.
            "air": float(np.mean([r.runs[name].flight_s / max(r.runs[name].drone_s, 1.0) for r in results])),
            # Per search, averaged: photos a person would have to review without triage, alarms they review with it,
            # and fresh photos that had the person in them but found nobody (the detector's misses).
            "photos": float(np.mean([r.runs[name].photos for r in results])),
            "alarms": float(np.mean([r.runs[name].alarms for r in results])),
            "missed_in_frame": float(np.mean([r.runs[name].in_frame - (r.runs[name].found_s is not None)
                                              for r in results])),
            # What each find cost, over all searches (failed ones included): drone flight hours and photos to review.
            "per_find": per_find([r.runs[name] for r in results]),
            "curve": metrics.time_to_find_curve(times),
        }
        if name != reference and reference in planner_names:
            ref_times = [r.runs[reference].found_s for r in results]
            entry["vs_reference"] = {f"{t / 3600:g}h": metrics.paired_difference(times, ref_times, t, rng) for t in checkpoints}
        summary["planners"][name] = entry
    summary["spark_gain"] = {name: spark_gain(results, name, checkpoints)
                             for name in planner_names if SPARK in name and name.replace(SPARK, "") in planner_names}
    summary["calibrated"] = all(run.calibrated for r in results for run in r.runs.values())
    return summary


def spark_gain(results: list[SeedResult], name: str, checkpoints: list[float]) -> dict:
    """With Spark minus without, same planner, paired by scenario: the evaluation's primary comparison. The primary
    checkpoint is also split by whether the sighting report was right, wrong or not given."""
    without = name.replace(SPARK, "")

    def diff(rs: list[SeedResult], t_s: float) -> tuple[float, float, float]:
        return metrics.paired_difference([r.runs[name].found_s for r in rs], [r.runs[without].found_s for r in rs],
                                         t_s, np.random.default_rng(0))

    primary_s = min(PRIMARY_H * 3600, checkpoints[-1])
    groups = {label: [r for r in results if sighting_label(r.cues.get("sighting")) == label]
              for label in ("sighting right", "sighting wrong", "no sighting")}
    return {"without": without, "diff": {f"{t / 3600:g}h": diff(results, t) for t in checkpoints},
            "by_sighting": {"at": f"{primary_s / 3600:g}h",
                            **{label: {"scenarios": len(rs), "diff": diff(rs, primary_s)} for label, rs in groups.items() if rs}}}


def sighting_label(true: bool | None) -> str:
    return "no sighting" if true is None else "sighting right" if true else "sighting wrong"


def per_find(runs: list[RunResult]) -> dict | None:
    finds = sum(r.found_s is not None for r in runs)
    if not finds:
        return None
    return {"drone_h": sum(r.flight_s for r in runs) / 3600 / finds, "photos": sum(r.photos for r in runs) / finds,
            "alarms": sum(r.alarms for r in runs) / finds}


def print_table(summary: dict, reference: str) -> None:
    keys = list(next(iter(summary["planners"].values()))["found_by"])
    header = "".join(f"{'found ' + k:>22}" for k in keys)
    print(f"{'planner':<16}{header}{'median min':>11}{'vs ' + reference + ' (end)':>24}{'RTL':>5}{'no-fly s':>9}"
          f"{'plan ms':>8}{'air':>6}{'photos':>8}{'alarms':>8}{'missed':>7}{'h/find':>8}")
    for name, e in summary["planners"].items():
        cells = "".join(f"{p:>8.0%} [{lo:>4.0%}-{hi:>4.0%}]" for p, lo, hi in e["found_by"].values())
        median = "-" if e["median_min"] is None else f"{e['median_min']:.0f}"
        diff = "" if "vs_reference" not in e else "{:>+7.0%} [{:>+4.0%},{:>+4.0%}]".format(*e["vs_reference"][keys[-1]])
        print(f"{name:<16}{cells}{median:>11}{diff:>24}{e['forced_returns']:>5}{e['no_fly_s']:>9.0f}{e['plan_ms_max']:>8.1f}"
              f"{e['air']:>6.0%}{e['photos']:>8.0f}{e['alarms']:>8.0f}{e['missed_in_frame']:>7.1f}"
              f"{'-' if e['per_find'] is None else format(e['per_find']['drone_h'], '.1f'):>8}")
    print(f"{summary['scenarios']} scenarios; subject inside the mapped area at the start in {summary['in_area_at_start']:.0%} "
          "(the ceiling). Brackets: 95% intervals (Wilson; paired bootstrap for differences).\n"
          "air: share of drone time spent flying; h/find: drone flight hours per person found; "
          "photos / alarms: per search, what a reviewer would check without / with SAM 3.1 + Spark triage; "
          "missed: photos with the person in them that found nobody.")
    for name, gain in summary["spark_gain"].items():
        split = gain["by_sighting"]
        at = split["at"]
        print(f"Spark, same planner: {name} minus {gain['without']}, found within {at}: "
              "{:+.0%} [{:+.0%}, {:+.0%}]".format(*gain["diff"][at]) + "; by sighting report: "
              + ", ".join(f"{label} {g['diff'][0]:+.0%} (n={g['scenarios']})"
                          for label, g in split.items() if label != "at"))
    if not summary["calibrated"]:
        print("POD table is the placeholder: numbers are uncalibrated.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--planners", default=",".join(BASELINES))
    parser.add_argument("--reference", default="spiral", help="baseline the others are compared with, paired by seed")
    parser.add_argument("--set", default="dev", choices=sorted(SEED_SETS))
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--offset", type=int, default=0, help="skip this many seeds of the set first")
    parser.add_argument("--drones", type=int, default=3)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--category", default="dementia")
    parser.add_argument("--size", type=int, default=120, help="map side in 25 m cells (120 = 3 km; hikers need ~320)")
    parser.add_argument("--place", choices=sorted(PLACES), help="a real area from OpenStreetMap instead of synthetic maps")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--usd-cap", type=float, default=0.5, help="most this run may spend on Spark readings")
    parser.add_argument("--json", help="also write the summary and per-seed times here")
    parser.add_argument("--imagery", action="store_true", help="decide photos of the person with real gate 1 answers")
    parser.add_argument("--truth-category", help="walk like this category while planners are told --category")
    parser.add_argument("--truth-moving", type=float, default=1.0, help="scale the truth's moving time")
    parser.add_argument("--truth-drift", type=float, default=1.0, help="scale the truth's heading drift")
    parser.add_argument("--truth-delay-h", help="'lo,hi': hours missing before the search starts, instead of the default")
    parser.add_argument("--alt", type=float, default=FleetSpec.alt_m, help="flying height in metres (40-120)")
    parser.add_argument("--swap-s", type=float, default=FleetSpec.swap_s, help="turnaround at base after each landing")
    parser.add_argument("--forest-factor", type=float, help="forest POD as a share of open ground (measured table: 0.4)")
    parser.add_argument("--lose-drone", action="store_true", help="one drone fails at a seeded time in the first 2 h")
    args = parser.parse_args()
    names = args.planners.split(",")
    fleet = FleetSpec(drones=args.drones, alt_m=args.alt, swap_s=args.swap_s)
    for name in names:  # unknown names and bad heights fail here, not in a worker halfway through
        make_planner(name), fleet_for(name, fleet)
    seeds = list(SEED_SETS[args.set][args.offset: args.offset + args.seeds])
    max_s = args.hours * 3600
    started = time.perf_counter()
    truth = None
    if args.truth_category or args.truth_moving != 1.0 or args.truth_drift != 1.0 or args.truth_delay_h:
        base = PARAMS[args.truth_category or args.category]
        delay = tuple(map(float, args.truth_delay_h.split(","))) if args.truth_delay_h else base.report_delay_h
        truth = dataclasses.replace(base, moving_h=base.moving_h * args.truth_moving,
                                    turn_sd_deg=base.turn_sd_deg * args.truth_drift, report_delay_h=delay)
    detection = DetectionModel.load()
    if args.forest_factor is not None:
        open_pod = detection.pod_by_terrain[Terrain.OPEN]
        detection = dataclasses.replace(detection, pod_by_terrain={
            **detection.pod_by_terrain, Terrain.FOREST: tuple(p * args.forest_factor for p in open_pod)})
    if any(n.endswith(SPARK) for n in names):
        os.environ["USD_CAP"] = str(args.usd_cap)
        readings, errors = prefetch_readings(seeds, args.category, args.size, truth, args.place)
        print(f"Spark readings ready: {readings} (spent ${spent_usd():.3f} now; cap ${args.usd_cap:.2f})"
              + "".join(f"\n  {e}" for e in errors[:5]))
        os.environ["MODEL_MODE"] = "replay"  # the simulations only replay; workers inherit this
    results = run(seeds, names, fleet, max_s, args.category, args.size,
                  args.workers, truth, args.imagery, detection, args.place, args.lose_drone)
    summary = summarise(results, names, args.reference, max_s)
    print_table(summary, args.reference)
    failures = sum(r.spark_failures for r in results)
    if failures:
        print(f"{failures} Spark readings were missing; those runs carried on without hypotheses.")
    print(f"{args.set} set, {len(seeds)} scenarios, {args.category}, {args.place or f"{args.size * 25 / 1000:g} km synthetic"} map, in {time.perf_counter() - started:.0f} s")
    if args.json:
        summary["args"] = vars(args)
        summary["per_seed"] = [{"seed": r.seed, "found_s": {k: v.found_s for k, v in r.runs.items()}, "cues": r.cues}
                               for r in results]
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
