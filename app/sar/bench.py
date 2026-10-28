"""Benchmark planners on the same synthetic scenarios (paired seeds). Fast mode: no model calls.

    uv run python -m app.sar.bench --seeds 200 --category dementia --workers 8
    uv run python -m app.sar.bench --set heldout --planners ours,spiral --json results.json

Tune on the `dev` seeds only. The `heldout` seeds are for final numbers, so a result is never fitted to its own test.
"""

import argparse
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from app.planner import SearchPlanner
from app.sar import metrics
from app.sar.interface import FleetSpec
from app.sar.planners import BASELINES
from app.sar.scenario import synthetic
from app.sar.sim import RunResult, Sim

PLANNERS = {**BASELINES, SearchPlanner.name: SearchPlanner}
SEED_SETS = {"dev": range(0, 100_000), "heldout": range(900_000, 1_000_000)}
CHECKPOINTS_H = (1.0, 2.0)


@dataclasses.dataclass
class SeedResult:
    seed: int
    in_area_at_start: bool  # findable at all at t = 0; the rest start outside the mapped area
    runs: dict[str, RunResult]


def run_seed(seed: int, planner_names: list[str], fleet: FleetSpec, max_s: float, category: str) -> SeedResult:
    scenario = synthetic(seed, category=category)
    runs = {name: Sim(scenario, PLANNERS[name](), fleet, seed=seed, max_s=max_s).run() for name in planner_names}
    return SeedResult(seed, scenario.subject_at(0) is not None, runs)


def run(seeds: list[int], planner_names: list[str], fleet: FleetSpec, max_s: float, category: str, workers: int) -> list[SeedResult]:
    job = partial(run_seed, planner_names=planner_names, fleet=fleet, max_s=max_s, category=category)
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
            "curve": metrics.time_to_find_curve(times),
        }
        if name != reference and reference in planner_names:
            ref_times = [r.runs[reference].found_s for r in results]
            entry["vs_reference"] = {f"{t / 3600:g}h": metrics.paired_difference(times, ref_times, t, rng) for t in checkpoints}
        summary["planners"][name] = entry
    summary["calibrated"] = all(run.calibrated for r in results for run in r.runs.values())
    return summary


def print_table(summary: dict, reference: str) -> None:
    keys = list(next(iter(summary["planners"].values()))["found_by"])
    header = "".join(f"{'found ' + k:>22}" for k in keys)
    print(f"{'planner':<12}{header}{'median min':>11}{'vs ' + reference + ' (end)':>24}{'RTL':>5}{'no-fly s':>9}{'plan ms':>8}")
    for name, e in summary["planners"].items():
        cells = "".join(f"{p:>8.0%} [{lo:>4.0%}-{hi:>4.0%}]" for p, lo, hi in e["found_by"].values())
        median = "-" if e["median_min"] is None else f"{e['median_min']:.0f}"
        diff = "" if "vs_reference" not in e else "{:>+7.0%} [{:>+4.0%},{:>+4.0%}]".format(*e["vs_reference"][keys[-1]])
        print(f"{name:<12}{cells}{median:>11}{diff:>24}{e['forced_returns']:>5}{e['no_fly_s']:>9.0f}{e['plan_ms_max']:>8.1f}")
    print(f"{summary['scenarios']} scenarios; subject inside the mapped area at the start in {summary['in_area_at_start']:.0%} "
          "(the ceiling). Brackets: 95% intervals (Wilson; paired bootstrap for differences).")
    if not summary["calibrated"]:
        print("POD table is the placeholder: numbers are uncalibrated.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--planners", default=",".join(BASELINES))
    parser.add_argument("--reference", default="spiral", help="baseline the others are compared with, paired by seed")
    parser.add_argument("--set", default="dev", choices=sorted(SEED_SETS))
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--drones", type=int, default=3)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--category", default="dementia")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", help="also write the summary and per-seed times here")
    args = parser.parse_args()
    names = args.planners.split(",")
    seeds = list(SEED_SETS[args.set][: args.seeds])
    max_s = args.hours * 3600
    started = time.perf_counter()
    results = run(seeds, names, FleetSpec(drones=args.drones), max_s, args.category, args.workers)
    summary = summarise(results, names, args.reference, max_s)
    print_table(summary, args.reference)
    print(f"{args.set} set, {len(seeds)} scenarios, {args.category}, in {time.perf_counter() - started:.0f} s")
    if args.json:
        summary["args"] = vars(args)
        summary["per_seed"] = [{"seed": r.seed, "found_s": {k: v.found_s for k, v in r.runs.items()}} for r in results]
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
