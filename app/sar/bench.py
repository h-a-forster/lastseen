"""Benchmark planners on the same synthetic scenarios (paired seeds). Fast mode: no model calls.

    uv run python -m app.sar.bench --seeds 50 --drones 3
"""

import argparse
import statistics
import time

import numpy as np

from app.sar.planners import PLANNERS
from app.sar.scenario import synthetic
from app.sar.sim import FleetSpec, RunResult, Sim


def run(planner_names: list[str], seeds: int, fleet: FleetSpec, max_s: float, category: str) -> dict[str, list[RunResult]]:
    results: dict[str, list[RunResult]] = {name: [] for name in planner_names}
    for seed in range(seeds):
        scenario = synthetic(seed, category=category)
        for name in planner_names:
            results[name].append(Sim(scenario, PLANNERS[name](), fleet, seed=seed, max_s=max_s).run())
    return results


def summarise(results: dict[str, list[RunResult]], max_s: float) -> None:
    print(f"{'planner':<12}{'found':>14}{'median min':>12}{'POS@1h':>8}{'POS@end':>9}{'forced RTL':>11}{'no-fly s':>9}")
    for name, runs in results.items():
        found = [r.found_s for r in runs if r.found_s is not None]
        rate = len(found) / len(runs)
        half_width = 1.96 * (rate * (1 - rate) / len(runs)) ** 0.5
        # Censored runs count as max_s, so the median is a lower bound when fewer than half are found.
        times = sorted((r.found_s if r.found_s is not None else max_s) / 60 for r in runs)
        print(f"{name:<12}{rate:>7.0%} +-{half_width:>4.0%}{statistics.median(times):>12.0f}"
              f"{_pos_at(runs, 3600):>8.2f}{_pos_at(runs, max_s):>9.2f}"
              f"{sum(r.forced_returns for r in runs):>11}{sum(r.no_fly_s for r in runs):>9.0f}")
    if not all(r.calibrated for runs in results.values() for r in runs):
        print("POD table is the placeholder: numbers are uncalibrated.")


def _pos_at(runs: list[RunResult], t_s: float) -> float:
    """Mean cumulative POS at time t (runs that ended early keep their last value)."""
    values = []
    for r in runs:
        before = [pos for t, pos in r.cumulative_pos if t <= t_s]
        values.append(before[-1] if before else 0.0)
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--planners", default=",".join(PLANNERS))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--drones", type=int, default=3)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--category", default="hiker")
    args = parser.parse_args()
    max_s = args.hours * 3600
    started = time.perf_counter()
    results = run(args.planners.split(","), args.seeds, FleetSpec(drones=args.drones), max_s, args.category)
    summarise(results, max_s)
    print(f"{args.seeds} scenarios in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
