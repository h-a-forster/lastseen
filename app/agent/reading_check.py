"""How well does Spark read the reports? For each scenario and each report as it arrives, ask Spark for hypotheses and
compare its map with the plain statistics map at the person's true position at that moment:

    gain = P(true cell | Spark's map) / P(true cell | statistics only)

Above 1 the reading helped; below 1 it pointed the wrong way (a mistaken witness, or a misreading). Answers are cached,
so a rerun is free. Costs about $0.01 per new call.

    uv run python -m app.agent.reading_check --seeds 3 --category dementia
"""

import argparse
import json
import math

import numpy as np

from app.agent.interpret import interpret
from app.meta_api import spent_usd
from app.sar.cues import reports_for
from app.sar.prior import ring_prior
from app.sar.scenario import synthetic
from app.sar.stats import lost_person_stats

USD_PER_CALL = 0.01  # estimate from the day-0 spikes; the ledger records the real cost


def check_seed(seed: int, category: str) -> list[dict]:
    scenario = synthetic(seed, category=category)
    m = scenario.map
    rings = lost_person_stats()["categories"][category]["rings_km"]
    base, _ = ring_prior(m.grid, m.ipp, rings, weight=m.passability)
    base = base / base.sum()
    reports, cues = reports_for(scenario, seed)
    rows = []
    for k in range(1, len(reports) + 1):
        result = interpret(m, reports[:k], base)
        cell = scenario.subject_at(reports[k - 1].t_s)
        gain = None if cell is None else float(result.mixture(base)[cell] / base[cell])
        rows.append({
            "seed": seed, "reports": k, "gain": gain, "usd": result.usd, "cached": result.cached,
            "cues": [(c.kind, c.true) for c in cues if c.t_s <= reports[k - 1].t_s],
            "hypotheses": [(h.name, round(h.weight, 2)) for h in result.hypotheses],
            "rest": round(result.rest_weight, 2), "dropped": list(result.dropped),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--category", default="dementia")
    parser.add_argument("--json", help="also write the rows here")
    args = parser.parse_args()
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    print(f"At most {2 * args.seeds} Spark calls, about ${2 * args.seeds * USD_PER_CALL:.2f} if none are cached.")
    rows = []
    for seed in seeds:
        for row in check_seed(seed, args.category):
            rows.append(row)
            gain = "outside map" if row["gain"] is None else f"{row['gain']:.2f}x"
            cues = ", ".join(f"{kind} {'true' if ok else 'FALSE'}" for kind, ok in row["cues"])
            print(f"seed {seed} after {row['reports']} report(s): gain {gain:>11}   [{cues}]")
            for name, w in row["hypotheses"]:
                print(f"      {w:.2f}  {name}")
            print(f"      {row['rest']:.2f}  (none of these)" + "".join(f"\n      dropped: {d}" for d in row["dropped"]))
    gains = [r["gain"] for r in rows if r["gain"]]
    honest = [r["gain"] for r in rows if r["gain"] and all(ok for _, ok in r["cues"])]
    if gains:
        print(f"\nGeometric mean gain: {math.exp(np.mean(np.log(gains))):.2f}x over {len(gains)} readings"
              + (f"; {math.exp(np.mean(np.log(honest))):.2f}x when every cue was true ({len(honest)})" if honest else ""))
    print(f"Spent this run: ${spent_usd():.3f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
