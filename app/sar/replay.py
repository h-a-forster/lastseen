"""Record a race: several planners on the same scenario, same witness reports, same seed, each with its event log.
The replay page plays the file back. Spark readings come from the cache only (no spending) unless `--live`.

    uv run python -m app.sar.replay --seed 17 --planners lawnmower,greedy,greedy+spark
    uv run python -m app.sar.replay --seed 3 --live --usd-cap 0.05
    uv run python -m app.sar.replay --seed 19 --place saint-ursanne  # a real valley; the page shows aerial imagery

Races run in imagery mode when gate 1's results are here: every photo that has the person in it is decided by a real
SAM 3.1 + Muse Spark answer on a real drone photo of a person that size, and the page shows that photo.

One scenario is an illustration, not evidence: the benchmark (`app.sar.bench`) is where claims come from.
"""

import argparse
import dataclasses
import json
import os
import re
from pathlib import Path

from app.perception import pool as photo_pool
from app.perception.heridal import CREDIT
from app.sar.bench import make_planner
from app.sar.cues import Cue, reports_for
from app.sar.interface import FleetSpec
from app.sar.recorder import Recorder
from app.sar.scenario import Scenario
from app.sar.terrain import PLACES, build
from app.sar.terrain import CREDIT as MAP_CREDIT
from app.sar.sim import Sim, drone_losses
from app.sar.walker import RECORD_S

RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "runs"
MAX_LANES = 3


def race(seed: int, planner_names: list[str], category: str = "dementia", size: int = 120, hours: float = 4.0,
         drones: int = 3, place: str | None = None, lose_drone: bool = False) -> dict:
    scenario = build(seed, place, category, size)
    reports, cues = reports_for(scenario, seed)
    fleet = FleetSpec(drones=drones)
    losses = drone_losses(seed, drones) if lose_drone else None
    photos = photo_pool.shared()
    lanes = {}
    for name in planner_names:
        recorder = Recorder()
        result = Sim(scenario, make_planner(name), fleet, seed=seed, max_s=hours * 3600, reports=reports,
                     recorder=recorder, photos=photos, losses=losses).run()
        lanes[name] = {"result": dataclasses.asdict(result), "events": recorder.events}
    return {
        "id": run_id(seed, category, planner_names, place) + ("-lost" if lose_drone else ""),
        "seed": seed,
        "category": category,
        "place": None if place is None else {"name": place, "title": PLACES[place].title,
                                             "bounds": PLACES[place].bounds(), "credit": MAP_CREDIT},
        "hours": hours,
        "imagery": photos is not None,
        "photo_credit": CREDIT if photos is not None else None,
        "fleet": {"drones": drones, "speed_mps": fleet.speed_mps, "alt_m": fleet.alt_m,
                  "footprint_m": fleet.footprint_m, "endurance_s": fleet.endurance_s},
        "map": map_json(scenario),
        "reports": [{"t_s": r.t_s, "text": r.text} for r in reports],
        "truth": truth_json(scenario, cues, hours),
        "lanes": lanes,
    }


def map_json(scenario: Scenario) -> dict:
    """The map as the page draws it: terrain as one digit per cell (Terrain values), trails and no-fly as 0/1."""
    m = scenario.map
    return {
        "rows": m.grid.rows, "cols": m.grid.cols, "cell_m": m.grid.cell_m,
        "terrain": "".join(str(int(v)) for v in m.terrain.ravel()),
        "trails": "".join("1" if v else "0" for v in m.trails.ravel()),
        "no_fly": "".join("1" if v else "0" for v in m.no_fly.ravel()),
        "ipp": list(m.ipp), "base": list(m.base),
    }


def truth_json(scenario: Scenario, cues: list[Cue], hours: float) -> dict:
    """Hidden during playback, revealed at the end: where the person really was, and which report details were right."""
    records = int(hours * 3600 // RECORD_S) + 1
    track = [[round(float(y), 1), round(float(x), 1)] for y, x in scenario.track[:records]]
    return {"every_s": RECORD_S, "track": track,
            "cues": [{"kind": c.kind, "true": c.true, "t_s": c.t_s, "detail": c.detail} for c in cues]}


def run_id(seed: int, category: str, planner_names: list[str], place: str | None = None) -> str:
    prefix = f"{place}-" if place else ""
    return re.sub(r"[^a-z0-9-]", "", f"{prefix}{category}-{seed}-" + "-".join(n.replace("+", "-") for n in planner_names))


def save(replay: dict) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{replay['id']}.json"
    partial = path.with_suffix(".part")  # written whole, then renamed, so a reader never sees half a file
    partial.write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
    os.replace(partial, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--planners", default="lawnmower,greedy,greedy+spark")
    parser.add_argument("--category", default="dementia")
    parser.add_argument("--size", type=int, default=120)
    parser.add_argument("--place", choices=sorted(PLACES), help="a real area from OpenStreetMap instead of a synthetic map")
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--live", action="store_true", help="allow paid Spark calls for readings not in the cache")
    parser.add_argument("--usd-cap", type=float, default=0.05)
    parser.add_argument("--lose-drone", action="store_true", help="one drone fails at a seeded time in the first 2 h")
    args = parser.parse_args()
    names = args.planners.split(",")[:MAX_LANES]
    os.environ["MODEL_MODE"] = "live" if args.live else "replay"
    os.environ["USD_CAP"] = str(args.usd_cap)
    replay = race(args.seed, names, args.category, args.size, args.hours, place=args.place, lose_drone=args.lose_drone)
    path = save(replay)
    for name, lane in replay["lanes"].items():
        found = lane["result"]["found_s"]
        print(f"{name:<14} {'not found' if found is None else f'found at {found / 60:.0f} min'}"
              f"  ({len(lane['events'])} events)")
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
