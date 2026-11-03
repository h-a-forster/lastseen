"""Record a race: several planners on the same scenario, same witness reports, same seed, each with its event log.
The replay page plays the file back. Spark readings come from the cache only (no spending) unless `--live`.

    uv run python -m app.sar.replay --seed 17 --planners lawnmower,greedy,greedy+spark
    uv run python -m app.sar.replay --seed 3 --live --usd-cap 0.05
    uv run python -m app.sar.replay --seed 19 --place saint-ursanne  # a real valley; the page shows aerial imagery
    uv run python -m app.sar.replay --seed 5 --planners spiral,hedge+spark --drones 1

Races run in imagery mode when gate 1's results are here: every photo that has the person in it is decided by a real
SAM 3.1 + Muse Spark answer on a real drone photo of a person that size, and the page shows that photo.

One scenario is an illustration, not evidence: the benchmark (`app.sar.bench`) is where claims come from.
"""

import argparse
import dataclasses
import json
import math
import os
import re
import zlib
from pathlib import Path

import numpy as np

from app.perception import pool as photo_pool
from app.perception.heridal import CREDIT
from app.sar.bench import fleet_for, make_planner
from app.sar.cues import Cue, Report, reports_for, right_sighting, true_sighting
from app.sar.interface import FleetSpec
from app.sar.planners.fixed_paths import ExpandingSquare
from app.sar.recorder import Recorder
from app.sar.scenario import Scenario
from app.sar.terrain import PLACES, build
from app.sar.terrain import CREDIT as MAP_CREDIT
from app.sar.sim import Sim, drone_losses
from app.sar.walker import RECORD_S

RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "runs"
MAX_LANES = 3
HEADING_TRIES = 60  # walks tried for one that goes the way a visitor chose (terrain makes some directions rare)
SEEN_RADIUS_M = 300.0  # a visitor's witness spot: she passed within this distance of it...
SEEN_AGE_S = (300.0, 2400.0)  # ...5-40 minutes before the call


def race(seed: int, planner_names: list[str], category: str = "dementia", size: int = 120, hours: float = 4.0,
         drones: int = 3, place: str | None = None, lose_drone: bool = False, sighting_s: float | None = None,
         subject: str | None = None, witness: tuple[str, tuple[float, float]] | None = None) -> dict:
    """Every planner flies the same hidden walk. With `sighting_s` (a visitor's test search), a sighting that is right
    arrives at that search time instead of the generated one, if any. `subject` replaces the person's name in the
    reports. `witness` is a visitor's own sighting at `sighting_s`: their words, read as they are, and the map spot
    (y_m, x_m) where they say she was seen, which the truth then describes (see `seen_near`)."""
    scenario = build(seed, place, category, size)
    reports, cues = reports_for(scenario, seed, subject)
    if sighting_s is not None:
        sighting, cue = (true_sighting(scenario, seed, sighting_s, subject) if witness is None
                         else _witness_sighting(scenario, sighting_s, *witness))
        reports, cues = [reports[0], sighting], [c for c in cues if c.kind != "sighting"] + [cue]
    fleet = FleetSpec(drones=drones)
    losses = drone_losses(seed, drones) if lose_drone else None
    photos = photo_pool.shared()
    lanes = {}
    for name in planner_names:
        recorder = Recorder()
        result = Sim(scenario, make_planner(name), fleet_for(name, fleet), seed=seed, max_s=hours * 3600, reports=reports,
                     recorder=recorder, photos=photos, losses=losses).run()
        lanes[name] = {"result": dataclasses.asdict(result), "events": recorder.events}
    return {
        "id": run_id(seed, category, planner_names, place, drones) + ("-lost" if lose_drone else "")
              + (f"-sighting-{sighting_s / 60:.0f}min" if sighting_s is not None else "")
              + (f"-witness-{zlib.crc32(repr(witness).encode()):08x}" if witness is not None else ""),
        "seed": seed,
        "category": category,
        "place": None if place is None else {"name": place, "title": PLACES[place].title,
                                             "bounds": PLACES[place].bounds(), "credit": MAP_CREDIT,
                                             "aerial": PLACES[place].aerial},
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


def _witness_sighting(scenario: Scenario, t_s: float, text: str, spot_m: tuple[float, float]) -> tuple[Report, Cue]:
    seen_s = seen_near(scenario, spot_m, t_s)
    if seen_s is None:
        raise ValueError("this walk doesn't pass the witness's spot before the call")
    return Report(t_s, text), right_sighting(scenario, t_s, t_s - seen_s)


def watchable_seed(place: str | None, first: int, sighting_s: float | None = None, tries: int = 5,
                   heading_deg: float | None = None, seen_m: tuple[float, float] | None = None) -> int:
    """For a visitor's place: the first of `tries` walks from seed `first` with a sighting (at `sighting_s`, or the
    generated one) while she is still in the mapped area, which the expanding square hasn't found her
    before, so the race shows the planners split. Up to the sighting both planners fly the same square, so the pick
    favours neither. Photos are decided as in the race; no model calls.
    With `heading_deg`, only walks that have her within 22.5 degrees of that bearing from the start point at the
    sighting count, from up to HEADING_TRIES: the walker is unchanged, this only picks among its walks. Likewise with
    `seen_m`, a witness's spot (y_m, x_m): only walks that pass near it shortly before `sighting_s` (see `seen_near`).
    Falls back to a walk that meets the place, direction and spot tests, then to `first`; ValueError if a chosen
    direction or spot has no walk at all."""
    if seen_m is not None and sighting_s is None:
        raise ValueError("a witness's spot needs the time of the call")
    photos = photo_pool.shared()
    fallback = None
    chosen = heading_deg is not None or seen_m is not None
    for seed in range(first, first + (HEADING_TRIES if chosen else tries)):
        scenario = build(seed, place, "dementia")
        seen_s = sighting_s if sighting_s is not None else _sighting_s(scenario, seed)
        if seen_s is None or scenario.subject_at(seen_s) is None:
            continue
        if heading_deg is not None and abs((bearing_at(scenario, seen_s) - heading_deg + 180) % 360 - 180) > 22.5:
            continue
        if seen_m is not None and seen_near(scenario, seen_m, seen_s) is None:
            continue
        fallback = seed if fallback is None else fallback
        if Sim(scenario, ExpandingSquare(), FleetSpec(drones=1), seed=seed, max_s=seen_s,
               photos=photos).run().found_s is None:
            return seed
    if fallback is None and heading_deg is not None:
        raise ValueError("none of the simulated walks from here goes that way; try another direction")
    if fallback is None and seen_m is not None:
        raise ValueError("none of the simulated walks passes there before the call; mark a spot closer to the start "
                         "or call later")
    return first if fallback is None else fallback


def seen_near(scenario: Scenario, spot_m: tuple[float, float], call_s: float) -> float | None:
    """The recorded search time, SEEN_AGE_S before `call_s`, when the person was closest to `spot_m` (y_m, x_m), the
    latest if she stayed there a while, or None if she was never within SEEN_RADIUS_M of it then."""
    first = max(0, math.ceil((call_s - SEEN_AGE_S[1]) / RECORD_S))
    last = min(math.floor((call_s - SEEN_AGE_S[0]) / RECORD_S), len(scenario.track) - 1)
    if last < first:
        return None
    distances = np.hypot(*(scenario.track[first:last + 1] - np.asarray(spot_m)).T)
    closest = len(distances) - 1 - int(distances[::-1].argmin())
    return (first + closest) * RECORD_S if distances[closest] <= SEEN_RADIUS_M else None


def bearing_at(scenario: Scenario, t_s: float) -> float:
    """Compass bearing (degrees, 0 = north) from the start point to where the person is at search time `t_s`."""
    y, x = scenario.track[min(int(t_s // RECORD_S), len(scenario.track) - 1)]
    start_y, start_x = scenario.map.grid.centre_m(*scenario.map.ipp)
    return math.degrees(math.atan2(x - start_x, start_y - y)) % 360  # rows grow southwards


def _sighting_s(scenario: Scenario, seed: int) -> float | None:
    reports, _cues = reports_for(scenario, seed)
    return reports[1].t_s if len(reports) > 1 else None


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


def run_id(seed: int, category: str, planner_names: list[str], place: str | None = None,
           drones: int = FleetSpec.drones) -> str:
    prefix = f"{place}-" if place else ""
    suffix = "" if drones == FleetSpec.drones else f"-{drones}drone"
    return re.sub(r"[^a-z0-9-]", "",
                  f"{prefix}{category}-{seed}-" + "-".join(n.replace("+", "-") for n in planner_names) + suffix)


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
    parser.add_argument("--drones", type=int, default=FleetSpec.drones)
    parser.add_argument("--live", action="store_true", help="allow paid Spark calls for readings not in the cache")
    parser.add_argument("--usd-cap", type=float, default=0.05)
    parser.add_argument("--lose-drone", action="store_true", help="one drone fails at a seeded time in the first 2 h")
    args = parser.parse_args()
    names = args.planners.split(",")[:MAX_LANES]
    os.environ["MODEL_MODE"] = "live" if args.live else "replay"
    os.environ["USD_CAP"] = str(args.usd_cap)
    replay = race(args.seed, names, args.category, args.size, args.hours, args.drones, args.place, args.lose_drone)
    path = save(replay)
    for name, lane in replay["lanes"].items():
        found = lane["result"]["found_s"]
        print(f"{name:<14} {'not found' if found is None else f'found at {found / 60:.0f} min'}"
              f"  ({len(lane['events'])} events)")
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
