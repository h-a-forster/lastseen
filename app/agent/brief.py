"""Start a real search from what people said. The brief (typed, or spoken and transcribed by Muse Voice Transcribe)
is read by Muse Spark into weighted hypotheses on a real place's map, exactly as in the simulated races, and Spark
grades who said what (`witnesses.py`); the reference planner then picks a first block for each drone, which downloads as a QGroundControl mission.

    uv run python -m app.agent.brief --example    # the page's example brief (one Spark call the first time)
"""

import argparse
import json
from pathlib import Path

from app.agent.advisor import base_map
from app.agent.interpret import interpret
from app.agent.witnesses import grade
from app.meta_api import SPARK_OPEN, spark_model
from app.sar import terrain
from app.sar.cues import Report
from app.sar.detection import DetectionModel
from app.sar.export import qgc_plan
from app.sar.interface import DroneState, FleetSpec
from app.sar.planners.greedy import ProbabilityGreedy
from app.sar.recorder import MAP_SIDE, _shrink
from app.sar.stats import lost_person_stats

MAX_BRIEF_CHARS = 2000
EXAMPLE_PLACE = "saint-ursanne"
EXAMPLE_BRIEF = ("Margaret, 81, has dementia and left her home in the old town about an hour ago, on foot. "
                 "A neighbour saw her walking along the river, heading east along the Doubs. "
                 "Her daughter says she used to walk to the chapel up on the hill. She walks slowly and tires quickly.")
EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "briefs" / "example.json"


def read_brief(place_name: str, text: str, drones: int = 1) -> dict:
    """Spark's hypotheses, the statistics-only map and the combined map (both MAP_SIDE x MAP_SIDE, 0-255), and the
    first sortie of each drone as map points and as a QGroundControl mission."""
    if not text.strip() or len(text) > MAX_BRIEF_CHARS:
        raise ValueError(f"a brief needs 1-{MAX_BRIEF_CHARS} characters")
    place, search = terrain.PLACES[place_name], terrain.search_map(place_name)
    stats, fleet = lost_person_stats(), FleetSpec(drones=drones)
    base = base_map(search, stats)
    reading = interpret(search, [Report(0.0, text)], base)
    witnesses = grade(text)
    planner = ProbabilityGreedy()
    planner.reset(search, fleet, DetectionModel.load(), stats, seed=0)
    planner.observe(reading.to_event())
    home = search.grid.centre_m(*search.base)
    drones_at_base = tuple(DroneState(i, *home, fleet.alt_m, fleet.endurance_s, False, ()) for i in range(drones))
    routes = [planner.plan(0.0, d, drones_at_base) for d in drones_at_base]
    return {
        "brief": text,
        "place": {"name": place_name, "title": place.title, "bounds": place.bounds(), "credit": terrain.CREDIT},
        "map": {"rows": search.grid.rows, "cols": search.grid.cols, "cell_m": search.grid.cell_m,
                "ipp": list(search.ipp), "side": MAP_SIDE},
        "statistics_only": _shrink(base), "combined": _shrink(planner.probability_map()),
        "hypotheses": [{"name": h.name, "weight": round(h.weight, 3), "reason": h.reason, "evidence": list(h.evidence)}
                       for h in reading.hypotheses],
        "rest_weight": round(reading.rest_weight, 3), "dropped": list(reading.dropped),
        "witnesses": witnesses.data["witnesses"],
        "sorties": [{"drone": i + 1, "route": [[round(y), round(x)] for y, x, _ in route],
                     "plan": qgc_plan(place, home, route, fleet.speed_mps)} for i, route in enumerate(routes)],
        "usd": round(reading.usd + witnesses.usd, 4), "cached": reading.cached and witnesses.cached,
        "model": "Muse Spark Open" if spark_model() == SPARK_OPEN else "Muse Spark",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--example", action="store_true", help="write the page's example brief")
    args = parser.parse_args()
    if args.example:
        result = read_brief(EXAMPLE_PLACE, EXAMPLE_BRIEF)
        EXAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXAMPLE_PATH.write_text(json.dumps(result), encoding="utf-8")
        for w in result["witnesses"]:
            print(f"{w['reliability']:<6}  {w['who']}: {w['claim']} ({w['why']})")
        for h in result["hypotheses"]:
            print(f"{h['weight']:.0%}  {h['name']}: {h['reason']}")
        print(f"rest {result['rest_weight']:.0%}; dropped {result['dropped']}; ${result['usd']:.4f}; wrote {EXAMPLE_PATH}")


if __name__ == "__main__":
    main()
