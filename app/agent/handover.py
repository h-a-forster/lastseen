"""When a drone finds someone, Muse Spark writes the handover for the ground team.

Code works out the facts (where the photo was taken, how far and which way from where she was last seen, the ground
there, the nearest path, what the photo check saw); Spark only words them for a radio or a text message. Code then
checks that every number in the message is one of the facts, so a wrong coordinate can't slip in.

    uv run python -m app.agent.handover data/runs/<race>.json ...   # adds a handover to each lane that found her
"""

import argparse
import json
import math
import re
from pathlib import Path

from app.meta_api import ModelCallError, spark_json
from app.sar import terrain
from app.sar.grid import Terrain

MAX_WORDS = 70
NUMBER_WORDS = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|"
                          r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b", re.IGNORECASE)
COMPASS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
DIRECTION = re.compile(r"\b(" + "|".join(sorted(COMPASS, key=len, reverse=True)) + r")\b")  # "north-east" before "north"

SYSTEM = """You write the handover a search drone pilot sends to the ground team when a photo shows the missing person.
Use only the facts given, at most 70 words, ready to read out on a radio: where to go (the photo centre's coordinates,
if given), how far and which way from where she was last seen, the ground there, the nearest path, what the photo
shows, and that the team should confirm on the ground. Copy every number exactly as written in the facts, in digits
and with its unit; never write a number as a word."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["message"],
          "properties": {"message": {"type": "string"}}}


def facts(run: dict, lane: dict) -> dict:
    """What the ground team needs, worked out from the recorded race; nothing here comes from a model."""
    m, found = run["map"], next(e for e in lane["events"] if e["type"] == "found")
    sighting = next(e for e in lane["events"] if e["type"] == "sighting" and e["found"])
    row, col = int(found["y_m"] // m["cell_m"]), int(found["x_m"] // m["cell_m"])
    last_y, last_x = ((v + 0.5) * m["cell_m"] for v in m["ipp"])
    north, east = last_y - found["y_m"], found["x_m"] - last_x
    out = {
        "found_after": f"{int(found['t_s'] // 3600)}:{int(found['t_s'] % 3600 // 60):02d}",
        "drone": found["drone"] + 1,
        "distance_from_last_seen": f"{round(math.hypot(north, east), -1):.0f} m",
        "direction_from_last_seen": COMPASS[round(math.degrees(math.atan2(east, north)) % 360 / 45) % 8],
        "ground": Terrain(int(m["terrain"][row * m["cols"] + col])).name.lower(),
        "nearest_path": _nearest_path(m, row, col),
        "photo_shows": sighting["spark"],
    }
    if run.get("place"):
        lat, lon = terrain.PLACES[run["place"]["name"]].latlon(found["y_m"], found["x_m"])
        out["photo_centre"] = f"{lat:.5f}, {lon:.5f}"
    return out


def _nearest_path(m: dict, row: int, col: int) -> str:
    trails = [divmod(i, m["cols"]) for i, c in enumerate(m["trails"]) if c == "1"]
    if not trails:
        return "no path on the map"
    metres = round(min(math.hypot(r - row, c - col) for r, c in trails) * m["cell_m"], -1)
    return "on a path" if metres == 0 else f"{metres:.0f} m away"


def write(found_facts: dict) -> dict:
    """Spark's handover message for these facts, checked; raises ModelCallError if it fails the checks."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Facts:\n" + json.dumps(found_facts, indent=1)}]
    answer = spark_json(messages, SCHEMA, "handover", max_tokens=3000)
    message = str(answer.data.get("message", "")).strip()
    check(message, found_facts)
    return {"text": message, "facts": found_facts, "usd": round(answer.usd, 5), "model": answer.model}


def check(message: str, found_facts: dict) -> None:
    """Every number in the message must appear in the facts, written in digits, any compass direction must be the one
    in the facts, and the message must be short enough to read out."""
    if not message or len(message.split()) > MAX_WORDS:
        raise ModelCallError(f"handover must be 1-{MAX_WORDS} words")
    if NUMBER_WORDS.search(message):
        raise ModelCallError("handover spells out a number; numbers must be digits so they can be checked")
    known = set(_numbers(json.dumps(found_facts)))
    invented = [n for n in _numbers(message) if n not in known]
    if invented:
        raise ModelCallError(f"handover has numbers that aren't in the facts: {invented}")
    wrong_way = [d for d in DIRECTION.findall(message.lower()) if d != found_facts["direction_from_last_seen"]]
    if wrong_way:
        raise ModelCallError(f"handover gives a direction that isn't in the facts: {wrong_way}")


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.:]\d+)?", text)


def add_to_run(path: Path) -> list[str]:
    """Add a handover note at the find to every lane that found her and has none yet; returns what was written."""
    run = json.loads(path.read_text(encoding="utf-8"))
    written = []
    for name, lane in run["lanes"].items():
        found = [e for e in lane["events"] if e["type"] == "found"]
        if not found or any(e.get("kind") == "handover" for e in lane["events"]):
            continue
        note = write(facts(run, lane))
        lane["events"].append({"type": "note", "t_s": found[0]["t_s"], "drone": found[0]["drone"], "kind": "handover",
                               "text": note["text"], "facts": note["facts"]})
        written.append(f"{name}: {note['text']}")
    path.write_text(json.dumps(run, separators=(",", ":")), encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    for path in parser.parse_args().runs:
        for line in add_to_run(path):
            print(f"{path.name} {line}")


if __name__ == "__main__":
    main()
