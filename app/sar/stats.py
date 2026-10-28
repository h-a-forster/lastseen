"""Published lost-person statistics, shared by the arena and planners. Every number in the file has a source."""

import json
from functools import cache
from pathlib import Path

STATS_PATH = Path(__file__).resolve().parents[2] / "data" / "lost_person_stats.json"


@cache
def lost_person_stats() -> dict:
    """The parsed statistics file. Shared and cached: copy before changing it."""
    return json.loads(STATS_PATH.read_text(encoding="utf-8"))
