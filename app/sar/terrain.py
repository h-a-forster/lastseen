"""Real search areas: OpenStreetMap land cover around a real place, rasterised into the same `SearchMap` as the
synthetic scenarios, so every planner, the bench and the replay run on it unchanged.

`fetch` downloads the place's features once from the Overpass API into `data/places/` (OpenStreetMap data, ODbL; not
committed). `search_map` paints them at 5 m (later classes win: scrub, forest, buildings, water) and gives each 25 m
cell its most common class; any cell a path or road crosses is a trail. The truth walker then walks on this map exactly
as it does on a synthetic one.

Not modelled: height. The map is flat, so slopes, cliffs and the valley sides don't slow the walker or change the
camera's ground resolution.
"""

import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw

from app.sar.grid import Grid, Terrain
from app.sar.scenario import TRACK_HOURS, Scenario, SearchMap, synthetic
from app.sar.walker import WalkerParams, truth_track

PLACES_DIR = Path(__file__).resolve().parents[2] / "data" / "places"
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
CREDIT = "Map data © OpenStreetMap contributors (ODbL)"
M_PER_DEG = 111_195.0  # metres per degree of latitude on a 6,371 km sphere
PAINT_M = 5.0


@dataclass(frozen=True)
class Place:
    title: str
    lat: float  # the last-seen point, also the centre of the map
    lon: float
    side_m: float = 3000.0
    cell_m: float = 25.0

    def bounds(self) -> dict[str, float]:
        """South, west, north, east edges in degrees."""
        half_lat = self.side_m / 2 / M_PER_DEG
        half_lon = self.side_m / 2 / (M_PER_DEG * math.cos(math.radians(self.lat)))
        return {"south": self.lat - half_lat, "west": self.lon - half_lon,
                "north": self.lat + half_lat, "east": self.lon + half_lon}


PLACES = {
    "saint-ursanne": Place("Saint-Ursanne, Doubs valley (Switzerland)", 47.36495, 7.15520),
}

# OpenStreetMap tags to terrain classes, in painting order: later classes cover earlier ones.
AREA_TAGS = [
    (Terrain.SCRUB, {"natural": {"scrub", "heath", "wetland"}}),
    (Terrain.FOREST, {"natural": {"wood"}, "landuse": {"forest"}}),
    (Terrain.BUILT, {"landuse": {"residential", "industrial", "commercial", "retail", "farmyard", "railway"},
                     "building": None}),
    (Terrain.WATER, {"natural": {"water"}, "landuse": {"basin", "reservoir"}, "waterway": {"riverbank"}}),
]
RIVER_WIDTH_M = 20.0
WALKABLE_WAYS = {"track", "path", "footway", "bridleway", "cycleway", "steps", "pedestrian", "living_street",
                 "residential", "service", "unclassified", "tertiary", "secondary", "primary"}  # not trunk or motorway


def fetch(name: str) -> dict:
    """The place's OpenStreetMap features, downloaded on first use and cached on disk."""
    path = PLACES_DIR / f"{name}.osm.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    b = PLACES[name].bounds()
    query = (f"[out:json][timeout:90][bbox:{b['south']},{b['west']},{b['north']},{b['east']}];"
             '(nwr[natural~"^(scrub|heath|wetland|wood|water)$"];nwr[landuse];way[waterway~"^(river|riverbank)$"];'
             "way[building];way[highway];nwr[aeroway];);out geom;")
    osm = _overpass(query)
    PLACES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(osm), encoding="utf-8")
    return osm


def _overpass(query: str, attempts: int = 4) -> dict:
    """Public Overpass servers often answer 429/502/504 when busy: retry with backoff, alternating two instances."""
    for attempt in range(attempts):
        try:
            response = httpx.post(OVERPASS_URLS[attempt % len(OVERPASS_URLS)], data={"data": query}, timeout=120,
                                  headers={"User-Agent": "lastseen-scenario-builder/0.1"})
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


@lru_cache(maxsize=4)
def search_map(name: str, category: str = "dementia") -> SearchMap:
    place = PLACES[name]
    osm = fetch(name)
    side = round(place.side_m / place.cell_m)
    to_px = _projector(place)
    paint = Image.new("L", (side * round(place.cell_m / PAINT_M),) * 2, int(Terrain.OPEN))
    for terrain, tags in AREA_TAGS:
        for element in osm["elements"]:
            if _matches(element.get("tags", {}), tags):
                _paint_area(paint, element, to_px, int(terrain))
    draw = ImageDraw.Draw(paint)
    for element in _ways(osm, "waterway", {"river"}):
        draw.line([to_px(p) for p in element["geometry"]], fill=int(Terrain.WATER), width=round(RIVER_WIDTH_M / PAINT_M))
    terrain = _majority(np.asarray(paint), side)
    trails = _trails(osm, to_px, side, place)
    grid = Grid(side, side, place.cell_m)
    ipp = grid.cell_of(place.side_m / 2, place.side_m / 2)
    no_fly = np.zeros(grid.shape, bool)  # no airfield or heliport in either preset; drone rules allow the whole area
    return SearchMap(grid, terrain, trails, no_fly, ipp, ipp, category)


def scenario(seed: int, name: str, category: str = "dementia", truth: WalkerParams | None = None) -> Scenario:
    """A hidden walk on the real map; the seed picks the walk, the map is always the same place."""
    search = search_map(name, category)
    track, heading = truth_track(search, category, np.random.default_rng(seed), TRACK_HOURS, truth)
    return Scenario(f"{name}-{seed}", search, track, heading)


def build(seed: int, place: str | None, category: str, size: int = 120, truth: WalkerParams | None = None) -> Scenario:
    """The scenario for a seed: on the named real place, or on a synthetic map of `size` cells when `place` is None."""
    return scenario(seed, place, category, truth) if place else synthetic(seed, size, category=category, truth=truth)


def _projector(place: Place):
    """Latitude/longitude to paint pixels (row 0 at the north edge). Equirectangular is exact enough over 3 km."""
    b = place.bounds()
    per_lon = M_PER_DEG * math.cos(math.radians(place.lat))
    return lambda p: ((p["lon"] - b["west"]) * per_lon / PAINT_M, (b["north"] - p["lat"]) * M_PER_DEG / PAINT_M)


def _matches(tags: dict, wanted: dict) -> bool:
    return any(key in tags and (values is None or tags[key] in values) for key, values in wanted.items())


def _ways(osm: dict, key: str, values: set[str]):
    return (e for e in osm["elements"] if e["type"] == "way" and e.get("tags", {}).get(key) in values)


def _paint_area(paint: Image.Image, element: dict, to_px, value: int) -> None:
    """Fill a closed way, or a multipolygon relation with its holes, with `value`."""
    if element["type"] == "way":
        if len(element["geometry"]) >= 3:
            ImageDraw.Draw(paint).polygon([to_px(p) for p in element["geometry"]], fill=value)
        return
    if element["type"] != "relation":
        return
    mask = Image.new("1", paint.size, 0)
    draw = ImageDraw.Draw(mask)
    for role, fill in (("outer", 1), ("inner", 0)):
        for ring in _rings([m["geometry"] for m in element.get("members", [])
                            if m.get("role") == role and m.get("geometry")]):
            if len(ring) >= 3:
                draw.polygon([to_px(p) for p in ring], fill=fill)
    paint.paste(value, mask=mask)


def _rings(parts: list[list[dict]]) -> list[list[dict]]:
    """Join a multipolygon's member ways end to end into closed rings (OSM splits long boundaries into many ways)."""
    key = lambda p: (p["lat"], p["lon"])
    parts, rings = [list(part) for part in parts], []
    while parts:
        ring = parts.pop()
        while key(ring[0]) != key(ring[-1]):
            for i, part in enumerate(parts):
                if key(part[0]) == key(ring[-1]):
                    ring += parts.pop(i)[1:]
                    break
                if key(part[-1]) == key(ring[-1]):
                    ring += parts.pop(i)[::-1][1:]
                    break
            else:
                break  # a boundary with a member missing; the polygon closes it with a straight edge
        rings.append(ring)
    return rings


def _majority(paint: np.ndarray, side: int) -> np.ndarray:
    k = paint.shape[0] // side
    blocks = paint.reshape(side, k, side, k).transpose(0, 2, 1, 3).reshape(side, side, k * k)
    counts = np.stack([(blocks == t).sum(axis=2) for t in Terrain], axis=2)
    return counts.argmax(axis=2).astype(np.int8)


def _trails(osm: dict, to_px, side: int, place: Place) -> np.ndarray:
    """Cells any walkable path or road crosses, drawn one paint pixel wide."""
    k = round(place.cell_m / PAINT_M)
    lines = Image.new("1", (side * k,) * 2, 0)
    draw = ImageDraw.Draw(lines)
    for way in _ways(osm, "highway", WALKABLE_WAYS):
        draw.line([to_px(p) for p in way["geometry"]], fill=1)
    return np.asarray(lines).reshape(side, k, side, k).any(axis=(1, 3))
