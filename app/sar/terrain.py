"""Real search areas: OpenStreetMap land cover around a real place, rasterised into the same `SearchMap` as the
synthetic scenarios, so every planner, the bench and the replay run on it unchanged.

`fetch` downloads the place's features once from the Overpass API into `data/places/` (OpenStreetMap data, ODbL; not
committed). `search_map` paints them at 5 m (later classes win: scrub, forest, buildings, water) and gives each 25 m
cell its most common class; any cell a path or road crosses is a trail. The truth walker then walks on this map exactly
as it does on a synthetic one.

`visitor_place` builds a place around any point a visitor picks, refusing ones it can't map honestly. Where its region
was downloaded ahead in 0.1° tiles (`python -m app.sar.terrain S W N E`), the place is cut from them instead of waiting
on Overpass.

Not modelled: height. The map is flat, so slopes, cliffs and the valley sides don't slow the walker or change the
camera's ground resolution.
"""

import json
import math
import os
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw

from app.sar.grid import Grid, Terrain
from app.sar.scenario import TRACK_HOURS, Scenario, SearchMap, synthetic
from app.sar.walker import SNAP_M, WalkerParams, guided_track, truth_track

DATA_DIR = Path(os.getenv("DATA_DIR") or Path(__file__).resolve().parents[2] / "data")
PLACES_DIR = DATA_DIR / "places"
OSM_TILES_DIR = DATA_DIR / "osm-tiles"  # OpenStreetMap features for a region, downloaded ahead (`download_tiles`)
TILE_DEG = 0.1
# Public Overpass instances (wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances), asked all at once
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
CREDIT = "Map data © OpenStreetMap contributors (ODbL)"
M_PER_DEG = 111_195.0  # metres per degree of latitude on a 6,371 km sphere
PAINT_M = 5.0
MAX_LANDMARKS = 25
LANDMARK_SPACING_M = 150.0
LANDMARK_TOURISM = {"viewpoint", "attraction"}
VISITOR_PREFIX = "at-"  # names of places visitors pick
MIN_ELEMENTS = 40  # fewer OpenStreetMap features than this in 3 km: too sparse to tell land cover apart
SWISS_IMAGERY = (45.8, 47.85, 5.9, 10.55)  # south, north, west, east: roughly where swisstopo's aerial photos exist


@dataclass(frozen=True)
class Place:
    title: str
    lat: float  # the last-seen point, also the centre of the map
    lon: float
    side_m: float = 3000.0
    cell_m: float = 25.0
    aerial: bool = True  # swisstopo aerial photos cover it

    def bounds(self) -> dict[str, float]:
        """South, west, north, east edges in degrees."""
        half_lat = self.side_m / 2 / M_PER_DEG
        half_lon = self.side_m / 2 / (M_PER_DEG * math.cos(math.radians(self.lat)))
        return {"south": self.lat - half_lat, "west": self.lon - half_lon,
                "north": self.lat + half_lat, "east": self.lon + half_lon}

    def latlon(self, y_m: float, x_m: float) -> tuple[float, float]:
        """A map position (metres south of the north edge, east of the west edge) in degrees."""
        b = self.bounds()
        return b["north"] - y_m / M_PER_DEG, b["west"] + x_m / (M_PER_DEG * math.cos(math.radians(self.lat)))

    def position(self, lat: float, lon: float) -> tuple[float, float]:
        """The inverse of `latlon`: a point in degrees as (y_m, x_m) on this place's map."""
        b = self.bounds()
        return (b["north"] - lat) * M_PER_DEG, (lon - b["west"]) * M_PER_DEG * math.cos(math.radians(self.lat))


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


class MapDataUnavailable(Exception):
    """OpenStreetMap's servers didn't answer in time."""


def visitor_place(lat: float, lon: float) -> str:
    """Register the 3 km square around a visitor's point as a place, download it and check it can be searched; returns
    its name. The point is rounded to about 100 m, so a place is downloaded once. Raises ValueError for coasts, water
    and places OpenStreetMap maps too sparsely, and MapDataUnavailable when its servers are busy."""
    name, square = _visitor_square(lat, lon)
    known = name in PLACES
    PLACES[name] = PLACES.get(name) or square
    try:
        try:
            osm = fetch(name, attempts=1, timeout_s=90)  # a busy server can take minutes; the page asks once more
        except (httpx.HTTPError, ValueError) as e:  # ValueError: an answer that isn't JSON, or a timed-out query
            raise MapDataUnavailable("OpenStreetMap's servers are busy; try again in a minute") from e
        _check_searchable(name, osm)
    except Exception:
        if not known:
            del PLACES[name]
        raise
    return name


def visitor_spot(lat: float, lon: float, spot_lat: float, spot_lon: float) -> tuple[float, float]:
    """Where a point lies on the map of the visitor's place around (lat, lon), as (y_m, x_m); no download. Raises
    ValueError outside its 3 km square."""
    square = _visitor_square(lat, lon)[1]
    y_m, x_m = square.position(spot_lat, spot_lon)
    if not (0 <= y_m < square.side_m and 0 <= x_m < square.side_m):
        raise ValueError("the witness's spot is outside the 3 km search area; mark it inside the square")
    return y_m, x_m


def _visitor_square(lat: float, lon: float) -> tuple[str, Place]:
    """The name and square of a visitor's place, centred on the point rounded to about 100 m."""
    lat, lon = round(lat, 3), round(lon, 3)
    name = f"{VISITOR_PREFIX}{abs(lat) * 1000:.0f}{'n' if lat >= 0 else 's'}-{abs(lon) * 1000:.0f}{'e' if lon >= 0 else 'w'}"
    south, north, west, east = SWISS_IMAGERY
    return name, Place(f"{abs(lat):.3f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.3f}° {'E' if lon >= 0 else 'W'}", lat,
                       lon, aerial=south <= lat <= north and west <= lon <= east)


def visitor_seed(name: str) -> int:
    """The same hidden walk every time for one place (a development seed), so repeat visits reuse Spark's readings."""
    return zlib.crc32(name.encode()) % 100_000


def _check_searchable(name: str, osm: dict) -> None:
    if any(e.get("tags", {}).get("natural") == "coastline" for e in osm["elements"]):
        raise ValueError("coasts aren't supported yet: OpenStreetMap doesn't map the sea as an area, so it would "
                         "count as land")
    m = search_map(name)
    if m.terrain[m.ipp] == Terrain.WATER:
        raise ValueError("that point is in water; pick a point on land")
    if (m.terrain == Terrain.WATER).mean() > 0.5:
        raise ValueError("most of this area is water; pick a point further inland")
    if len(osm["elements"]) < MIN_ELEMENTS:
        raise ValueError("OpenStreetMap has too little detail here to build a search map (or it is open water); try "
                         "a spot with roads and paths nearby")


def fetch(name: str, attempts: int = 4, timeout_s: float = 120, wait_s: float = 10) -> dict:
    """The place's OpenStreetMap features, downloaded on first use and cached on disk."""
    path = PLACES_DIR / f"{name}.osm.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    b = PLACES[name].bounds()
    osm = _from_tiles(b) or _overpass(_query(b, timeout_s), attempts, timeout_s, wait_s)
    PLACES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(osm), encoding="utf-8")
    return osm


def _query(b: dict[str, float], timeout_s: float) -> str:
    return (f"[out:json][timeout:{round(timeout_s * 0.75)}][bbox:{b['south']},{b['west']},{b['north']},{b['east']}];"
            '(nwr[natural~"^(scrub|heath|wetland|wood|water)$"];nwr[landuse];way[waterway~"^(river|riverbank)$"];'
            "way[building];way[highway];nwr[aeroway];"
            'nwr[amenity=place_of_worship];nwr[historic];nwr[tourism~"^(viewpoint|attraction)$"];'
            'node[natural~"^(peak|cave_entrance)$"];node[place~"^(town|village|hamlet)$"];way[natural=coastline];);out geom;')


def _tiles(b: dict[str, float]) -> list[tuple[int, int]]:
    """The 0.1° tiles (row, column: south-west corner in tenths of a degree) that cover these bounds."""
    index = lambda deg: math.floor(round(deg / TILE_DEG, 6))  # 47.3 / 0.1 is 472.99999...
    rows = range(index(b["south"]), index(b["north"]) + 1)
    cols = range(index(b["west"]), index(b["east"]) + 1)
    return [(r, c) for r in rows for c in cols]


def _tile_path(row: int, col: int) -> Path:
    return OSM_TILES_DIR / f"{row}_{col}.osm.json"


def _from_tiles(b: dict[str, float]) -> dict | None:
    """The square cut from tiles downloaded ahead, or None unless they cover it all. Like Overpass's bbox query, it keeps
    every feature that reaches into the square, whole; one that crosses tiles is kept once."""
    paths = [_tile_path(*tile) for tile in _tiles(b)]
    if not all(path.exists() for path in paths):
        return None
    kept = {}
    for path in paths:
        for element in json.loads(path.read_text(encoding="utf-8"))["elements"]:
            if _reaches(element, b):
                kept[element["type"], element["id"]] = element
    order = {"node": 0, "way": 1, "relation": 2}  # Overpass's own order, so a cut map draws like a downloaded one
    return {"elements": sorted(kept.values(), key=lambda e: (order[e["type"]], e["id"]))}


def _reaches(element: dict, b: dict[str, float]) -> bool:
    if "bounds" in element:
        e = element["bounds"]
        return e["minlat"] <= b["north"] and e["maxlat"] >= b["south"] and e["minlon"] <= b["east"] and e["maxlon"] >= b["west"]
    return b["south"] <= element["lat"] <= b["north"] and b["west"] <= element["lon"] <= b["east"]


def download_tiles(south: float, west: float, north: float, east: float) -> None:
    """Download a region's tiles ahead of visitors, one at a time; tiles already on disk are skipped, so a rerun resumes."""
    OSM_TILES_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(row, col) for row in range(round(south / TILE_DEG), round(north / TILE_DEG))
            for col in range(round(west / TILE_DEG), round(east / TILE_DEG))]
    for i, (row, col) in enumerate(todo, 1):
        path = _tile_path(row, col)
        if not path.exists():
            edge = lambda i: round(i * TILE_DEG, 6)
            b = {"south": edge(row), "west": edge(col), "north": edge(row + 1), "east": edge(col + 1)}
            osm = _overpass(_query(b, 300), attempts=3, timeout_s=300, wait_s=30)
            path.write_text(json.dumps(osm), encoding="utf-8")
        print(f"tile {i}/{len(todo)} {path.name}: {path.stat().st_size / 1e6:.1f} MB", flush=True)


def _overpass(query: str, attempts: int, timeout_s: float, wait_s: float = 10) -> dict:
    """Public Overpass servers are often busy: one answers 502 at once while another takes minutes or times out. Each
    attempt asks every server at once and takes the first complete answer; attempts are retried with backoff."""
    for attempt in range(attempts):
        try:
            return _first_answer(query, timeout_s)
        except (httpx.HTTPError, ValueError):
            if attempt == attempts - 1:
                raise
            time.sleep(wait_s * (attempt + 1))
    raise AssertionError("unreachable")


def _first_answer(query: str, timeout_s: float) -> dict:
    pool = ThreadPoolExecutor(len(OVERPASS_URLS))
    asked = [pool.submit(_ask_overpass, url, query, timeout_s) for url in OVERPASS_URLS]
    try:
        for answer in as_completed(asked):
            if answer.exception() is None:
                return answer.result()
        raise asked[0].exception()  # every server failed
    finally:
        pool.shutdown(wait=False)  # the slower servers' answers are dropped when they come


def _ask_overpass(url: str, query: str, timeout_s: float) -> dict:
    response = httpx.post(url, data={"data": query}, timeout=timeout_s, headers={"User-Agent": "lastseen-scenario-builder/0.1"})
    response.raise_for_status()
    osm = response.json()
    if "runtime error" in osm.get("remark", ""):  # the server gave up part-way: an incomplete map must not win
        raise ValueError(osm["remark"])
    return osm


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
    centre = grid.cell_of(place.side_m / 2, place.side_m / 2)
    ipp = grid.nearest(terrain != Terrain.WATER, centre, SNAP_M) or centre  # a point on a river means its bank
    no_fly = np.zeros(grid.shape, bool)  # no airfield or heliport in either preset; drone rules allow the whole area
    return SearchMap(grid, terrain, trails, no_fly, ipp, ipp, category, _landmarks(osm, place))


def scenario(seed: int, name: str, category: str = "dementia", truth: WalkerParams | None = None,
             seen: tuple[tuple[float, float], float] | None = None) -> Scenario:
    """A hidden walk on the real map; the seed picks the walk, the map is always the same place. `seen` as in
    `scenario.synthetic`."""
    search, rng = search_map(name, category), np.random.default_rng(seed)
    track, heading = (truth_track(search, category, rng, TRACK_HOURS, truth) if seen is None
                      else guided_track(search, category, rng, TRACK_HOURS, *seen, truth))
    return Scenario(f"{name}-{seed}", search, track, heading)


def build(seed: int, place: str | None, category: str, size: int = 120, truth: WalkerParams | None = None,
          seen: tuple[tuple[float, float], float] | None = None) -> Scenario:
    """The scenario for a seed: on the named real place, or on a synthetic map of `size` cells when `place` is None."""
    return (scenario(seed, place, category, truth, seen) if place
            else synthetic(seed, size, category=category, truth=truth, seen=seen))


def _landmarks(osm: dict, place: Place) -> tuple[tuple[str, float, float], ...]:
    """Named places a witness might mention (chapels, gates, bridges, villages, the river), as (name, y_m, x_m) at
    their middle, or for the river its point nearest the last-seen point. Nearest first, at most MAX_LANDMARKS, and none
    within LANDMARK_SPACING_M of one of the same kind already kept, so the old town's gates and sights don't crowd
    out the hills."""
    to_m = _projector(place)
    centre = place.side_m / 2
    found: dict[str, tuple[float, float, float, str]] = {}
    for e in osm["elements"]:
        tags = e.get("tags", {})
        kind = _landmark_kind(tags)
        if not kind or "name" not in tags:
            continue
        points = [e] if e["type"] == "node" else e.get("geometry") or []
        xy = [tuple(v * PAINT_M for v in to_m(p)) for p in points if p]
        if not xy:
            continue
        if kind == "river":
            x, y = min(xy, key=lambda q: math.hypot(q[0] - centre, q[1] - centre))
        else:
            x, y = sum(q[0] for q in xy) / len(xy), sum(q[1] for q in xy) / len(xy)
        if not (0 <= x < place.side_m and 0 <= y < place.side_m):
            continue
        label, distance = f"{tags['name']} ({kind})", math.hypot(x - centre, y - centre)
        if label not in found or distance < found[label][0]:
            found[label] = (distance, y, x, kind)
    kept: list[tuple[str, float, float, str]] = []
    for label, (_, y, x, kind) in sorted(found.items(), key=lambda kv: kv[1][0]):
        if len(kept) < MAX_LANDMARKS and all(math.hypot(y - ky, x - kx) >= LANDMARK_SPACING_M
                                             for _, ky, kx, k in kept if k == kind):
            kept.append((label, round(y, 1), round(x, 1), kind))
    return tuple(k[:3] for k in kept)


def _landmark_kind(tags: dict) -> str | None:
    if tags.get("waterway") == "river":
        return "river"
    if tags.get("building") in {"chapel", "church"}:
        return tags["building"]
    if tags.get("amenity") == "place_of_worship":
        return "place of worship"
    if tags.get("bridge") == "yes":
        return "bridge"
    if tags.get("tourism") in LANDMARK_TOURISM:
        return tags["tourism"]
    for key in ("place", "historic", "natural"):
        if tags.get(key, "yes") != "yes":
            return tags[key].replace("_", " ")
    return None


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


if __name__ == "__main__":
    import sys

    download_tiles(*map(float, sys.argv[1:5]))
