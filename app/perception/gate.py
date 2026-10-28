"""Gate 1: how often SAM 3.1 + Muse Spark find a person in a real drone photo, by how big the person looks.

    uv run python -m app.perception.gate --pilot           # 8 tiles, a few cents
    uv run python -m app.perception.gate --usd-cap 1.5     # the full gate

Altitude is simulated honestly: at scale s, a square of 1000*s photo pixels is shrunk into the same 1000 px tile,
which is what a camera s times higher would see (people s times smaller, s^2 times more ground per tile).
Each tile's result is appended to `data/perception/gate1.jsonl` when it finishes, so a rerun skips finished tiles,
and every model answer is cached, so rerunning the scoring costs nothing.
"""

import argparse
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from app.meta_api import ModelCallError, SpendCapReached, sam_find, spent_usd
from app.perception.heridal import Box, Photo, photos
from app.perception.verify import _jpeg, verify
from app.sar.metrics import wilson

TILE_PX = 1000
CONCEPT = "person"
VERIFY_FROM = 0.1  # SAM confidence from which a candidate is shown to Spark
VERIFY_TOP = 20  # at most this many candidates per tile go to Spark (SAM returned at most 16 in gate 1)
SIZE_BINS = (0, 16, 32, 64, math.inf)  # person's longest side in tile pixels
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)
OUT = Path(__file__).resolve().parents[2] / "data" / "perception"
RESULTS = OUT / "gate1.jsonl"


@dataclass(frozen=True)
class Tile:
    photo: Photo
    scale: int
    x0: int
    y0: int

    @property
    def key(self) -> str:
        return f"{self.photo.id}@{self.scale}:{self.x0},{self.y0}"

    def to_tile(self, box: Box) -> list[float]:
        side = TILE_PX * self.scale
        return [(box[0] - self.x0) * TILE_PX / side, (box[1] - self.y0) * TILE_PX / side,
                (box[2] - self.x0) * TILE_PX / side, (box[3] - self.y0) * TILE_PX / side]

    def people(self, fully: bool) -> list[Box]:
        """People wholly inside the tile (fully=True), or touching it at all."""
        side = TILE_PX * self.scale
        x1, y1 = self.x0 + side, self.y0 + side
        if fully:
            return [b for b in self.photo.people if b[0] >= self.x0 and b[1] >= self.y0 and b[2] <= x1 and b[3] <= y1]
        return [b for b in self.photo.people if b[0] < x1 and b[2] > self.x0 and b[1] < y1 and b[3] > self.y0]


def tiles_at(photo: Photo, scale: int, width: int = 4000, height: int = 3000) -> list[Tile]:
    """Squares of TILE_PX*scale photo pixels, spread evenly so together they cover the photo."""
    side = TILE_PX * scale
    return [Tile(photo, scale, x, y) for y in tile_starts(height, side) for x in tile_starts(width, side)]


def tile_starts(length: int, side: int) -> list[int]:
    """Where squares of `side` start along an edge of `length` so that, evenly spread, they cover it. A photo smaller
    than one square gets a single square (the crop runs past its edge, padded black)."""
    if length <= side:
        return [0]
    return [int(v) for v in np.linspace(0, length - side, math.ceil(length / side)).astype(int)]


def sample(all_photos: list[Photo], scale: int, n_pos: int, n_neg: int, seed: int = 0) -> list[Tile]:
    """n_pos tiles with at least one whole person and n_neg tiles with nobody in them, fixed by the seed."""
    rng = np.random.default_rng([seed, scale])
    tiles = [t for p in all_photos for t in tiles_at(p, scale)]
    order = rng.permutation(len(tiles))
    pos = [tiles[i] for i in order if tiles[i].people(fully=True)][:n_pos]
    neg = [tiles[i] for i in order if not tiles[i].people(fully=False)][:n_neg]
    return pos + neg


@lru_cache(maxsize=4)
def _open(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def render(tile: Tile) -> Image.Image:
    side = TILE_PX * tile.scale
    crop = _open(tile.photo.path).crop((tile.x0, tile.y0, tile.x0 + side, tile.y0 + side))
    return crop if tile.scale == 1 else crop.resize((TILE_PX, TILE_PX), Image.Resampling.LANCZOS)


def run_tile(tile: Tile) -> dict:
    """SAM on the tile, then Spark on its most confident candidates."""
    image = render(tile)
    segments, usd, _ = sam_find(_jpeg(image), CONCEPT)
    candidates = []
    for rank, s in enumerate(sorted(segments, key=lambda s: -s.confidence)):
        entry = {"box": [round(v, 1) for v in s.box], "c": round(s.confidence, 4), "spark": None}
        if rank < VERIFY_TOP and s.confidence >= VERIFY_FROM:
            v = verify(image, ((s.box[0] + s.box[2]) / 2, (s.box[1] + s.box[3]) / 2))
            entry["spark"] = {"person": v.person, "confidence": v.confidence, "looks_like": v.looks_like}
            usd += v.usd
        candidates.append(entry)
    return {"key": tile.key, "photo": tile.photo.id, "scale": tile.scale,
            "people": [tile.to_tile(b) for b in tile.people(fully=True)],
            "touching": [tile.to_tile(b) for b in tile.people(fully=False)],
            "candidates": candidates, "usd": round(usd, 5)}


def hits(candidate: list[float], person: list[float]) -> bool:
    """Search, not localisation: the candidate's centre falls on the person (with a margin), or the reverse."""
    cx, cy = (candidate[0] + candidate[2]) / 2, (candidate[1] + candidate[3]) / 2
    px, py = (person[0] + person[2]) / 2, (person[1] + person[3]) / 2
    mx, my = (person[2] - person[0]) / 2, (person[3] - person[1]) / 2
    return (person[0] - mx <= cx <= person[2] + mx and person[1] - my <= cy <= person[3] + my) or \
        (candidate[0] <= px <= candidate[2] and candidate[1] <= py <= candidate[3])


def kept(candidate: dict, threshold: float, with_spark: bool) -> bool:
    if candidate["c"] < threshold:
        return False
    return not with_spark or bool(candidate["spark"] and candidate["spark"]["person"])


def score(records: list[dict], threshold: float, with_spark: bool) -> dict:
    """Recall per person-size bin and false alarms per tile, for one confidence threshold."""
    found, total = np.zeros(len(SIZE_BINS) - 1, int), np.zeros(len(SIZE_BINS) - 1, int)
    false_alarms = 0
    for r in records:
        alarms = [c["box"] for c in r["candidates"] if kept(c, threshold, with_spark)]
        for person in r["people"]:
            size = max(person[2] - person[0], person[3] - person[1])
            b = np.searchsorted(SIZE_BINS, size, side="right") - 1
            total[b] += 1
            found[b] += any(hits(a, person) for a in alarms)
        false_alarms += sum(not any(hits(a, p) for p in r["touching"]) for a in alarms)
    bins = {f"{SIZE_BINS[i]:g}-{SIZE_BINS[i + 1]:g}px": [int(found[i]), int(total[i]), *wilson(int(found[i]), int(total[i]))]
            for i in range(len(found)) if total[i]}
    return {"threshold": threshold, "with_spark": with_spark, "recall_by_size": bins,
            "recall": [int(found.sum()), int(total.sum())], "false_alarms_per_tile": false_alarms / max(len(records), 1)}


def load() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(tiles: list[Tile], workers: int) -> None:
    done = {r["key"] for r in load()}
    todo = [t for t in tiles if t.key not in done]
    lock = threading.Lock()
    OUT.mkdir(parents=True, exist_ok=True)

    def one(tile: Tile) -> None:
        try:
            record = run_tile(tile)
        except SpendCapReached:
            raise
        except ModelCallError as e:  # one bad tile shouldn't stop the gate; it is retried on the next run
            print(f"  {tile.key}: {e}", flush=True)
            return
        with lock, RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  {tile.key}: {len(record['people'])} people, {len(record['candidates'])} candidates, "
              f"spent ${spent_usd():.3f}", flush=True)

    print(f"{len(todo)} tiles to run ({len(tiles) - len(todo)} already done)", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(one, t) for t in todo]:
            future.result()


def report(records: list[dict]) -> dict:
    summary = {"tiles": len(records), "by_scale": {}}
    for scale in sorted({r["scale"] for r in records}):
        rs = [r for r in records if r["scale"] == scale]
        summary["by_scale"][scale] = {
            "tiles": len(rs), "with_people": sum(bool(r["people"]) for r in rs),
            "sam": [score(rs, t, False) for t in THRESHOLDS], "sam+spark": [score(rs, t, True) for t in THRESHOLDS]}
    summary["all_scales"] = {"sam": [score(records, t, False) for t in THRESHOLDS],
                             "sam+spark": [score(records, t, True) for t in THRESHOLDS]}
    return summary


def print_report(summary: dict) -> None:
    for label, block in [*((f"scale {s}", b) for s, b in summary["by_scale"].items()), ("all scales", summary["all_scales"])]:
        print(f"\n{label}:")
        for method in ("sam", "sam+spark"):
            for s in block[method]:
                bins = "  ".join(f"{k} {f}/{n}" for k, (f, n, *_ci) in s["recall_by_size"].items())
                f, n = s["recall"]
                print(f"  {method:<9} c>={s['threshold']:<4}  recall {f}/{n} = {f / max(n, 1):.0%}  "
                      f"false alarms/tile {s['false_alarms_per_tile']:.2f}  | {bins}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true", help="3 positive + 1 empty tile at scales 1 and 3")
    parser.add_argument("--scales", default="1,2,3")
    parser.add_argument("--pos", type=int, default=40, help="tiles with people, per scale")
    parser.add_argument("--neg", type=int, default=20, help="empty tiles, per scale")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--usd-cap", type=float, default=1.5)
    args = parser.parse_args()
    os.environ["USD_CAP"] = str(args.usd_cap)
    scales, n_pos, n_neg = ((1, 3), 3, 1) if args.pilot else (tuple(map(int, args.scales.split(","))), args.pos, args.neg)
    all_photos = photos()
    tiles = [t for s in scales for t in sample(all_photos, s, n_pos, n_neg)]
    try:
        run(tiles, args.workers)
    except SpendCapReached as e:
        print(f"stopped: {e}")
    keys = {t.key for t in tiles}
    summary = report([r for r in load() if r["key"] in keys])
    print_report(summary)
    (OUT / ("gate1-pilot.json" if args.pilot else "gate1-summary.json")).write_text(json.dumps(summary, indent=1))
    print(f"\nspent this run ${spent_usd():.3f}")


if __name__ == "__main__":
    main()
