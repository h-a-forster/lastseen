"""Real photo outcomes for the simulator: what SAM 3.1 + Muse Spark actually answered on a real drone photo of a person
the same size.

The fast simulator rolls dice against the detection table. In imagery mode, each time a drone photographs the missing
person, the arena instead picks a real labelled person from gate 1 whose size in pixels matches what the drone would
see at that altitude, and uses the pipeline's recorded answer for that person: found, or missed (and why). Every find
and every miss in a replay is then a real model answer on a real photo, served from the gate 1 results ($0).

The pool is the calibration data itself, so imagery mode reproduces the table's rates by construction; it makes the
replay concrete, it doesn't validate the table.
"""

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.perception.calibrate import NOMINAL_PERSON_M, SIZE_BINS_PX
from app.perception.gate import RESULTS, hits, load

TILES_DIR = Path(__file__).resolve().parents[2] / "data" / "tiles"
MIN_BIN = 5  # people needed in a size bin to draw from it; otherwise draw from the nearest sizes


@dataclass(frozen=True)
class Shot:
    """One real person in one real tile, and what the pipeline made of them."""

    tile: str  # tile id (see tile_id), for the image
    box: tuple[float, float, float, float]  # the person, in tile pixels
    size_px: float
    detected: bool
    sam: float | None  # SAM's confidence on its box over the person; None if SAM proposed nothing there
    spark: str | None  # what Spark said that box showed


def tile_id(key: str) -> str:
    """A short, URL-safe id for a gate tile key such as 'test_BRS_0001@2:0,1000'."""
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def person_px(gsd_cm: float) -> float:
    """How long a person looks, in pixels, at this ground resolution (the same conversion as the table)."""
    return NOMINAL_PERSON_M * 100 / gsd_cm


class PhotoPool:
    def __init__(self, records: list[dict]) -> None:
        shots, self.keys = [], {}
        for r in records:
            self.keys[tile_id(r["key"])] = r["key"]
            for person in r["people"]:
                over = [c for c in r["candidates"] if hits(c["box"], person)]
                best = max(over, key=lambda c: (bool(c["spark"] and c["spark"]["person"]), c["c"]), default=None)
                spark = best["spark"] if best else None
                shots.append(Shot(tile_id(r["key"]), tuple(person), max(person[2] - person[0], person[3] - person[1]),
                                  bool(spark and spark["person"]), best["c"] if best else None,
                                  spark["looks_like"] if spark else None))
        self.shots = sorted(shots, key=lambda s: s.size_px)
        self._sizes = np.array([s.size_px for s in self.shots])

    @classmethod
    def load(cls, path: Path = RESULTS) -> "PhotoPool | None":
        """The pool from gate 1's results, or None if gate 1 hasn't been run here."""
        return cls(load()) if path.exists() else None

    def shot(self, gsd_cm: float, rng: np.random.Generator) -> Shot:
        """A real person as big as one looks at `gsd_cm`, drawn from the same size bin the detection table uses."""
        size_px = person_px(gsd_cm)
        b = int(np.searchsorted(SIZE_BINS_PX, size_px, side="right")) - 1
        lo, hi = SIZE_BINS_PX[max(b, 0)], SIZE_BINS_PX[min(b + 1, len(SIZE_BINS_PX) - 1)]
        in_bin = np.flatnonzero((self._sizes >= lo) & (self._sizes < hi))
        if len(in_bin) < MIN_BIN:  # off the measured range: the nearest sizes
            in_bin = np.argsort(np.abs(np.log(self._sizes / size_px)))[:MIN_BIN]
        return self.shots[int(rng.choice(in_bin))]


@lru_cache(maxsize=1)
def shared() -> PhotoPool | None:
    return PhotoPool.load()


def tile_jpeg(tid: str) -> Path | None:
    """The tile's image, rendered from the local dataset on first use and kept in TILES_DIR. None if unknown, or if
    neither a rendered copy nor the dataset is here."""
    pool = shared()
    if pool is None or tid not in pool.keys:
        return None
    path = TILES_DIR / f"{tid}.jpg"
    if path.exists():
        return path
    from app.perception.gate import Tile, render
    from app.perception.heridal import photos
    from app.perception.verify import _jpeg

    photo_id, rest = pool.keys[tid].split("@")
    scale, xy = rest.split(":")
    photo = next((p for p in photos() if p.id == photo_id), None)
    if photo is None:
        return None
    x0, y0 = map(int, xy.split(","))
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_bytes(_jpeg(render(Tile(photo, int(scale), x0, y0)), quality=85))
    partial.replace(path)
    return path


def main() -> None:
    """Render every tile in the pool (so a host without the dataset can serve them) and print the pool by size."""
    pool = shared()
    if pool is None:
        raise SystemExit(f"no gate 1 results at {RESULTS}")
    rendered = sum(tile_jpeg(t) is not None for t in pool.keys)
    print(f"{rendered}/{len(pool.keys)} tiles in {TILES_DIR}")
    for lo, hi in zip(SIZE_BINS_PX, SIZE_BINS_PX[1:]):
        bin_shots = [s for s in pool.shots if lo <= s.size_px < hi]
        if bin_shots:
            print(f"  {lo:>3}-{hi:<5g} px: {sum(s.detected for s in bin_shots)}/{len(bin_shots)} found")
    print(json.dumps({"example_person_px_at_60m": round(person_px(1.65), 1)}))


if __name__ == "__main__":
    main()
