"""Gate 1 again with Muse Spark Open: the same SAM 3.1 candidates, each checked by the open-weight model instead.

    SPARK_BACKEND=open uv run python -m app.perception.gate_open --usd-cap 3

SAM's answers and gate 1's tiles come from gate 1, so only Spark Open's checks are new calls. Each tile is appended to
`data/perception/gate1-open.jsonl` when it finishes, so a rerun skips finished tiles; with every answer cached, a
rerun in MODEL_MODE=replay prints the same comparison for $0.
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from app.meta_api import ModelCallError, spent_usd
from app.perception.gate import OUT, THRESHOLDS, VERIFY_FROM, Tile, load, render, sample, score
from app.perception.heridal import photos
from app.perception.verify import verify

RESULTS = OUT / "gate1-open.jsonl"


def recheck(tile: Tile, record: dict) -> dict:
    """The gate 1 record with every candidate that Muse Spark saw checked again by the backend in SPARK_BACKEND. A
    failed check counts as "not a person", so errors can only lower Spark Open's recall."""
    image = render(tile)
    candidates = []
    for c in record["candidates"]:
        if c["spark"] is not None:
            b = c["box"]
            try:
                v = verify(image, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
                c = {**c, "spark": {"person": v.person, "confidence": v.confidence, "looks_like": v.looks_like}}
            except ModelCallError as e:
                c = {**c, "spark": {"person": False, "confidence": 0, "looks_like": f"error: {e}"[:80]}}
        candidates.append(c)
    return {**record, "candidates": candidates}


def load_open() -> dict[str, dict]:
    if not RESULTS.exists():
        return {}
    records = (json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["key"]: r for r in records}


def run(tiles: list[Tile], spark: dict[str, dict], workers: int) -> None:
    done = load_open()
    todo = [t for t in tiles if t.key not in done]
    lock = threading.Lock()

    def one(tile: Tile) -> None:
        record = recheck(tile, spark[tile.key])
        with lock, RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  {tile.key}: spent ${spent_usd():.3f}", flush=True)

    print(f"{len(todo)} tiles to run ({len(tiles) - len(todo)} already done)", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(one, t) for t in todo]:
            future.result()


def compare(spark: list[dict], spark_open: list[dict]) -> None:
    """Recall and false alarms per tile for both models, per scale and overall, at the threshold the product uses."""
    threshold = THRESHOLDS[THRESHOLDS.index(VERIFY_FROM)]
    scales = sorted({r["scale"] for r in spark})
    for label, keep in [*((f"scale {s}", lambda r, s=s: r["scale"] == s) for s in scales), ("all scales", None)]:
        print(f"\n{label}:")
        for name, records in (("Muse Spark", spark), ("Muse Spark Open", spark_open)):
            s = score([r for r in records if keep is None or keep(r)], threshold, with_spark=True)
            f, n = s["recall"]
            print(f"  {name:<16} found {f}/{n} = {f / max(n, 1):.0%}  false alarms/tile {s['false_alarms_per_tile']:.2f}")
    errors = sum(str(c["spark"]["looks_like"]).startswith("error")
                 for r in spark_open for c in r["candidates"] if c["spark"])
    print(f"\nSpark Open checks that failed (counted as misses): {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--usd-cap", type=float, default=3.0)
    args = parser.parse_args()
    if os.getenv("SPARK_BACKEND") != "open":
        raise SystemExit("set SPARK_BACKEND=open: this gate checks gate 1's candidates with Muse Spark Open")
    os.environ["USD_CAP"] = str(args.usd_cap)
    spark = {r["key"]: r for r in load()}
    tiles = [t for s in (1, 2, 3) for t in sample(photos(), s, 40, 20) if t.key in spark]
    run(tiles, spark, args.workers)
    spark_open = load_open()
    compare([spark[t.key] for t in tiles], [spark_open[t.key] for t in tiles])
    print(f"spent this run ${spent_usd():.3f}")


if __name__ == "__main__":
    main()
