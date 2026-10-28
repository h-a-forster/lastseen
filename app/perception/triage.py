"""Triage: one drone photo in, a short list of spots for a person to check.

The photo is cut into the same 1000 px tiles the detection rates were measured on (`gate.py`), SAM 3.1 proposes every
person it sees in each tile, and Muse Spark checks the most confident proposals on a close-up. A reviewer then looks
at the spots Spark accepted (alarms) instead of the whole photo; rejected proposals stay listed with Spark's reason,
so a reviewer can overrule it.

    uv run python -m app.perception.triage photo.jpg [more.jpg ...] --usd-cap 0.2
    uv run python -m app.perception.triage --example    # the page's example, from cached answers only

Each photo costs one SAM call per tile (12 for a 4000 x 3000 photo) plus one Spark call per checked proposal, at most
MAX_CHECKS. Answers are cached, so the same photo again is free.
"""

import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from app.meta_api import sam_find, spent_usd
from app.perception.gate import CONCEPT, TILE_PX, VERIFY_FROM, tile_starts
from app.perception.heridal import CREDIT, photos
from app.perception.verify import _jpeg, verify

MAX_CHECKS = 30  # Spark checks per photo, most confident proposals first; bounds the cost of a crowded photo
MAX_PIXELS = 24_000_000  # larger photos are refused rather than tiled into dozens of calls
EXAMPLE_PHOTO = "test_GRO_0008"  # HERIDAL test photo shown on the page before anyone uploads one
EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "triage"
EXAMPLE_WIDTH_PX = 2000  # the page shows a smaller copy; boxes are stored in the original photo's pixels


def load(data: bytes) -> Image.Image:
    """A photo from its file bytes, upright as a browser shows it (phones and some drones store rotation in EXIF)."""
    with Image.open(io.BytesIO(data)) as im:
        if im.width * im.height > MAX_PIXELS:
            raise ValueError(f"photo is {im.width} x {im.height}; at most {MAX_PIXELS / 1e6:.0f} megapixels")
        return ImageOps.exif_transpose(im).convert("RGB")


def triage(photo: Image.Image, threads: int = 4) -> dict:
    """Every SAM proposal in the photo (boxes in photo pixels, likeliest first), Spark's verdict on the likeliest
    ones, how many are alarms, and what it cost."""
    tiles = [(x, y, photo.crop((x, y, x + TILE_PX, y + TILE_PX)))
             for y in tile_starts(photo.height, TILE_PX) for x in tile_starts(photo.width, TILE_PX)]
    with ThreadPoolExecutor(threads) as pool:
        found = list(pool.map(lambda tile: sam_find(_jpeg(tile[2]), CONCEPT), tiles))
    proposals = sorted(((tile, s) for tile, (segments, _, _) in zip(tiles, found) for s in segments),
                       key=lambda p: -p[1].confidence)
    checked = [p for p in proposals if p[1].confidence >= VERIFY_FROM][:MAX_CHECKS]
    with ThreadPoolExecutor(threads) as pool:
        verdicts = list(pool.map(lambda p: verify(p[0][2], _centre(p[1].box)), checked))
    candidates = []
    for i, ((x0, y0, _), s) in enumerate(proposals):
        v = verdicts[i] if i < len(verdicts) else None
        candidates.append({
            "box": [round(s.box[0] + x0, 1), round(s.box[1] + y0, 1), round(s.box[2] + x0, 1), round(s.box[3] + y0, 1)],
            "sam": round(s.confidence, 3),
            "spark": None if v is None else {"person": v.person, "confidence": v.confidence, "looks_like": v.looks_like},
            "alarm": bool(v and v.person),
        })
    usd = sum(f[1] for f in found) + sum(v.usd for v in verdicts)
    return {"width": photo.width, "height": photo.height, "tiles": len(tiles), "candidates": candidates,
            "alarms": sum(c["alarm"] for c in candidates), "usd": round(usd, 4),
            "cached": all(f[2] for f in found) and all(v.cached for v in verdicts)}


def _centre(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def write_example() -> Path:
    """The example result and a web-sized copy of its photo, for hosts without the dataset. Cached answers only."""
    os.environ["MODEL_MODE"] = "replay"
    photo = next(p for p in photos("test") if p.id == EXAMPLE_PHOTO)
    image = load(photo.path.read_bytes())
    result = {**triage(image), "name": f"HERIDAL {photo.id}", "credit": CREDIT,
              "people_labelled": [[round(v) for v in box] for box in photo.people]}
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    image.resize((EXAMPLE_WIDTH_PX, round(image.height * EXAMPLE_WIDTH_PX / image.width))).save(
        EXAMPLE_DIR / "example.jpg", quality=85)
    path = EXAMPLE_DIR / "example.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos", nargs="*", type=Path)
    parser.add_argument("--example", action="store_true", help="write the page's example from cached answers")
    parser.add_argument("--usd-cap", type=float, default=0.2)
    parser.add_argument("--json", type=Path, help="write every photo's result here")
    args = parser.parse_args()
    if args.example:
        print(f"wrote {write_example()}")
        return
    os.environ["USD_CAP"] = str(args.usd_cap)
    results = {}
    for path in args.photos:
        result = triage(load(path.read_bytes()))
        results[path.name] = result
        print(f"{path.name}: {len(result['candidates'])} proposals, {result['alarms']} alarms "
              f"({result['tiles']} tiles, ${result['usd']:.3f}{', cached' if result['cached'] else ''})")
        for c in result["candidates"]:
            if c["spark"]:
                print(f"  {'ALARM' if c['alarm'] else 'no   '} sam {c['sam']:.2f}  {c['spark']['looks_like']}  {c['box']}")
    print(f"spent ${spent_usd():.3f} in this run")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
