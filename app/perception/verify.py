"""Muse Spark as the second opinion on each SAM candidate.

SAM proposes; Spark looks at an enlarged close-up of each proposal, with the surrounding tile for context, and says
whether a person is there. A trained eye at base does the same with a tool's alarms: most rocks and logs that look like
a person from 60 m stop looking like one up close.
"""

import io
from dataclasses import dataclass

from PIL import Image

from app.meta_api import jpeg_data_url, spark_json

CLOSE_UP_PX = 128  # side of the square cut around the candidate's centre, in tile pixels
ENLARGE = 3

SYSTEM = (
    "You check alarms from a person detector for a drone search-and-rescue team. Photos are taken straight down from "
    "40-120 m over mountains, scrub and forest. From above a person is small: a head and shoulders, or a body lying "
    "down, in bright or dull clothes. Rocks, logs, stumps, bushes, shadows, bags and animals are not people. "
    "Missing a real person is worse than a false alarm, but a vague blob is not a person."
)
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["person", "confidence", "looks_like"],
    "properties": {
        "person": {"type": "boolean", "description": "a person is at the centre of the close-up"},
        "confidence": {"type": "number", "description": "0-1"},
        "looks_like": {"type": "string", "description": "a few words: what is at the centre"},
    },
}


@dataclass(frozen=True)
class Verdict:
    person: bool
    confidence: float
    looks_like: str
    usd: float
    cached: bool


def verify(tile: Image.Image, centre: tuple[float, float]) -> Verdict:
    """Spark's view on whether a person is at `centre` (x, y in tile pixels)."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"Image 1: the whole photo tile, for context. Image 2: a close-up of the flagged "
                                     f"spot, enlarged {ENLARGE}x. Is a person at the centre of image 2?"},
            {"type": "image_url", "image_url": {"url": jpeg_data_url(_jpeg(tile))}},  # shared by a tile's candidates,
            {"type": "image_url", "image_url": {"url": jpeg_data_url(_jpeg(close_up(tile, centre)))}},  # so cached
        ]},
    ]
    answer = spark_json(messages, SCHEMA, "verdict", effort="low", max_tokens=3000)
    d = answer.data
    return Verdict(bool(d["person"]), min(max(float(d["confidence"]), 0.0), 1.0), str(d["looks_like"])[:80],
                   answer.usd, answer.cached)


def close_up(tile: Image.Image, centre: tuple[float, float]) -> Image.Image:
    x, y = centre
    half = CLOSE_UP_PX / 2
    box = (int(x - half), int(y - half), int(x + half), int(y + half))  # may run past the edge; padded black
    return tile.crop(box).resize((CLOSE_UP_PX * ENLARGE,) * 2, Image.Resampling.LANCZOS)


def _jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()
