import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import llm  # noqa: E402
from app.meta_api import ModelCallError, SpendCapReached  # noqa: E402
from app.perception import pool as photo_pool  # noqa: E402
from app.perception import triage  # noqa: E402
from app.sar import replay  # noqa: E402
from app.sar.bench import SEED_SETS  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="LastSeen")


@app.get("/api/health")
def health():
    return {"ok": True, "model": llm.current_model()}


RACE_PLANNERS = {"lawnmower", "spiral", "concentric", "sector", "greedy", "greedy+spark"}
# One model job at a time (a race, or a photo check). Both set MODEL_MODE for the process, so they must not overlap.
_model_lock = threading.Lock()
MAX_STORED_RUNS = 200  # about 1 MB each
_summaries: dict[str, tuple[float, dict]] = {}  # file name -> (modified time, list entry), so listing doesn't reparse


class RaceRequest(BaseModel):
    seed: int = Field(ge=0, lt=SEED_SETS["heldout"].start)  # held-out scenarios stay unseen
    planners: list[str] = Field(min_length=1, max_length=replay.MAX_LANES)
    place: str | None = None


@app.get("/api/runs")
def list_runs():
    """Recorded races, newest first, with when each planner found the person."""
    runs = []
    for path in sorted(replay.RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        mtime = path.stat().st_mtime
        if _summaries.get(path.name, (None,))[0] != mtime:
            data = json.loads(path.read_text(encoding="utf-8"))
            _summaries[path.name] = (mtime, {"id": data["id"], "seed": data["seed"], "category": data["category"],
                                             "place": (data.get("place") or {}).get("title"),
                                             "found_s": {n: lane["result"]["found_s"] for n, lane in data["lanes"].items()}})
        runs.append(_summaries[path.name][1])
    return runs


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    path = replay.RUNS_DIR / f"{run_id}.json"
    if not re.fullmatch(r"[a-z0-9-]+", run_id) or not path.is_file():
        raise HTTPException(404, "no such run")
    return FileResponse(path, media_type="application/json")


@app.post("/api/runs")
def new_race(req: RaceRequest):
    """Record a new race. Spark readings come from the cache only: this public endpoint never spends credits, and a
    reading that isn't cached leaves that lane searching on statistics alone."""
    unknown = set(req.planners) - RACE_PLANNERS
    if unknown:
        raise HTTPException(422, f"unknown planners: {', '.join(sorted(unknown))}")
    if req.place is not None and req.place not in replay.PLACES:
        raise HTTPException(422, f"unknown place: {req.place}")
    if len(list(replay.RUNS_DIR.glob("*.json"))) >= MAX_STORED_RUNS:
        raise HTTPException(507, "storage for recorded races is full")
    if not _model_lock.acquire(blocking=False):
        raise HTTPException(429, "busy recording a race or checking a photo; try again in a few seconds")
    try:
        os.environ["MODEL_MODE"] = "replay"
        path = replay.save(replay.race(req.seed, req.planners, place=req.place))
    finally:
        _model_lock.release()
    return {"id": path.stem}


@app.get("/api/tiles/{tile_id}.jpg")
def get_tile(tile_id: str):
    """A real drone-photo tile from gate 1 (HERIDAL, CC BY 3.0), as shown in replays."""
    path = photo_pool.tile_jpeg(tile_id) if re.fullmatch(r"[0-9a-f]{16}", tile_id) else None
    if path is None:
        raise HTTPException(404, "no such tile")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


LIVE_TRIAGE = os.getenv("LIVE_TRIAGE") == "1"  # paid: off unless the server is set up for it
TRIAGE_PER_HOUR = 5  # photo checks per visitor address per hour
MAX_UPLOAD_BYTES = 20_000_000
_triage_times: dict[str, deque] = defaultdict(deque)


@app.get("/api/triage/example")
def triage_example():
    """A labelled HERIDAL photo already checked by SAM 3.1 + Spark, shown before anyone uploads a photo."""
    path = triage.EXAMPLE_DIR / "example.json"
    if not path.is_file():
        raise HTTPException(404, "no example on this server (python -m app.perception.triage --example)")
    return FileResponse(path, media_type="application/json")


@app.get("/api/triage/example.jpg")
def triage_example_photo():
    path = triage.EXAMPLE_DIR / "example.jpg"
    if not path.is_file():
        raise HTTPException(404, "no example photo")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/triage")
async def triage_photo(request: Request):
    """Check an uploaded photo (the raw image as the request body) with SAM 3.1 + Spark. The photo is not stored."""
    if not LIVE_TRIAGE:
        raise HTTPException(403, "live photo checks are switched off on this server")
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"photo is larger than {MAX_UPLOAD_BYTES // 1_000_000} MB")
    _rate_limit(request.client.host if request.client else "unknown")
    data = await request.body()
    try:
        image = await run_in_threadpool(triage.load, data)
    except (ValueError, OSError) as e:  # not an image, or too many pixels
        raise HTTPException(422, f"could not use this photo: {e}")
    if not _model_lock.acquire(blocking=False):
        raise HTTPException(429, "busy recording a race or checking a photo; try again in a few seconds")
    try:
        os.environ["MODEL_MODE"] = "live"
        return await run_in_threadpool(triage.triage, image)
    except SpendCapReached:
        raise HTTPException(503, "this server's spending limit for model calls is reached")
    except ModelCallError as e:
        raise HTTPException(502, f"the model call failed: {e}")
    finally:
        os.environ["MODEL_MODE"] = "replay"
        _model_lock.release()


def _rate_limit(address: str) -> None:
    times, now = _triage_times[address], time.monotonic()
    while times and now - times[0] > 3600:
        times.popleft()
    if len(times) >= TRIAGE_PER_HOUR:
        raise HTTPException(429, f"at most {TRIAGE_PER_HOUR} photo checks an hour; try again later")
    times.append(now)


# Serves static/index.html at "/". Keep this last so /api routes win.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
