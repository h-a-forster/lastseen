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

from app import llm, meta_api  # noqa: E402
from app.agent import brief  # noqa: E402
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
        raise HTTPException(429, "busy with another request; try again in a few seconds")
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


LIVE_MODELS = os.getenv("LIVE_MODELS") == "1"  # paid photo checks, briefs and transcripts: off unless set up for it
PAID_PER_HOUR = 10  # paid requests per visitor address per hour, all kinds together
MAX_UPLOAD_BYTES = 20_000_000
_paid_times: dict[str, deque] = defaultdict(deque)


class BriefRequest(BaseModel):
    place: str
    text: str = Field(min_length=1, max_length=brief.MAX_BRIEF_CHARS)


@app.get("/api/triage/example")
def triage_example():
    """A labelled HERIDAL photo already checked by SAM 3.1 + Spark, shown before anyone uploads a photo."""
    return _example_file(triage.EXAMPLE_DIR / "example.json", "python -m app.perception.triage --example")


@app.get("/api/triage/example.jpg")
def triage_example_photo():
    path = triage.EXAMPLE_DIR / "example.jpg"
    if not path.is_file():
        raise HTTPException(404, "no example photo")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/triage")
async def triage_photo(request: Request):
    """Check an uploaded photo (the raw image as the request body) with SAM 3.1 + Spark. The photo is not stored."""
    data = await _paid_upload(request)
    try:
        image = await run_in_threadpool(triage.load, data)
    except (ValueError, OSError) as e:  # not an image, or too many pixels
        raise HTTPException(422, f"could not use this photo: {e}")
    return await _live(triage.triage, image)


@app.get("/api/brief/example")
def brief_example():
    """A brief for the Saint-Ursanne preset, already read by Spark."""
    return _example_file(brief.EXAMPLE_PATH, "python -m app.agent.brief --example")


@app.post("/api/brief")
async def read_brief(req: BriefRequest, request: Request):
    """Spark reads a brief into hypotheses on a real place's map; the first sorties come back as QGroundControl plans."""
    if req.place not in replay.PLACES:
        raise HTTPException(422, f"unknown place: {req.place}")
    _check_paid(request)
    return await _live(brief.read_brief, req.place, req.text)


@app.post("/api/transcribe")
async def transcribe(request: Request):
    """Muse Voice Transcribe on a recorded brief (16 kHz mono 16-bit WAV as the request body). The audio isn't stored."""
    data = await _paid_upload(request)
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HTTPException(422, "send a WAV recording")
    text, usd, cached = await _live(meta_api.transcribe, data)
    return {"transcript": text, "usd": round(usd, 5), "cached": cached}


def _example_file(path: Path, how: str):
    if not path.is_file():
        raise HTTPException(404, f"no example on this server ({how})")
    return FileResponse(path, media_type="application/json")


async def _paid_upload(request: Request) -> bytes:
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"upload is larger than {MAX_UPLOAD_BYTES // 1_000_000} MB")
    _check_paid(request)
    return await request.body()


def _check_paid(request: Request) -> None:
    """Paid requests need LIVE_MODELS=1 on the server and stay under PAID_PER_HOUR per visitor address."""
    if not LIVE_MODELS:
        raise HTTPException(403, "live model calls are switched off on this server")
    times, now = _paid_times[request.client.host if request.client else "unknown"], time.monotonic()
    while times and now - times[0] > 3600:
        times.popleft()
    if len(times) >= PAID_PER_HOUR:
        raise HTTPException(429, f"at most {PAID_PER_HOUR} model requests an hour; try again later")
    times.append(now)


async def _live(fn, *args):
    """Run a paid model job: one at a time, live calls allowed only while it runs, failures as clear HTTP errors."""
    if not _model_lock.acquire(blocking=False):
        raise HTTPException(429, "busy with another request; try again in a few seconds")
    try:
        os.environ["MODEL_MODE"] = "live"
        return await run_in_threadpool(fn, *args)
    except SpendCapReached:
        raise HTTPException(503, "this server's spending limit for model calls is reached")
    except ModelCallError as e:
        raise HTTPException(502, f"the model call failed: {e}")
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.environ["MODEL_MODE"] = "replay"
        _model_lock.release()


# Serves static/index.html at "/". Keep this last so /api routes win.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
