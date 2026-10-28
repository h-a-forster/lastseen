import json
import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import llm  # noqa: E402
from app.perception import pool as photo_pool  # noqa: E402
from app.sar import replay  # noqa: E402
from app.sar.bench import SEED_SETS  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="LastSeen")


@app.get("/api/health")
def health():
    return {"ok": True, "model": llm.current_model()}


RACE_PLANNERS = {"lawnmower", "spiral", "concentric", "sector", "greedy", "greedy+spark"}
_race_lock = threading.Lock()  # one race at a time: each takes a few seconds of CPU
MAX_STORED_RUNS = 200  # about 1 MB each
_summaries: dict[str, tuple[float, dict]] = {}  # file name -> (modified time, list entry), so listing doesn't reparse


class RaceRequest(BaseModel):
    seed: int = Field(ge=0, lt=SEED_SETS["heldout"].start)  # held-out scenarios stay unseen
    planners: list[str] = Field(min_length=1, max_length=replay.MAX_LANES)


@app.get("/api/runs")
def list_runs():
    """Recorded races, newest first, with when each planner found the person."""
    runs = []
    for path in sorted(replay.RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        mtime = path.stat().st_mtime
        if _summaries.get(path.name, (None,))[0] != mtime:
            data = json.loads(path.read_text(encoding="utf-8"))
            _summaries[path.name] = (mtime, {"id": data["id"], "seed": data["seed"], "category": data["category"],
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
    if len(list(replay.RUNS_DIR.glob("*.json"))) >= MAX_STORED_RUNS:
        raise HTTPException(507, "storage for recorded races is full")
    if not _race_lock.acquire(blocking=False):
        raise HTTPException(429, "another race is being recorded; try again in a few seconds")
    try:
        os.environ["MODEL_MODE"] = "replay"
        path = replay.save(replay.race(req.seed, req.planners))
    finally:
        _race_lock.release()
    return {"id": path.stem}


@app.get("/api/tiles/{tile_id}.jpg")
def get_tile(tile_id: str):
    """A real drone-photo tile from gate 1 (HERIDAL, CC BY 3.0), as shown in replays."""
    path = photo_pool.tile_jpeg(tile_id) if re.fullmatch(r"[0-9a-f]{16}", tile_id) else None
    if path is None:
        raise HTTPException(404, "no such tile")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


# Serves static/index.html at "/". Keep this last so /api routes win.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
