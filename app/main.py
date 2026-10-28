import json
import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import llm  # noqa: E402
from app.sar import replay  # noqa: E402

SYSTEM_PROMPT = "You are a helpful assistant. Be concise."  # change per project

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="hackathon-starter")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.get("/api/health")
def health():
    return {"ok": True, "model": llm.current_model()}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(m.model_dump() for m in req.messages)]

    async def generate():
        try:
            async for text in llm.stream(messages):
                yield text
        except Exception as e:  # surface provider errors in the UI instead of a dead stream
            yield f"\n\n[error: {type(e).__name__}]"
            raise

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


RACE_PLANNERS = {"lawnmower", "spiral", "concentric", "sector", "greedy", "greedy+spark"}
_race_lock = threading.Lock()  # one race at a time: each takes a few seconds of CPU


class RaceRequest(BaseModel):
    seed: int = Field(ge=0, lt=1_000_000)
    planners: list[str] = Field(min_length=1, max_length=replay.MAX_LANES)


@app.get("/api/runs")
def list_runs():
    """Recorded races, newest first, with when each planner found the person."""
    runs = []
    for path in sorted(replay.RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = json.loads(path.read_text(encoding="utf-8"))
        runs.append({"id": data["id"], "seed": data["seed"], "category": data["category"],
                     "found_s": {name: lane["result"]["found_s"] for name, lane in data["lanes"].items()}})
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
    if not _race_lock.acquire(blocking=False):
        raise HTTPException(429, "another race is being recorded; try again in a few seconds")
    try:
        os.environ["MODEL_MODE"] = "replay"
        path = replay.save(replay.race(req.seed, req.planners))
    finally:
        _race_lock.release()
    return {"id": path.stem}


# Serves static/index.html at "/". Keep this last so /api routes win.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
