# LastSeen: notes for coding agents

Read [README.md](README.md) first, and [docs/TEAM.md](docs/TEAM.md) for the git workflow.

## Rules
- Meta's models are the product: SAM 3.1, Muse Spark and Muse Voice Transcribe via the Meta Model API. Don't add other
  model providers.
- Only use code, data and assets we have the rights to. Never commit `.env`, API keys, downloaded datasets or raw API
  responses (`data/` keeps them local; see `.gitignore`).
- Work on a branch and open a pull request; `main` is protected.
- Before a pull request: `uv run pytest`, and run the page and the bench path your change touches.
- Model calls cost money: estimate calls and cost before any loop or batch, and keep `USD_CAP` low while developing.

## Stack
- FastAPI in `app/main.py`; the page is `static/` (HTML, Tailwind CDN, plain JS). Run:
  `uv run uvicorn app.main:app --reload`. Add dependencies with `uv add`.
- Every Meta call goes through `app/meta_api.py`: retries per the API's error guide, a per-process `USD_CAP`, every
  answer cached in `data/cache/`, `MODEL_MODE=replay` for cache-only runs.

## Where things are
- `app/sar/`: the search simulator. Maps (`scenario.py` synthetic, `terrain.py` real places from OpenStreetMap), the
  hidden person (`walker.py`), the arena (`sim.py`), witness reports (`cues.py`), baselines (`planners/`), the bench
  (`bench.py`) and replays (`replay.py`, `recorder.py`).
- `app/sar/interface.py`: the contract a planner implements. Planners never see the truth.
- `app/planner/`: the team's planner (`ours` in the bench; `ours+spark` adds Spark's reading of the reports).
- `app/agent/`: Spark reads reports into weighted hypotheses (`interpret.py`, `layers.py`), the advisor that feeds them
  to any planner (`advisor.py`), and the live brief (`brief.py`).
- `app/perception/`: detection rates measured on HERIDAL photos (`gate.py`, `calibrate.py` → `data/pod_table.json`),
  the photo pool for imagery mode (`pool.py`) and the photo check (`triage.py`).
- `docs/evaluation.md` fixes the final evaluation in advance; `docs/parameters.md` sources every parameter.

## Conventions
- Tune on the `dev` seeds only; the `heldout` seeds are for the final run in `docs/evaluation.md`.
- Paid server endpoints stay off unless `LIVE_MODELS=1`.
- Small functions, clear names, brief comments only where intent isn't obvious, no dead code.
