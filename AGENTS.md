# Hackathon mode: Meta Global AI Developer Hackathon

A team hackathon project. Submission due **Wed 4 Nov, 21:00 ET**. Optimise for a working, impressive demo.
Read [docs/HACKATHON.md](docs/HACKATHON.md) (rules and judging) and [docs/TEAM.md](docs/TEAM.md) (git workflow) before proposing or building anything.

## Hard rules (break these and we're disqualified)
- **Meta models must be central**: Muse Spark, Muse Wonder, Muse Image, Muse Voice Transcribe or SAM 3.1 via the Meta Model API, or the open-weight Muse Glimmer / Muse Spark Open. Not one swappable LLM call.
- **New work only.** Commit `be7036a` (the starter boilerplate) is the only code written before the event. Never paste in code from older projects.
- Stay off the anti-project list in docs/HACKATHON.md (medical advice, basic RAG, simple image analysers, and so on).
- Only use code, data and assets we have the rights to. Never commit `.env` or print API keys.

## Judging
Impact 30% · Creative use of Meta tooling 25% · Demo 25% (live plus a 3-minute video) · Technical depth 20%.
When choosing what to build next, ask which of these it moves. Features that don't show in the demo are low priority.

## Rules for agents
- Speed over polish in code; polish in the UI and the demo path.
- No tests, linters or refactors unless asked. Verify by running the app and hitting the endpoint.
- Hardcode or fake anything off the demo path.
- **Git:** work on your own branch (`name/feature`), never commit to `main`. Commit small; pull `main` into your branch often. See docs/TEAM.md.
- Stay inside the area of the task you were given; don't edit other people's files without saying so.
- If something takes more than ~20 minutes to fix, stop and say so.

## Stack
- Backend: FastAPI in `app/main.py`. Run: `uv run uvicorn app.main:app --reload`
- Every model call goes through `app/llm.py` (`complete()` / `stream()`, LiteLLM). `MODEL=meta/muse-spark-1.3` routes to the Meta Model API
  (OpenAI-compatible, base `https://api.meta.ai/v1`, key `MODEL_API_KEY`). `MAX_LLM_CALLS` caps calls per run to protect credits.
  Image, SAM and ASR endpoints are not chat completions: add helpers in `app/llm.py` (docs: https://dev.meta.ai/docs).
- Structured Meta calls used by the search logic go through `app/meta_api.py` (httpx; retries per the API error guide,
  `USD_CAP` per process, every answer cached in `data/cache/`; `MODEL_MODE=replay` = cache only, no network).
- Frontend: `static/index.html` (HTML + Tailwind CDN + vanilla JS), served at `/`.
- Deploy: `Dockerfile` -> Railway (`railway up`). Add deps with `uv add <pkg>`, never pip.

## Project notes
<!-- Fill in once the idea is chosen. Agents read this. -->
- Problem statement: Push the Frontier.
- User and problem: volunteer land search-and-rescue teams flying 1-3 drones. Where to fly next, and finding a person a few pixels wide in thousands of photos.
- Meta models used, and why each:
  - SAM 3.1 finds people in aerial tiles and segments terrain.
  - Muse Spark verifies candidates, reads clues and parses the mission brief.
  - Muse Wonder evolves the planner's priority function in a sandbox.
  - Muse Voice Transcribe turns radio reports into map updates.
  - Muse Spark Open is the open-weights verifier.
- Demo path: brief, then probability map, then a drone race against a lawnmower sweep on real terrain, then candidates on real photos, then a human confirms, then found.
- Core maths lives in `app/sar/` (see its module docstrings). Run `uv run python -m app.sar.bench` and `uv run pytest`.
- Planners plug into the arena through `app/sar/interface.py` and never see the truth (`Scenario`). Baselines live in `app/sar/planners/`; the team's planner in `app/planner/`.
- The page at `/` replays recorded races: `uv run python -m app.sar.replay --seed 17` writes `data/runs/` (Spark from the cache only unless `--live`); `POST /api/runs` records one and never spends. Planners may offer `drain_notes()` to explain themselves in the log.
- Detection rates are measured, not chosen: `app/perception/` fetches HERIDAL test photos (CC BY 3.0; never committed), runs SAM 3.1 + Spark on tiles (`gate`), and writes `data/pod_table.json` (`calibrate`), which the simulator loads. Rerunning the gate replays cached answers for free.
- Imagery mode (`bench --imagery`, and every replay): a photo of the person is decided by a recorded SAM 3.1 + Spark answer on a real person of that pixel size (`app/perception/pool.py`); `GET /api/tiles/{id}.jpg` serves the tile (`python -m app.perception.pool` pre-renders them for a host without the dataset). Every photo also raises false alarms at the measured empty-ground rate.
- Stress tests: `bench --truth-moving/--truth-drift/--truth-category/--truth-delay-h` change how the hidden person behaves while planners keep the same statistics; `--alt`, `--drones`, `--forest-factor` change the setup. The final evaluation is fixed in advance in `docs/evaluation.md`.
- Witness reports come from `app/sar/cues.py` (fixed reliability, frozen before any model read them). Spark turns them into weighted map layers in `app/agent/`; check its reading with `uv run python -m app.agent.reading_check`.
