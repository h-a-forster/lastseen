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
- Frontend: `static/index.html` (HTML + Tailwind CDN + vanilla JS), served at `/`.
- Deploy: `Dockerfile` -> Railway (`railway up`). Add deps with `uv add <pkg>`, never pip.

## Project notes
<!-- Fill in once the idea is chosen. Agents read this. -->
- Problem statement: Build for Your World / Push the Frontier
- User and problem:
- Meta models used, and why each:
- Demo path (the one flow that must work):
