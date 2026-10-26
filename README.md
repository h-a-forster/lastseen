# (Project name TBD)

> Meta Global AI Developer Hackathon 2026 · Indie track · Problem statement: TBD

One-line pitch goes here.

## Built during the hackathon
Everything after commit `be7036a` was built during the event (26 Oct - 4 Nov 2026).
Commit `be7036a` is a generic FastAPI + LiteLLM starter template written beforehand; it contains no project-specific code.

<!-- Before submitting: list the features built at the event, add a screenshot, the live link and the demo video. -->

## Run locally
```bash
uv sync
cp .env.example .env      # works with MODEL=mock and no keys; set MODEL=meta/muse-spark-1.3 and MODEL_API_KEY for real calls
uv run uvicorn app.main:app --reload
```
Open http://localhost:8000.

## Team
- Rules and judging: [docs/HACKATHON.md](docs/HACKATHON.md)
- Git workflow: [docs/TEAM.md](docs/TEAM.md)
- Coding agents read [AGENTS.md](AGENTS.md)

## Deploy (Railway)
```bash
railway login
railway init        # once
railway up
```
Set `MODEL`, `MODEL_API_KEY` and `MAX_LLM_CALLS` under Variables, then generate a public domain.

## Layout
```
app/main.py        routes (/api/health, /api/chat streaming)
app/llm.py         all model calls; meta/<model> -> Meta Model API
static/index.html  frontend
docs/              hackathon rules, team workflow
```
