# Team workflow

## Setup
```bash
git clone https://github.com/h-a-forster/lastseen
cd lastseen
uv sync
cp .env.example .env    # Windows: Copy-Item .env.example .env
```
Each teammate puts their own `MODEL_API_KEY` in `.env`, which is never committed. Model answers are cached, so most
work (tests, benches, replays) makes no paid calls.

## How changes land
- `main` is protected: every change goes in through a pull request and is squash-merged.
- CI runs the test suite on every pull request, with no model calls.
- One branch per change. The pull request template asks how the change was checked and which judging criterion it
  moves.
- Search-time claims are tuned on the development seeds only; the held-out seeds are run once, after the feature
  freeze, as fixed in [evaluation.md](evaluation.md).
- Coding agents read [AGENTS.md](../AGENTS.md); each gets one task on its own branch.
