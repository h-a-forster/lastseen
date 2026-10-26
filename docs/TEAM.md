# Team workflow

## One-time setup (each teammate)
```bash
git clone <repo-url>
cd <repo>
uv sync
cp .env.example .env    # Windows: Copy-Item .env.example .env
```
Put **your own** `MODEL_API_KEY` in `.env` (each person claims their own $150 Meta credits). Never commit `.env`.

## How we avoid stepping on each other
1. **`main` is protected.** Nobody pushes to it directly; everything goes in through a pull request (PR).
2. **One branch per task**, named `yourname/what`, e.g. `sara/upload-ui`.
3. **Claim the task first**: create a GitHub issue (or comment "taking this" on one) so two people don't build the same thing.
4. **Own an area.** Split by folder where possible (backend `app/`, frontend `static/`, demo and video `docs/`). Two people editing the same file at once is what causes conflicts.
5. **Small PRs, merged often.** Aim for at least a few merges a day. A branch that lives a whole day will conflict.
6. **Keep your branch fresh**: before opening a PR, run
   ```bash
   git fetch origin
   git merge origin/main     # fix any conflicts locally, run the app, then push
   ```
7. **Merging**: "Squash and merge" on GitHub, then delete the branch. Anyone can merge once it runs; a quick look from one teammate is better.

## What actually stops two people merging over each other
- Protected `main` means changes only land through PRs.
- GitHub refuses to merge a PR that conflicts with `main`; the author must merge `main` in and resolve the conflict first.
- "Require branches to be up to date" means a PR has to include the latest `main` before it can merge, so a second PR can't silently undo the first.

## Daily rhythm
- Morning (5 min): what's each person doing today, and what's blocking? Update the issues.
- Integrate on `main` at least twice a day. Run the app from `main` and check the demo path still works.
- From ~2 Nov: feature freeze. Only fixes and demo polish go into `main`.

## Using coding agents (e.g. Muse Code)
- Agents read `AGENTS.md`. Give each agent one task on one branch.
- With several agents on one laptop, give each its own git worktree so they don't share files.
