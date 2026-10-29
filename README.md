# LastSeen

> Meta Global AI Developer Hackathon 2026 · Push the Frontier

Drone search and rescue for volunteer teams: where to fly next, and whether the missing person is in the photos.

When someone goes missing, a team with one to three drones faces two problems. Thousands of photos come back, each
with a person a few dozen pixels wide, or nobody. And every new witness report changes where the drones should go.
LastSeen uses Meta's models for both, and tests every claim on a simulator built for it.

## What it does
- **Is anyone in this photo?** SAM 3.1 proposes possible people in each 1000 px tile of a drone photo, and Muse Spark
  checks each proposal on a close-up. A reviewer checks a few alarms instead of every photo.
- **Start a search here.** Type or speak what the family and witnesses said. Muse Voice Transcribe writes it down,
  and Muse Spark turns it into weighted hypotheses on a real map, using the named places on it ("the chapel up on the
  hill"). Each drone gets a first block to fly, downloadable as a QGroundControl mission.
- **Watch a search.** Replays of the same search flown by different planners, on synthetic maps or a real valley
  (Saint-Ursanne, Switzerland), with the person hidden until the end. Every photo that had the person in it was
  decided by a real SAM 3.1 + Spark answer on a real drone photo of a person that size (in forest, canopy also hides
  them at an assumed rate, since the photos have no one under trees).

## Meta models
| Model | Job |
|---|---|
| SAM 3.1 | Finds people in drone photos (text prompt "person", boxes with confidence) |
| Muse Spark | Checks each SAM proposal on a close-up; reads briefs and witness reports into hypotheses, as strict JSON whose quotes and numbers code then checks |
| Muse Voice Transcribe | Turns a spoken brief into text |

## How we know whether it works
- **Detection is measured, not assumed.** SAM 3.1 + Spark ran on 660 labelled people in the HERIDAL test photos and on
  empty ground; the simulator uses the measured rates by person size (`data/pod_table.json`).
- **The simulator can't grade its own homework.** Planners search with published lost-person statistics, while the
  hidden person follows behaviour rules; stress tests make the person behave unlike those statistics.
- **The final comparison is fixed in advance** ([docs/evaluation.md](docs/evaluation.md)): 200 unseen scenarios,
  one primary endpoint, and wording rules for the result whatever it shows.
- **Every parameter has a source and a sensitivity run** ([docs/parameters.md](docs/parameters.md)).

## Run it
```bash
uv sync
cp .env.example .env          # add MODEL_API_KEY for live model calls
uv run uvicorn app.main:app --reload
uv run pytest
```
Open http://localhost:8000. The simulator and bench need no key:
```bash
uv run python -m app.sar.bench --planners lawnmower,spiral,greedy --seeds 40 --workers 4
uv run python -m app.sar.replay --seed 19 --place saint-ursanne --planners spiral,greedy
```
A `+spark` planner (e.g. `greedy+spark`) needs `MODEL_API_KEY` for Spark's readings, then replays them from the cache.

Optional data (downloaded, never committed): `uv run python -m app.perception.heridal` fetches the HERIDAL test
photos (about 0.5 GB); `uv run python -m app.perception.gate` measures detection on them (paid, about $2.50); then
`uv run python -m app.perception.triage --example` and `uv run python -m app.agent.brief --example` build the page's
examples.

## Plug in a planner
A planner implements `app/sar/interface.py`: `reset` (the map, fleet, detection rates and statistics), `observe`
(photos that found nobody, witness reports, Spark's hypotheses, a drone that fails), `plan` (the next sortie for a
drone at base) and optionally `retask` (a new route for a drone in the air when a report comes in) and
`probability_map`. It never sees the hidden person. Start from `app/planner/__init__.py`
(`ours`), with `app/sar/planners/greedy.py` as a complete example, then:
```bash
uv run python -m app.sar.bench --planners ours,spiral,greedy --seeds 100 --workers 8
uv run python -m app.sar.bench --planners ours,ours+spark --seeds 30   # with Spark's reading of the reports (paid)
uv run python -m app.sar.bench --planners ours+spark,spiral+spark --seeds 30   # vs a spiral that restarts on Spark's readings (paid)
uv run python -m app.sar.bench --planners ours,spiral,greedy --lose-drone      # one drone fails in the first 2 h
```

## Layout
```
app/sar/          simulator: maps, hidden person, arena, witness reports, baselines, bench, replays
app/agent/        Spark reads reports and briefs into hypotheses
app/perception/   SAM 3.1 + Spark on photos: calibration, photo pool, photo check
app/planner/      the team's planner
app/main.py       web API;  static/  the page
docs/             evaluation plan, parameters, team workflow
```

## Data and credits
- Drone photos: HERIDAL database (University of Split), repacked by P. Pyrrö, Aalto University
  (doi:10.5281/zenodo.5662350), CC BY 3.0.
- Map data © OpenStreetMap contributors (ODbL). Aerial imagery © swisstopo, loaded by the browser from swisstopo.
- Lost-person statistics: Koester, *Lost Person Behavior* (ISRID data); Hashimoto et al. (2022); sources per number
  in `data/lost_person_stats.json`.

## Built during the hackathon
Commit `be7036a` is a generic FastAPI starter template written beforehand; it contains no project-specific code.
The two commits on 26 Oct, kickoff day, are setup only (model API routing, spend guard, team docs, licence). All
product code was written during the event, from 28 Oct.

## License
MIT
