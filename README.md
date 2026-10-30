# LastSeen

> Meta Global AI Developer Hackathon 2026 · Push the Frontier

Drone search and rescue for a volunteer with one drone: where to fly next, and whether the missing person is in the
photos.

When an older person with dementia walks away from home, a volunteer with a drone faces two problems. Hundreds of
photos come back, each with a person a few dozen pixels wide, or nobody. And every new witness report changes where
the drone should go. LastSeen uses Meta's models for both, and tests every claim on a simulator built for it.

**Try it live:** the hosted demo is the website link in this repository's About panel (the first visit takes a few seconds to wake the server).

![One drone searching the Saint-Ursanne valley: the expanding square (left) against the team planner, which broke off
to follow Muse Spark's reading of a dog walker's sighting and found her at 1 h 48; the find was decided on a real
drone photo of a person her size](docs/img/race.jpg)

## In numbers
- **79%** of people about 44 px wide (a drone at 100 m) are found in a single photo by SAM 3.1 + Muse Spark, measured
  on 227 labelled people in real drone photos (95% interval 70-86%, resampling whole photos).
- **About 150 close-ups instead of 660 photos** for a reviewer in a 4-hour, one-drone search.
- **53% found within 2 hours instead of 46%** with the expanding square alone: Muse Spark reads the witness reports,
  and when someone reports a sighting the drone breaks off its square to follow Spark's reading (one drone, 200
  development scenarios in photo mode, as the final run; when the sighting was wrong, it did as well as the square). The held-out result, run once
  after the feature freeze, goes in [docs/evaluation.md](docs/evaluation.md) whatever it shows.

## What it does
- **Is anyone in this photo?** SAM 3.1 proposes possible people in each 1000 px tile of a drone photo, and Muse Spark
  checks each proposal on a close-up. A reviewer checks a few alarms instead of every photo.
- **Start a search here.** Type or speak what the family and witnesses said. Muse Voice Transcribe writes it down,
  and Muse Spark turns it into weighted hypotheses on a real map, using the named places on it ("the chapel up on the
  hill"). Spark also grades each witness the way a search manager would (do they know her, did they see it
  themselves, does anyone back it up?); the grades are shown, not used to change weights. The drone gets a first block
  to fly, downloadable as a QGroundControl mission (it loads and displays in QGroundControl; it hasn't been flown on a
  real drone).
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
`probability_map`. It never sees the hidden person. The team's planner, `hedge` (`app/planner/hedge.py`), flies the
expanding square until a witness reports a sighting, then sends one drone after Spark's reading of it with the map
planner `ours` (`app/planner/__init__.py`); `app/sar/planners/greedy.py` is a complete single-file example. Then:
```bash
uv run python -m app.sar.bench --planners hedge,ours,spiral,greedy --seeds 200 --drones 1 --workers 8
uv run python -m app.sar.bench --planners hedge,hedge+spark --seeds 200 --drones 1 --imagery   # with Spark's reading (paid once, then cached)
uv run python -m app.sar.bench --planners hedge+spark,spiral+spark --seeds 200 --drones 1      # vs a spiral that restarts on Spark's readings
uv run python -m app.sar.bench --planners spiral,hedge+spark --drones 3                        # three drones
uv run python -m app.sar.bench --planners spiral,hedge+spark --drones 3 --lose-drone           # one of three fails in the first 2 h
```

## Layout
```
app/sar/          simulator: maps, hidden person, arena, witness reports, baselines, bench, replays
app/agent/        Spark reads reports and briefs into hypotheses
app/perception/   SAM 3.1 + Spark on photos: calibration, photo pool, photo check
app/planner/      the team's planner (`hedge`) and the map planner it hunts with (`ours`)
app/main.py       web API;  static/  the page
docs/             evaluation plan, parameters, team workflow
```

## Data and credits
- Drone photos: HERIDAL database (University of Split), repacked by P. Pyrrö, Aalto University
  (doi:10.5281/zenodo.5662350), CC BY 3.0.
- Map data © OpenStreetMap contributors (ODbL). Aerial imagery © swisstopo, loaded by the browser from swisstopo
  (and in `docs/img/race.jpg`).
- Lost-person statistics: Koester, *Lost Person Behavior* (ISRID data); Hashimoto et al. (2022); sources per number
  in `data/lost_person_stats.json`.

## Built during the hackathon
Commit `be7036a` is a generic FastAPI starter template written beforehand; it contains no project-specific code.
The two commits on 26 Oct, kickoff day, are setup only (model API routing, spend guard, team docs, licence). All
product code was written during the event, from 28 Oct.

## License
MIT
