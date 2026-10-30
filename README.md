# LastSeen

> Meta Global AI Developer Hackathon 2026 · Push the Frontier

Search at sea is planned with Bayesian software; on land, teams are taught probability methods but practitioners
report rarely applying them in full, and planning a drone search is still "more art than science"
([MIT Technology Review, 2024](https://www.technologyreview.com/2024/05/30/1092988/ai-directed-drones-could-help-find-lost-hikers-faster/)).
What witnesses say reaches the search map by hand, or not at all: a fixed search pattern only uses it when someone
moves the pattern by hand. LastSeen closes that gap. Muse
Spark turns every report into weighted, checked scenarios in seconds; when a witness reports a sighting, the drone
breaks off to search where the reading points, and it flies the search itself. SAM 3.1 and Muse Spark check every
photo for a person a few dozen pixels wide. One volunteer launches it, swaps batteries and stays in
command.

It is built for searches where minutes matter most, such as an older person with dementia who has walked away from
home, with one volunteer and one drone. Every claim is tested on a simulator built for it.

## How a search runs
The volunteer types or says what the family told them and launches the drone. From then on the search runs itself:
- the drone flies each battery's search on autopilot at 100 m;
- each photo goes to SAM 3.1 and Muse Spark as it is taken, and only the alarms reach a person, as close-ups;
- each witness report goes to Muse Spark, and the drone's next leg is planned from the updated map, mid-flight if
  needed;
- when the battery runs low the drone flies home, the volunteer swaps it, and the search carries on.

The volunteer stays in command, as the law requires, and the ground team confirms any find. The drone keeps its radio link to the base
throughout, which is how a new leg reaches it mid-flight. LastSeen also expects each photo back as it is taken: in
the simulator's flight plan that is one photo of about 5 MB every 10 seconds, around 4 Mbit/s. Some enterprise drones
can send photos during flight, and on others a small onboard computer with a 4G modem can. Where photos only come off
after landing, a find waits for the battery swap, up to 25 minutes.

The simulator runs this loop end to end. What isn't built is LastSeen's connection to a real drone: each battery's
mission exports as a QGroundControl file (it loads, but hasn't been flown) instead of going over the radio link, and
nothing receives photos from a drone yet.

**Try it live:** the hosted demo is the website link in this repository's About panel (the first visit takes a few seconds to wake the server).

![One drone searching the Saint-Ursanne valley: the expanding square (left) against the team planner, which broke off
to follow Muse Spark's reading of a dog walker's sighting and found her at 1 h 48; the find was decided on a real
drone photo of a person her size](docs/img/race.jpg)

## In numbers
- **79%** of people about 44 px wide (a drone at 100 m) are found in a single photo by SAM 3.1 + Muse Spark, measured
  on 227 labelled people in real drone photos (95% interval 70-86%, resampling whole photos).
- **About 270 close-ups instead of 1,200 photos** for a reviewer in 4 hours of one-drone flying. A mountain-rescue
  team reports 5 to 45 seconds to look through one full photo
  ([Gotovac et al., 2020](https://doi.org/10.3390/rs12203295)), so 1,200 photos are about 2 to 15 hours of looking; a
  close-up is one spot to accept or reject (not timed yet).
- **53% found within 2 hours instead of 46%** with the expanding square alone: Muse Spark reads the witness reports,
  and when someone reports a sighting the drone breaks off its square to follow Spark's reading (one drone, 200
  development scenarios in photo mode, as the final run; when the sighting was wrong, it did as well as the square).
  A careful volunteer who restarts the square where the witness said gets 51%: on the simulator's simple reports,
  most of the gain is acting on a sighting at all. Reworded the way people talk ("a 20-minute walk", "NE-ish",
  witnesses who aren't sure), the same searches found 52% (-1.5 points, 95% interval -3.5 to 0), though Spark's
  readings put a little less weight near the spot the witness named. The held-out result, run once after the feature freeze, goes in
  [docs/evaluation.md](docs/evaluation.md) whatever it shows.

## What it does
- **Is anyone in this photo?** SAM 3.1 proposes possible people in each 1000 px tile of a drone photo, and Muse Spark
  checks each proposal on a close-up. A reviewer checks a few alarms instead of every photo.
- **Start a search here.** Type or speak what the family and witnesses said. Muse Voice Transcribe writes it down,
  and Muse Spark turns it into weighted hypotheses on a real map, using the named places on it ("the chapel up on the
  hill"). Spark also grades each witness the way a search manager would (do they know her, did they see it
  themselves, does anyone back it up?); the grades are shown, not used to change weights. LastSeen plans the drone's first
  battery, downloadable as a QGroundControl mission (it loads and displays in QGroundControl; it hasn't been flown on
  a real drone).
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
| Muse Spark Open | Spark's jobs on open weights, served by Baseten (`SPARK_BACKEND=open`) |

**Open weights.** With `SPARK_BACKEND=open`, every Spark call goes to Muse Spark Open, Meta's open-weight model, on
Baseten. On the same 200 development scenarios (one drone, photo mode), the team planner found 52% within 2 h reading
the reports with Muse Spark Open, against 53% with Muse Spark (paired difference -0.5 points, 95% interval -2.5 to
+1.5; 5 of 200 scenarios changed). All 327 readings passed our checks, which matter more here because Baseten
doesn't enforce the JSON schema for this model; an answer cut off at the token limit is asked again with twice the
budget. On the photo check's test tiles, SAM 3.1 + Muse Spark
Open found 177 of 265 people and Muse Spark 179 (a one-off script, not in the repo). On the hosted page, "Use open
weights" reads a brief with Muse Spark Open. Open weights mean a search team
isn't tied to one hosted API; we tested them through Baseten, not on our own hardware.

## How we know whether it works
- **Detection is measured, not assumed.** SAM 3.1 + Spark ran on 660 labelled people in the HERIDAL test photos and on
  empty ground; the simulator uses the measured rates by person size (`data/pod_table.json`).
- **The simulator can't grade its own homework.** Planners search with published lost-person statistics, while the
  hidden person follows behaviour rules; stress tests make the person behave unlike those statistics.
- **The final comparison is fixed in advance** ([docs/evaluation.md](docs/evaluation.md)): 200 unseen scenarios,
  one primary endpoint, and wording rules for the result whatever it shows.
- **Every parameter has a source and a sensitivity run** ([docs/parameters.md](docs/parameters.md)).

## Before anyone relies on it
- The searches are simulated: the map and photos are real, the hidden person's walk is not. Limits are listed in
  [docs/evaluation.md](docs/evaluation.md#known-limits).
- Mission heights are 100 m above the take-off point, and the simulator doesn't model terrain height; over hills a
  real mission needs terrain-following heights (QGroundControl can adjust them).
- LastSeen isn't connected to a real drone yet: missions export as QGroundControl files (they load but haven't been
  flown), and nothing receives photos in flight. Where a drone can't send photos in flight, a find waits for the
  battery swap, up to 25 minutes later.
- Next: connecting to a real drone's radio link, a test flight over open ground, and a trial with a volunteer search team on a training exercise.

## Run it
```bash
uv sync
cp .env.example .env          # add MODEL_API_KEY for live model calls
uv run uvicorn app.main:app --reload
uv run pytest
```
Open http://localhost:8000 (with `RECORD_RACES=1` the page can also record a race for any development seed, from
cached readings only). The simulator and bench need no key:
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
SPARK_BACKEND=open uv run python -m app.sar.bench --planners spiral,hedge+spark --seeds 200 --drones 1 --imagery   # on Muse Spark Open
uv run python -m app.sar.bench --planners hedge+spark,spiral+spark --seeds 200 --drones 1      # vs a spiral that restarts on Spark's readings
uv run python -m app.sar.bench --planners spiral,spiral+witness,hedge+spark --seeds 200 --drones 1 --imagery   # vs a volunteer who restarts where the witness said
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
