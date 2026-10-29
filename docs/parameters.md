# Parameters: where each number comes from, and whether results depend on it

Every constant the simulator uses, its source, and what changes when it is wrong. The planners and the hidden
person are deliberately given different information: planners search with published lost-person statistics, while
the person is moved by behaviour rules (`app/sar/walker.py`). The sweeps below go further and make the person behave
unlike the statistics the planners were given.

## Detection (measured)
| Parameter | Value | Source |
|---|---|---|
| Chance of finding a person in a photo, open ground | 1.00 / 0.97 / 0.98 / 0.79 / 0.62 / 0.47 at 1 / 1.5 / 2 / 3 / 5 / 8 cm per pixel | Measured: SAM 3.1 proposals checked by Muse Spark on 660 labelled people in the HERIDAL test photos (`data/pod_table.json`, `app/perception/`). At 100 m a person is about 44 px: 179 of 227 people that size were found, 79% (95% interval 70-86%, resampling whole photos, since people in one photo share light and ground) |
| False alarms | 0.0094 per 1000 px tile of empty ground (3 of 320; 95% upper bound 0.027) | Measured on 320 HERIDAL tiles without people. Looking at them, 3 of the 4 alarms raised on empty tiles at any scale were real people the labels missed, so this rate is pessimistic |
| Forest | 0.4 of the open-ground chance | **Assumption**: the dataset has no people under canopy. Swept at 0.25 and 0.6 |
| Person size from above | 1.2 m | Between standing (about 0.5 m) and lying (about 1.7 m); converts altitude into the pixel size the detector was measured at |

## Drones and camera
| Parameter | Value | Source |
|---|---|---|
| Camera | 72° × 57° field of view, 5280 px wide | A 4/3-inch, 84° diagonal survey camera, common on enterprise drones |
| Altitude | 100 m (2.75 cm per pixel, a 145 × 109 m photo) | A person is then about 44 px across. Chosen from the sweep below: 100-120 m found more people within 2 h than 40-80 m, and 100 m leaves a margin under the usual 120 m legal ceiling |
| Speed | 10 m/s | Typical cruise for sharp survey stills |
| Usable flight time | 25 min per battery, 3 min swap | Rated times of 40-45 min, minus real-world loss and a landing reserve |
| Photo overlap | 10% along track | Search needs coverage, not stitched maps, so less than mapping's 60-80% |
| Drones | 3 | Volunteer teams fly 1-3. Swept 1-4 |
| Search time | 4 hours | One operational period |
| Drone failure (`--lose-drone`) | One drone, at a time drawn uniformly from 15-120 min | **Assumption**, a stress test rather than a measured failure rate; the same failure for every planner |
| Restart block (`spiral+spark`) | The square one battery can sweep (about 1.4 km at 100 m) | Choice: the area a team would cover first around a new datum; not tuned |

## The hidden person
| Parameter | Value | Source |
|---|---|---|
| Behaviour | Dementia: keep going one way until stuck or tired, deflect off barriers. Hikers: a per-step mix of direction, route and random travel | Koester, *Lost Person Behavior*; mix fitted to ISRID hikers by Hashimoto et al. (2022) |
| Speed, drift, moving time | Fitted (`PARAMS` in `app/sar/walker.py`) | Fitted so end points match the published distance rings; `python -m app.sar.walker_check` prints the comparison |
| Time missing before the search | Dementia 0.5-2 h, hikers 2-6 h | **Assumption**. Swept at 0.25-0.5 h and 3-5 h |
| Witness reports | Direction true 70%, fitness 80%, sighting 75%; a sighting in 60% of searches | **Assumption**, fixed in `app/sar/cues.py` before any model read a report |

## Maps
| Parameter | Value | Source |
|---|---|---|
| Cell size | 25 m | Smaller than a photo, so coverage is counted per photo |
| Synthetic maps | 3 km square, random forest, scrub, water, paths and a no-fly zone | Generated per seed (`app/sar/scenario.py`) |
| Real place | Saint-Ursanne, Doubs valley, 3 km square | OpenStreetMap land cover and paths (`app/sar/terrain.py`); aerial imagery © swisstopo, shown only |

Not modelled: height (slopes don't slow the walker or change the photo scale), and repeated photos of one spot
count as independent chances, which flatters re-searching a little.

## Sensitivity
Each row is one run of `uv run python -m app.sar.bench --seeds 200 --workers 8 --category dementia
--planners lawnmower,spiral,concentric,sector,greedy` plus the flags shown: 200 development scenarios (never the
held-out ones), 3 km synthetic maps, no model calls. The `--truth-*` flags change how the hidden person behaves while
the planners keep the statistics they were given. The rows were run at 60 m, the default until the altitude rows
below moved it to 100 m: add `--alt 60` to reproduce them.

Columns: the share found within 2 hours by each planner; the probability-map planner minus the expanding square,
paired by scenario, with a 95% bootstrap interval; and per search for the expanding square, the photos taken, the
alarms a reviewer would check after SAM 3.1 + Spark, and photos that had the person in them but raised no alarm.
"In area" is the share of scenarios where the person is inside the map when the search starts: no planner can beat it.

| Flags | In area at start | lawnmower | spiral | concentric | sector | greedy | Greedy − spiral at 2 h | Photos (spiral) | Alarms (spiral) | Missed in frame (spiral) |
|---|---|---|---|---|---|---|---|---|---|---|
| `(defaults)` | 87% | 58% | 60% | 60% | 57% | 47% | -12% [-18%, -7%] | 2669 | 601 | 0.41 |
| `--imagery` | 87% | 60% | 60% | 61% | 60% | 49% | -11% [-16%, -6%] | 2634 | 593 | 0.34 |
| `--truth-moving 0.6` | 90% | 66% | 66% | 61% | 62% | 58% | -8% [-13%, -2%] | 2296 | 518 | 0.46 |
| `--truth-moving 1.5` | 86% | 54% | 50% | 54% | 52% | 40% | -10% [-16%, -6%] | 3063 | 689 | 0.41 |
| `--truth-drift 2` | 93% | 70% | 76% | 68% | 72% | 59% | -17% [-24%, -10%] | 1813 | 410 | 0.47 |
| `--truth-category hiker` | 32% | 26% | 27% | 26% | 25% | 20% | -7% [-12%, -3%] | 4485 | 1009 | 0.12 |
| `--truth-delay-h 0.25,0.5` | 100% | 65% | 80% | 74% | 69% | 62% | -18% [-24%, -11%] | 1648 | 372 | 0.48 |
| `--truth-delay-h 3,5` | 64% | 52% | 55% | 53% | 50% | 45% | -10% [-15%, -4%] | 2980 | 671 | 0.34 |
| `--alt 40` | 87% | 55% | 52% | 54% | 50% | 40% | -13% [-19%, -7%] | 4510 | 1017 | 0.33 |
| `--alt 80` | 87% | 64% | 66% | 64% | 60% | 52% | -14% [-21%, -8%] | 1776 | 401 | 0.43 |
| `--alt 100` | 87% | 65% | 68% | 62% | 66% | 58% | -9% [-14%, -4%] | 1340 | 302 | 0.58 |
| `--alt 120` | 87% | 64% | 68% | 64% | 65% | 59% | -9% [-14%, -4%] | 1106 | 250 | 0.63 |
| `--drones 1` | 87% | 33% | 40% | 42% | 41% | 38% | -2% [-8%, +3%] | 1223 | 276 | 0.28 |
| `--drones 2` | 87% | 53% | 52% | 52% | 50% | 44% | -9% [-14%, -4%] | 2069 | 466 | 0.35 |
| `--drones 4` | 87% | 64% | 64% | 66% | 64% | 51% | -12% [-18%, -7%] | 3274 | 738 | 0.42 |
| `--forest-factor 0.25` | 87% | 54% | 55% | 56% | 54% | 45% | -10% [-16%, -4%] | 2900 | 653 | 0.59 |
| `--forest-factor 0.6` | 87% | 62% | 65% | 64% | 62% | 50% | -16% [-22%, -10%] | 2385 | 538 | 0.20 |
| `--lose-drone` | 87% | 55% | 57% | 58% | 54% | 46% | -12% [-18%, -6%] | 2230 | 502 | 0.36 |
| `--swap-s 0` | 87% | 60% | 61% | 61% | 59% | 52% | -8% [-14%, -3%] | 2918 | 657 | 0.41 |

What this shows:
- **The four fixed patterns are within a few points of each other** in most settings (up to 15 when the person is
  reported quickly, when the expanding square does best), so none of them is "the" baseline. The expanding square is
  the reference because it is the standard pattern around a known last-seen point.
- **The probability-map planner trails the expanding square in every setting**, by 2-18 points, including when the
  person behaves exactly as its statistics say. That is why the headline evaluation (`docs/evaluation.md`) compares
  the same planner with and without Spark, rather than claiming a better planner.
- **A wrong behaviour model moves every planner together.** People who walk 50% longer, walk like hikers or are
  reported late are found less often; people who wander twice as much stay nearer and are found more often. In every
  case the probability-map planner stays behind the expanding square.
- **Higher flights found people sooner.** Within 2 hours, the expanding square found 52% at 40 m, 60% at 60 m and
  68% at 100-120 m: each photo covers more ground, which outweighs the smaller person in the photo, although the
  detector misses more people who are in frame (0.3 to 0.6 a search). This rests on the measured detection rates down
  to 36 px a person and on the forest assumption, so it is a result of this model, not field advice.
- **Imagery mode agrees with fast mode** (60% for the expanding square in both), so deciding finds by real photo
  answers instead of the measured table doesn't change the picture.
- **Losing a drone** (one per search, at a seeded time in the first 2 hours) costs every planner 1-3 points within
  2 hours. The fixed patterns hand the lost drone's path to the others, so a failure doesn't favour the planner that
  keeps a map.
- **Short sorties cost the probability-map planner a third of its gap.** It flies one small block per sortie and then
  sits through the 3-minute turnaround, so its drones fly about 71% of the time against 93-94% for the fixed
  patterns (the bench's `air` column). With no turnaround (`--swap-s 0`) it trails the expanding square by 8 points
  instead of 12, so filling each battery is the first thing a better planner should do.
