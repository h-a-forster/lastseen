# Parameters: where each number comes from, and whether results depend on it

Every constant the simulator uses, its source, and what changes when it is wrong. The planners and the hidden
person are deliberately given different information: planners search with published lost-person statistics, while
the person is moved by behaviour rules (`app/sar/walker.py`). The sweeps below go further and make the person behave
unlike the statistics the planners were given.

## Detection (measured)
| Parameter | Value | Source |
|---|---|---|
| Chance of finding a person in a photo, open ground | 1.00 / 0.96 / 0.96 / 0.78 / 0.57 / 0.44 at 1 / 1.5 / 2 / 3 / 5 / 8 cm per pixel | Measured: SAM 3.1 proposals checked by Muse Spark on 263 people in the HERIDAL test photos (`data/pod_table.json`, `app/perception/`) |
| False alarms | 0.0094 per 1000 px tile of empty ground (3 of 320; 95% upper bound 0.027) | Measured on 320 HERIDAL tiles without people. Looking at them, 3 of the 4 alarms raised on empty tiles at any scale were real people the labels missed, so this rate is pessimistic |
| Forest | 0.4 of the open-ground chance | **Assumption**: the dataset has no people under canopy. Swept at 0.25 and 0.6 |
| Person size from above | 1.2 m | Between standing (about 0.5 m) and lying (about 1.7 m); converts altitude into the pixel size the detector was measured at |

## Drones and camera
| Parameter | Value | Source |
|---|---|---|
| Camera | 72° × 57° field of view, 5280 px wide | A 4/3-inch, 84° diagonal survey camera, common on enterprise drones |
| Altitude | 60 m (1.65 cm per pixel, an 87 × 65 m photo) | A person is then about 70 px across, where the measured detection is high. Swept 40-120 m: higher found people sooner in this model (below) |
| Speed | 10 m/s | Typical cruise for sharp survey stills |
| Usable flight time | 25 min per battery, 3 min swap | Rated times of 40-45 min, minus real-world loss and a landing reserve |
| Photo overlap | 10% along track | Search needs coverage, not stitched maps, so less than mapping's 60-80% |
| Drones | 3 | Volunteer teams fly 1-3. Swept 1-4 |
| Search time | 4 hours | One operational period |

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
the planners keep the statistics they were given.

Columns: the share found within 2 hours by each planner; the probability-map planner minus the expanding square,
paired by scenario, with a 95% bootstrap interval; and per search for the expanding square, the photos taken, the
alarms a reviewer would check after SAM 3.1 + Spark, and photos that had the person in them but raised no alarm.
"In area" is the share of scenarios where the person is inside the map when the search starts: no planner can beat it.

| Flags | In area at start | lawnmower | spiral | concentric | sector | greedy | Greedy − spiral at 2 h | Photos (spiral) | Alarms (spiral) | Missed in frame (spiral) |
|---|---|---|---|---|---|---|---|---|---|---|
| `(defaults)` | 87% | 57% | 56% | 57% | 57% | 47% | -10% [-15%, -4%] | 2196 | 495 | 0.39 |
| `--imagery` | 87% | 58% | 57% | 58% | 60% | 49% | -8% [-14%, -2%] | 2089 | 470 | 0.29 |
| `--truth-moving 0.6` | 90% | 64% | 61% | 58% | 62% | 58% | -3% [-8%, +2%] | 1930 | 435 | 0.42 |
| `--truth-moving 1.5` | 86% | 52% | 49% | 52% | 52% | 40% | -9% [-14%, -4%] | 2492 | 561 | 0.36 |
| `--truth-drift 2` | 93% | 69% | 70% | 63% | 72% | 59% | -11% [-18%, -4%] | 1515 | 342 | 0.41 |
| `--truth-category hiker` | 32% | 25% | 26% | 26% | 24% | 20% | -6% [-10%, -2%] | 3583 | 806 | 0.12 |
| `--truth-delay-h 0.25,0.5` | 100% | 64% | 74% | 73% | 68% | 62% | -12% [-19%, -6%] | 1447 | 326 | 0.47 |
| `--truth-delay-h 3,5` | 64% | 52% | 51% | 50% | 50% | 45% | -6% [-11%, -2%] | 2468 | 556 | 0.33 |
| `--alt 40` | 87% | 53% | 49% | 51% | 50% | 40% | -10% [-15%, -4%] | 3817 | 862 | 0.32 |
| `--alt 80` | 87% | 62% | 62% | 60% | 60% | 52% | -10% [-16%, -4%] | 1538 | 347 | 0.43 |
| `--alt 100` | 87% | 62% | 64% | 58% | 66% | 58% | -5% [-11%, +0%] | 1167 | 264 | 0.55 |
| `--alt 120` | 87% | 62% | 64% | 61% | 64% | 59% | -6% [-11%, -0%] | 996 | 225 | 0.59 |
| `--drones 1` | 87% | 32% | 40% | 42% | 41% | 38% | -2% [-8%, +3%] | 1087 | 246 | 0.27 |
| `--drones 2` | 87% | 52% | 50% | 50% | 50% | 44% | -6% [-12%, -2%] | 1747 | 393 | 0.34 |
| `--drones 4` | 87% | 63% | 58% | 62% | 64% | 51% | -7% [-13%, -2%] | 2730 | 615 | 0.41 |
| `--forest-factor 0.25` | 87% | 52% | 52% | 54% | 53% | 45% | -6% [-12%, -1%] | 2339 | 527 | 0.54 |
| `--forest-factor 0.6` | 87% | 62% | 62% | 61% | 62% | 50% | -13% [-19%, -7%] | 1952 | 441 | 0.19 |

What this shows:
- **The four fixed patterns are within a few points of each other** in most settings (up to 10 with one drone or
  a quick report), so none of them is
  "the" baseline. The expanding square is the reference because it is the standard pattern around a known
  last-seen point.
- **The probability-map planner trails the expanding square in every setting**, by 2-13 points, including when the
  person behaves exactly as its statistics say. That is why the headline evaluation (`docs/evaluation.md`) compares
  the same planner with and without Spark, rather than claiming a better planner.
- **A wrong behaviour model moves every planner together.** People who walk 50% longer, wander twice as much, walk like
  hikers, or are reported late lower all find rates, but the order of the planners doesn't change.
- **Higher flights found people sooner.** Within 2 hours, the expanding square found 49% at 40 m, 56% at 60 m and
  64% at 100-120 m: each photo covers more ground, which outweighs the smaller person in the photo, although the
  detector misses more people who are in frame (0.3 to 0.6 a search). This rests on the measured detection rates down
  to 36 px a person and on the forest assumption, so it is a result of this model, not field advice.
- **Imagery mode agrees with fast mode** (57% against 56% for the expanding square), so deciding finds by real photo
  answers instead of the measured table doesn't change the picture.
