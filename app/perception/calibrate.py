"""Turn gate 1's measurements into the simulator's detection table (`data/pod_table.json`).

    uv run python -m app.perception.calibrate

The chance of spotting a person in one photo depends on how many pixels they cover, so the table is built from recall
per person-size bin (SAM 3.1 proposals kept by Muse Spark), then read at the pixel size a person has at each ground
resolution the simulator may fly: pixels = NOMINAL_PERSON_M / GSD. Terrain classes the photos don't cover are scaled
from the measured curve and marked as such.
"""

import json
from pathlib import Path

import numpy as np

from app.perception.gate import RESULTS, hits, load
from app.sar.metrics import wilson

TABLE = Path(__file__).resolve().parents[2] / "data" / "pod_table.json"
NOMINAL_PERSON_M = 1.2  # longest side of a person seen from above, between standing (~0.5 m) and lying (~1.7 m)
SIZE_BINS_PX = (0, 12, 16, 24, 32, 48, 64, 96, 1e9)
GSD_CM = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
# How each simulator terrain class relates to the photographed ground (open mountain meadows, scrub, rocky slopes):
# 1.0 = measured directly; lower = under canopy, where only part of a person shows. Forest is not measured here.
TERRAIN_FACTOR = {"OPEN": 1.0, "SCRUB": 1.0, "WATER": 1.0, "BUILT": 1.0, "FOREST": 0.4}


def recall_by_size(records: list[dict]) -> list[dict]:
    found, total = np.zeros(len(SIZE_BINS_PX) - 1, int), np.zeros(len(SIZE_BINS_PX) - 1, int)
    for r in records:
        alarms = [c["box"] for c in r["candidates"] if c["spark"] and c["spark"]["person"]]
        for p in r["people"]:
            b = np.searchsorted(SIZE_BINS_PX, max(p[2] - p[0], p[3] - p[1]), side="right") - 1
            total[b] += 1
            found[b] += any(hits(a, p) for a in alarms)
    return [{"px": [SIZE_BINS_PX[i], SIZE_BINS_PX[i + 1]], "found": int(found[i]), "people": int(total[i]),
             "recall": found[i] / total[i], "ci95": list(wilson(int(found[i]), int(total[i])))}
            for i in range(len(found)) if total[i] >= 5]


def pod_curve(bins: list[dict]) -> dict[str, list[float]]:
    """POD at each GSD: recall interpolated over the bins' centres (the top bin is open-ended; use its lower edge)."""
    centres = [min(b["px"][1], 2 * b["px"][0]) / 2 + b["px"][0] / 2 if b["px"][1] < 1e9 else b["px"][0] for b in bins]
    recall = [b["recall"] for b in bins]
    pixels = [NOMINAL_PERSON_M * 100 / g for g in GSD_CM]
    open_pod = np.interp(pixels, centres, recall)
    return {t: [round(float(p * f), 3) for p in open_pod] for t, f in TERRAIN_FACTOR.items()}


def false_alarms_per_tile(records: list[dict]) -> float:
    alarms = sum(1 for r in records for c in r["candidates"]
                 if c["spark"] and c["spark"]["person"] and not any(hits(c["box"], p) for p in r["touching"]))
    return alarms / len(records)


def main() -> None:
    records = load()
    bins = recall_by_size(records)
    table = {
        "source": "HERIDAL test split, 101 photos (CC BY 3.0 repack, doi:10.5281/zenodo.5662350); "
                  f"{len(records)} tiles of 1000 px at 1x, 2x and 3x shrink; {sum(b['people'] for b in bins)} people",
        "pipeline": "SAM 3.1 'person' proposals, each checked by Muse Spark on a 3x close-up",
        "nominal_person_m": NOMINAL_PERSON_M,
        "recall_by_size_px": bins,
        "false_alarms_per_tile": round(false_alarms_per_tile(records), 3),
        "terrain_factor": TERRAIN_FACTOR,
        "gsd_cm": list(GSD_CM),
        "pod_by_terrain": pod_curve(bins),
        "measured_terrain": ["OPEN", "SCRUB"],
    }
    TABLE.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"wrote {TABLE} from {RESULTS.name}")
    for b in bins:
        print(f"  {b['px'][0]:>3}-{b['px'][1]:<4g} px  {b['found']}/{b['people']}  {b['recall']:.0%}")
    for g, p in zip(GSD_CM, table["pod_by_terrain"]["OPEN"]):
        print(f"  {g:g} cm/px ({NOMINAL_PERSON_M * 100 / g:.0f} px person): POD open {p:.2f}, forest {p * 0.4:.2f}")
    print(f"  false alarms per tile {table['false_alarms_per_tile']}")


if __name__ == "__main__":
    main()
