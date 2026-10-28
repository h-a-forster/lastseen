"""Probability of detection (POD) for one nadir photo, by ground sample distance and terrain.

The table is meant to be measured, not chosen: `calibrate.py` runs the perception pipeline on labelled real
aerial photos and writes the recall per GSD band and terrain class. Until that has run, the placeholder table
below is used and every result is marked uncalibrated.
"""

from dataclasses import dataclass, field

import numpy as np

from app.sar.grid import Terrain

# Prior estimate of POD per terrain at each GSD (cm/px), before calibration: zero-shot detection results on small
# aerial people extrapolated by pixels on target (a lying adult is ~85 px long at 2 cm/px, ~34 px at 5 cm/px), with
# canopy cutting recall to roughly 0.4x. Replaced by the measured table.
_PLACEHOLDER_GSD = (1.0, 2.0, 3.0, 5.0)
_PLACEHOLDER_POD = {
    Terrain.OPEN: (0.85, 0.65, 0.45, 0.25),
    Terrain.SCRUB: (0.77, 0.57, 0.37, 0.17),
    Terrain.FOREST: (0.35, 0.25, 0.18, 0.10),
    Terrain.WATER: (0.85, 0.65, 0.45, 0.25),
    Terrain.BUILT: (0.70, 0.50, 0.32, 0.15),
}


@dataclass(frozen=True)
class DetectionModel:
    gsd_cm: tuple[float, ...] = _PLACEHOLDER_GSD
    pod_by_terrain: dict[Terrain, tuple[float, ...]] = field(default_factory=lambda: dict(_PLACEHOLDER_POD))
    calibrated: bool = False

    def pod(self, gsd_cm: float, terrain: np.ndarray) -> np.ndarray:
        """POD for each cell of `terrain` seen in one photo taken at `gsd_cm`."""
        by_class = np.zeros(len(Terrain))
        for t in Terrain:
            by_class[t] = np.interp(gsd_cm, self.gsd_cm, self.pod_by_terrain[t])
        return by_class[terrain]
