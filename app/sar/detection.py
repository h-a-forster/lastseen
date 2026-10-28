"""Probability of detection (POD) for one nadir photo, by ground sample distance and terrain.

The table is meant to be measured, not chosen: `calibrate.py` runs the perception pipeline on labelled real
aerial photos and writes the recall per GSD band and terrain class. Until that has run, the placeholder table
below is used and every result is marked uncalibrated.
"""

from dataclasses import dataclass, field

import numpy as np

from app.sar.grid import Terrain

# Placeholder POD per terrain at each GSD (cm/px). Replaced by calibration output.
_PLACEHOLDER_GSD = (1.0, 2.0, 3.0, 5.0)
_PLACEHOLDER_POD = {
    Terrain.OPEN: (0.90, 0.85, 0.70, 0.45),
    Terrain.SCRUB: (0.75, 0.65, 0.50, 0.30),
    Terrain.FOREST: (0.40, 0.30, 0.20, 0.10),
    Terrain.WATER: (0.85, 0.80, 0.65, 0.40),
    Terrain.BUILT: (0.60, 0.50, 0.35, 0.20),
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
