"""The starting probability map from lost-person behaviour: distance rings around the initial planning point (IPP).

Lost-person statistics give, per subject category, the distances from the IPP within which 25, 50, 75 and 95% of
subjects were found. Each ring gets that share of probability, spread over its cells by a terrain weight
(trails up, open water down). The part of each ring that falls outside the area goes to rest of world.
"""

import numpy as np

from app.sar.grid import Grid
from app.sar.stats import lost_person_stats

# Straight-line distance from the IPP (km) within which 25/50/75/95% of subjects were found (Koester 2008, ISRID).
RINGS_KM: dict[str, tuple[float, ...]] = {
    name: tuple(c["rings_km"]) for name, c in lost_person_stats()["categories"].items()
}

_RING_MASS = (0.25, 0.25, 0.25, 0.20, 0.05)


def ring_prior(grid: Grid, ipp: tuple[int, int], rings_km: tuple[float, ...], weight: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Return (poa, rest_of_world). `weight` is a per-cell terrain/behaviour multiplier (default uniform)."""
    dist = grid.distance_km(*ipp)
    weight = np.ones(grid.shape) if weight is None else weight
    edges = (0.0, *rings_km, 2 * rings_km[-1])
    cell_km2 = (grid.cell_m / 1000) ** 2
    poa = np.zeros(grid.shape)
    row = 0.0
    for mass, inner, outer in zip(_RING_MASS, edges[:-1], edges[1:]):
        in_ring = (dist >= inner) & (dist < outer)
        ring_km2 = np.pi * (outer**2 - inner**2)
        inside_share = min(in_ring.sum() * cell_km2 / ring_km2, 1.0)
        row += mass * (1 - inside_share)
        w = weight * in_ring
        if w.sum() > 0:
            poa += mass * inside_share * w / w.sum()
        else:
            row += mass * inside_share
    return poa, row
