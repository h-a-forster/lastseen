"""Turn a hypothesis ("she went west along the stream") into a map layer: a probability over cells.

A hypothesis is a few simple shapes relative to the house (the IPP), all optional and multiplied together:
- a compass direction with a spread (a cone);
- a distance band (min-max km);
- nearness to a feature type (trail, water, woods, open ground);
- a spot (bearing, distance, radius), e.g. where a witness saw someone.

Edges are soft, because witnesses round and misjudge. Each layer keeps the lost-person distance statistics
underneath: it is the base prior times the shapes, rescaled to sum to 1.
"""

import math
from dataclasses import dataclass

import numpy as np

from app.sar.grid import Terrain
from app.sar.scenario import SearchMap

EDGE_M = 150.0  # softness of band edges and feature distances
NEAR_M = 100.0  # "near a feature" means within this; linear features matter up to ~100 m (Jacobs 2016)
FEATURES = ("trail", "water", "woods", "open", "any")


@dataclass(frozen=True)
class Shape:
    direction_deg: float | None = None  # compass bearing from the house, 0 = north
    spread_deg: float | None = None  # full width of the cone
    min_km: float | None = None
    max_km: float | None = None
    near: str = "any"
    spot_bearing_deg: float | None = None
    spot_distance_km: float | None = None
    spot_radius_km: float | None = None


def shape_mask(search_map: SearchMap, shape: Shape) -> np.ndarray:
    """Weight 0..1 per cell for one hypothesis' shapes."""
    grid = search_map.grid
    rows, cols = np.indices(grid.shape)
    dy = (rows - search_map.ipp[0]) * grid.cell_m
    dx = (cols - search_map.ipp[1]) * grid.cell_m
    dist_m = np.hypot(dy, dx)
    bearing = np.degrees(np.arctan2(dx, -dy)) % 360
    mask = np.ones(grid.shape)
    if shape.direction_deg is not None:
        off = np.abs((bearing - shape.direction_deg + 180) % 360 - 180)
        half = (shape.spread_deg or 90.0) / 2
        mask *= np.where(off <= half, 1.0, np.exp(-0.5 * ((off - half) / (half / 2)) ** 2))  # full inside, soft edges
        mask[dist_m < grid.cell_m] = 1.0  # direction means nothing at the house itself
    if shape.min_km is not None:
        mask *= _soft_at_least(dist_m, shape.min_km * 1000)
    if shape.max_km is not None:
        mask *= _soft_at_least(-dist_m, -shape.max_km * 1000)
    if shape.near != "any":
        mask *= _near(search_map, shape.near)
    if shape.spot_distance_km is not None and shape.spot_bearing_deg is not None:
        b = math.radians(shape.spot_bearing_deg)
        sy, sx = -shape.spot_distance_km * 1000 * math.cos(b), shape.spot_distance_km * 1000 * math.sin(b)
        sigma = max((shape.spot_radius_km or 0.3) * 1000, grid.cell_m)
        mask *= np.exp(-0.5 * ((dy - sy) ** 2 + (dx - sx) ** 2) / sigma**2)
    return mask


def layer(search_map: SearchMap, base_poa: np.ndarray, shape: Shape) -> np.ndarray | None:
    """Base prior times the shape, summing to 1 over the area; None if the shape covers nothing inside it."""
    weighted = base_poa * shape_mask(search_map, shape)
    total = weighted.sum()
    return weighted / total if total > 1e-12 else None


def _soft_at_least(value: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(value >= threshold, 1.0, np.exp(-0.5 * ((threshold - value) / EDGE_M) ** 2))


def _near(search_map: SearchMap, feature: str) -> np.ndarray:
    target = {
        "trail": search_map.trails,
        "water": search_map.terrain == Terrain.WATER,
        "woods": search_map.terrain == Terrain.FOREST,
        "open": search_map.terrain == Terrain.OPEN,
    }[feature]
    dist_m = _distance_to(target, search_map.grid.cell_m, max_m=NEAR_M + 3 * EDGE_M)
    return np.where(dist_m <= NEAR_M, 1.0, np.exp(-0.5 * ((dist_m - NEAR_M) / EDGE_M) ** 2))


def _distance_to(target: np.ndarray, cell_m: float, max_m: float) -> np.ndarray:
    """Approximate distance (m) to the nearest target cell, by growing the target one ring of cells at a time;
    inf beyond `max_m`."""
    dist = np.where(target, 0.0, np.inf)
    reached = target.copy()
    for step in range(1, int(max_m // cell_m) + 1):
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        dist[grown & ~reached] = step * cell_m
        reached = grown
    return dist
