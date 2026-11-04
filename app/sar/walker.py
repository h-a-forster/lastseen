"""The hidden truth: where the missing person actually goes.

The walker follows published behaviour rules rather than the statistics planners use, so a planner is never tested
on its own prior. Dementia subjects keep going in one direction until they get stuck (in brush, against water) or
tire; they deflect off barriers instead of turning back (Koester, Lost Person Behavior). Hikers pick a behaviour each
step from the mix fitted to ISRID hikers (Hashimoto et al. 2022): direction travel, route travel along trails,
random walk, staying put. Speed scales with how easy the ground is.

The parameters below are fitted so that end points match the published distance rings; `python -m
app.sar.walker_check` prints the comparison (distance, dispersion angle, distance from trails).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from app.sar.grid import Grid, Terrain

if TYPE_CHECKING:
    from app.sar.scenario import SearchMap

DT_S = 20.0
RECORD_S = 60.0
BARRIER = 0.1  # passability below this blocks walking (open water)
OUTSIDE_PASSABILITY = 0.6  # terrain beyond the mapped area is unknown; assume average going
SET_OFF_M = 300.0  # how far along her way a neighbour's "she set off towards..." describes
SNAP_M = 150.0  # a point a visitor puts on a river or pond means the bank, if one is this close
GUIDED_PACE = (0.6, 1.65)  # slowest and quickest pace of a guided walk, as a share of the category's speed (+-2 sd)
_STEPS = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
_STEP_ANGLE = np.arctan2(_STEPS[:, 0], _STEPS[:, 1])  # heading convention: dy = sin, dx = cos


@dataclass(frozen=True)
class WalkerParams:
    speed_mps: float  # walking speed on easy ground, before the per-person spread
    turn_sd_deg: float  # heading drift per minute
    stuck_per_min: float  # chance per minute in scrub or forest of getting stuck for good
    mix: dict[str, float]  # per-step behaviour: direction, route, random, stay
    moving_h: float  # median total time spent moving (log-normal)
    moving_sd: float  # log-normal spread of moving time
    report_delay_h: tuple[float, float]  # missing for this long (uniform range) before the search starts


# Published ring distances are close to log-normal (dementia spread ~1.4, hikers ~1.1), so moving time is drawn
# log-normally; its median and the drift are fitted to the rings and dispersion angles with walker_check.
# Report delays are assumptions: dementia is usually noticed within hours, hikers are reported overdue at day's end.
PARAMS = {
    "dementia": WalkerParams(0.6, 8.0, 0.0005, {"direction": 1.0}, 1.3, 1.4, (0.5, 2.0)),
    # Mix: Hashimoto 2022, with view-enhancing folded into stay (synthetic maps have no elevation).
    "hiker": WalkerParams(1.2, 5.0, 0.0, {"direction": 0.559, "route": 0.377, "random": 0.055, "stay": 0.009},
                          1.8, 1.0, (2.0, 6.0)),
}


@dataclass(frozen=True)
class Walk:
    start_heading: np.ndarray  # (n,) radians
    positions: np.ndarray  # (records, n, 2) metres (y, x), one record per RECORD_S; may leave the mapped area


def walk(search_map: SearchMap, category: str, rng: np.random.Generator, n: int, duration_s: float,
         params: WalkerParams | None = None, start_m: tuple[float, float] | None = None,
         heading: float | None = None, moving_s: float | None = None) -> Walk:
    """Walk `n` subjects for `duration_s` seconds, from the IPP in random directions unless `start_m` and `heading`
    say otherwise; `moving_s` replaces the drawn time each keeps moving."""
    p = params or PARAMS[category]
    grid, passability, trails = search_map.grid, search_map.passability, search_map.trails
    stuck_terrain = np.isin(search_map.terrain, (Terrain.SCRUB, Terrain.FOREST))
    pos = np.tile(np.array(start_m or grid.centre_m(*search_map.ipp)), (n, 1))
    start_heading = rng.uniform(0, 2 * np.pi, n) if heading is None else np.full(n, heading)
    heading = start_heading.copy()
    speed = p.speed_mps * rng.lognormal(0, 0.25, n)
    moving_left_s = p.moving_h * 3600 * rng.lognormal(0, p.moving_sd, n)
    if moving_s is not None:
        moving_left_s[:] = moving_s
    modes = list(p.mix)
    probs = np.array([p.mix[m] for m in modes]) / sum(p.mix.values())
    drift = math.radians(p.turn_sd_deg) * math.sqrt(DT_S / 60)
    records = []
    for k in range(int(duration_s / DT_S) + 1):
        if k % int(RECORD_S / DT_S) == 0:
            records.append(pos.copy())
        r, c, inside = _cells(grid, pos)
        mode = rng.choice(len(modes), n, p=probs)
        heading += rng.normal(0, drift, n)
        if "route" in p.mix:
            on_trail = (mode == modes.index("route")) & inside & trails[r, c]
            heading[on_trail] = _along_trail(trails, r[on_trail], c[on_trail], heading[on_trail])
        direction = heading.copy()
        if "random" in p.mix:  # one step in a random direction; the intended heading is kept
            rand = mode == modes.index("random")
            direction[rand] = rng.uniform(0, 2 * np.pi, rand.sum())
        ease = np.where(inside, passability[r, c], OUTSIDE_PASSABILITY)
        new = pos + (speed * DT_S * ease)[:, None] * np.stack([np.sin(direction), np.cos(direction)], axis=1)
        nr, nc, new_inside = _cells(grid, new)
        blocked = new_inside & (passability[nr, nc] < BARRIER)
        heading[blocked] += rng.choice((-1, 1), blocked.sum()) * np.pi / 2  # deflect; they don't turn back
        stay = (mode == modes.index("stay")) if "stay" in p.mix else np.zeros(n, bool)
        moving = (moving_left_s > 0) & ~blocked & ~stay
        pos[moving] = new[moving]
        stuck = inside & stuck_terrain[r, c] & (rng.random(n) < p.stuck_per_min * DT_S / 60)
        moving_left_s -= DT_S
        moving_left_s[stuck] = 0.0
    return Walk(start_heading, np.array(records))


def truth_track(search_map: SearchMap, category: str, rng: np.random.Generator, hours: float,
                params: WalkerParams | None = None) -> tuple[np.ndarray, float]:
    """One subject: positions (records, 2) from the moment the search starts, one per RECORD_S, and the heading
    (radians) they set off in, which a witness may have seen. `params` overrides the category's behaviour."""
    p = params or PARAMS[category]
    delay_s = rng.uniform(*p.report_delay_h) * 3600
    w = walk(search_map, category, rng, 1, delay_s + hours * 3600, p)
    return w.positions[int(delay_s // RECORD_S):, 0], float(w.start_heading[0])


def guided_track(search_map: SearchMap, category: str, rng: np.random.Generator, hours: float,
                 spot_m: tuple[float, float], seen_s: float, params: WalkerParams | None = None
                 ) -> tuple[np.ndarray, float]:
    """Like `truth_track`, for a subject a witness saw at `spot_m` at search time `seen_s` (a whole number of
    RECORD_S). She walks the easiest way there from the IPP and arrives as the witness sees her: her pace is set by how
    long she has been gone (the usual report delay plus `seen_s`), within GUIDED_PACE. A spot too far for her quickest
    pace means she has been gone longer; one too near for her slowest means she was missed sooner, and only if even that
    isn't enough does she sit there until she is seen. From the spot she carries on by the category's rules for whatever
    is left of her moving time (none, if getting there used it up). The heading is the way she went for her first
    SET_OFF_M, which is what a neighbour would have seen."""
    p = params or PARAMS[category]
    path = easiest_path(search_map, spot_m)
    steps = np.diff(path, axis=0)
    rows, cols, _ = _cells(search_map.grid, path[:-1])
    effort_m = np.concatenate([[0.0], np.cumsum(np.hypot(*steps.T) / search_map.passability[rows, cols])])
    gone_s = rng.uniform(*p.report_delay_h) * 3600 + seen_s
    speed = p.speed_mps * np.clip(effort_m[-1] / gone_s / p.speed_mps, *GUIDED_PACE)
    walked_s = effort_m / speed
    delay_s = max(0.0, walked_s[-1] - seen_s)
    moving_s = p.moving_h * 3600 * rng.lognormal(0, p.moving_sd)  # in all, as in `walk`
    clock = np.arange(0.0, seen_s, RECORD_S) + delay_s  # her walking time at each record before she is seen
    before = np.stack([np.interp(clock, walked_s, path[:, 0]), np.interp(clock, walked_s, path[:, 1])], axis=1)
    last = steps[-1] if len(steps) else np.zeros(2)
    after = walk(search_map, category, rng, 1, hours * 3600 - seen_s, p, start_m=tuple(path[-1]),
                 heading=math.atan2(*last), moving_s=max(0.0, moving_s - walked_s[-1])).positions[:, 0]
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*steps.T))])
    set_off = path[min(int(np.searchsorted(along, SET_OFF_M)), len(path) - 1)] - path[0]
    return np.concatenate([before, after]), math.atan2(*set_off)


def easiest_path(search_map: SearchMap, spot_m: tuple[float, float]) -> np.ndarray:
    """Points (y, x) in metres from the IPP's centre to `spot_m`, through cell centres along the walk that costs least
    (distance over passability), never into water (only out of it, when the start point is on water). A spot on water
    moves to the nearest bank within SNAP_M. ValueError if there is none, or water cuts the spot off."""
    grid, passability = search_map.grid, search_map.passability
    goal = grid.nearest(passability >= BARRIER, grid.cell_of(*spot_m), SNAP_M)
    if goal is None:
        raise ValueError("the witness's spot is on water; mark it on land")
    cost = np.full(grid.shape, math.inf)
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost[search_map.ipp] = 0.0
    frontier = [(0.0, search_map.ipp)]
    while frontier:
        d, (r, c) = heapq.heappop(frontier)
        if (r, c) == goal:
            break
        if d > cost[r, c]:
            continue
        wading = passability[r, c] < BARRIER  # only near a start point on water: she wades out, never back in
        for dr, dc in _STEPS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
                continue
            if not wading and min(passability[nr, nc], passability[r + dr, c], passability[r, c + dc]) < BARRIER:
                continue  # no stepping into water or cutting the corner of a water cell
            step = d + math.hypot(dr, dc) * grid.cell_m / passability[r, c]
            if step < cost[nr, nc]:
                cost[nr, nc], came_from[(nr, nc)] = step, (r, c)
                heapq.heappush(frontier, (step, (nr, nc)))
    if math.isinf(cost[goal]):
        raise ValueError("water cuts the witness's spot off from the starting point; mark a spot she could walk to")
    cells = [goal]
    while cells[-1] != search_map.ipp:
        cells.append(came_from[cells[-1]])
    end = tuple(spot_m) if goal == grid.cell_of(*spot_m) else grid.centre_m(*goal)
    return np.array([grid.centre_m(*cell) for cell in reversed(cells)] + [end])


def _cells(grid: Grid, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell indices (clipped, for safe lookups) and whether each position is inside the mapped area."""
    r = np.floor(pos[:, 0] / grid.cell_m).astype(int)
    c = np.floor(pos[:, 1] / grid.cell_m).astype(int)
    inside = (r >= 0) & (r < grid.rows) & (c >= 0) & (c < grid.cols)
    return np.clip(r, 0, grid.rows - 1), np.clip(c, 0, grid.cols - 1), inside


def _along_trail(trails: np.ndarray, r: np.ndarray, c: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Turn onto the neighbouring trail cell closest to the current heading, never back the way they came."""
    nr = np.clip(r[:, None] + _STEPS[:, 0], 0, trails.shape[0] - 1)
    nc = np.clip(c[:, None] + _STEPS[:, 1], 0, trails.shape[1] - 1)
    score = np.where(trails[nr, nc], np.cos(_STEP_ANGLE[None, :] - heading[:, None]), -np.inf)
    best = score.argmax(axis=1)
    keep = score[np.arange(len(r)), best] > 0
    return np.where(keep, _STEP_ANGLE[best], heading)
