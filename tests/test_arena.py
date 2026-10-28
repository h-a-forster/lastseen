"""The arena: correct statistics, the same subject whichever planner flies, legal routes and detours around no-fly areas."""

import json
import math

import numpy as np
import pytest

from app.sar import metrics, terrain
from app.sar.grid import Terrain
from app.sar.planners import BASELINES
from app.sar.scenario import synthetic
from app.sar.sim import Sim


def test_wilson_interval_contains_the_rate_and_stays_in_bounds():
    lo, hi = metrics.wilson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    lo, hi = metrics.wilson(10, 20)
    assert lo < 0.5 < hi


def test_time_to_find_curve_is_the_cumulative_find_fraction():
    curve = metrics.time_to_find_curve([60.0, None, 120.0, None])
    assert curve == [(0.0, 0.0), (60.0, 0.25), (120.0, 0.5)]
    assert metrics.median_time([60.0, None, 120.0, None]) == 120.0
    assert metrics.median_time([None, None, 5.0]) is None


def test_paired_difference_of_identical_planners_is_zero():
    times = [60.0, None, 300.0, None, 90.0]
    diff, lo, hi = metrics.paired_difference(times, times, 3600, np.random.default_rng(0))
    assert diff == lo == hi == 0.0


def test_truth_track_is_fixed_by_the_seed_and_not_by_the_planner():
    a, b = synthetic(7, category="dementia"), synthetic(7, category="dementia")
    np.testing.assert_array_equal(a.track, b.track)
    for name in ("lawnmower", "greedy"):
        Sim(a, BASELINES[name](), seed=7, max_s=1800).run()
    np.testing.assert_array_equal(a.track, b.track)


def test_every_baseline_flies_legal_routes():
    scenario = synthetic(3, category="dementia")
    for name, planner in BASELINES.items():
        result = Sim(scenario, planner(), seed=3, max_s=3600).run()
        assert result.forced_returns == 0, name
        assert result.photos > 0, name


def test_detours_go_around_no_fly_and_cost_the_extra_distance():
    from app.sar.grid import Grid
    from app.sar.routing import distance_field, is_clear, path_length, safe_path

    grid = Grid(20, 20, 25.0)
    no_fly = np.zeros(grid.shape, bool)
    no_fly[2:18, 9:11] = True  # a wall with gaps at the top and bottom
    a, b = grid.centre_m(10, 2), grid.centre_m(10, 17)
    path = safe_path([a, b], no_fly, grid)
    assert all(is_clear(p, q, no_fly, grid) for p, q in zip(path, path[1:]))
    assert path_length(path) > 1.5 * path_length([a, b])
    assert distance_field(grid, no_fly, (10, 2))[10, 17] == pytest.approx(path_length(path), rel=0.15)


def test_reports_are_fixed_by_the_seed_and_mistaken_witnesses_happen_at_the_set_rate():
    from app.sar.cues import RELIABILITY, reports_for

    first = reports_for(synthetic(5, category="dementia"), 5)
    assert first == reports_for(synthetic(5, category="dementia"), 5)
    headings = [c.true for s in range(60) for c in reports_for(synthetic(s, category="dementia"), s)[1] if c.kind == "heading"]
    assert abs(np.mean(headings) - RELIABILITY["heading"]) < 0.15  # about 2.5 standard errors at 60 scenarios


def test_a_true_sighting_never_describes_a_time_before_the_track_starts():
    from app.sar.cues import reports_for

    scenario = synthetic(174, category="dementia")  # its sighting was drawn older than the search itself
    sighting = next(c for c in reports_for(scenario, 174)[1] if c.kind == "sighting")
    assert sighting.true
    assert sighting.detail["minutes_ago"] * 60 <= sighting.t_s + 150  # rounding to 5 minutes
    assert sighting.detail["distance_km"] < 3  # not where the person ends up 8 hours later


def test_forced_return_prices_the_detour_around_a_long_no_fly_wall():
    from app.sar.grid import Grid
    from app.sar.interface import FleetSpec
    from app.sar.scenario import Scenario, SearchMap

    grid = Grid(200, 200, 25.0)
    no_fly = np.zeros(grid.shape, bool)
    no_fly[5:195, 20:22] = True  # a 4.75 km wall beside the base, with gaps only at its ends
    flat = np.zeros(grid.shape, np.int8)
    m = SearchMap(grid, flat, np.zeros(grid.shape, bool), no_fly, (100, 10), (100, 10), "dementia")
    nobody = Scenario("wall", m, np.array([[-5000.0, -5000.0]]), 0.0)

    class BehindTheWall:
        name = "behind"
        def reset(self, *args): self.sent = False
        def observe(self, event): pass
        def probability_map(self): return None
        def plan(self, t_s, drone, fleet):
            if self.sent:
                return []
            self.sent = True
            return [(2512.5, 800.0 + 25 * (k % 2), 60.0) for k in range(400)]  # patrol just behind the wall

    class Watched(Sim):
        lowest = math.inf
        def _move(self, d):
            Watched.lowest = min(Watched.lowest, d.battery_s)
            return super()._move(d)

    result = Watched(nobody, BehindTheWall(), FleetSpec(drones=1), max_s=3000).run()
    assert result.forced_returns == 1
    assert Watched.lowest >= 0


def test_planners_cannot_regenerate_the_truth_or_change_the_map():
    scenario = synthetic(5, category="dementia")

    class Snoop:
        name = "snoop"
        def reset(self, search_map, fleet, detection, stats, seed):
            self.seed = seed
            with pytest.raises(ValueError):
                search_map.no_fly[0, 0] = True
        def observe(self, event): pass
        def plan(self, t_s, drone, fleet): return []
        def probability_map(self): return None

    snoop = Snoop()
    Sim(scenario, snoop, seed=5, max_s=60).run()
    assert not np.array_equal(synthetic(snoop.seed, category="dementia").track, scenario.track)


def test_false_alarms_are_counted_without_moving_any_find():
    import dataclasses

    from app.sar.detection import DetectionModel

    quiet = dataclasses.replace(DetectionModel.load(), alarms_per_tile=0.0)
    noisy = dataclasses.replace(quiet, alarms_per_tile=0.05)
    for seed in (4, 8, 13):
        scenario = synthetic(seed, category="dementia")
        for name in ("lawnmower", "greedy"):
            a = Sim(scenario, BASELINES[name](), detection=quiet, seed=seed, max_s=3 * 3600).run()
            b = Sim(scenario, BASELINES[name](), detection=noisy, seed=seed, max_s=3 * 3600).run()
            assert (a.found_s, a.photos) == (b.found_s, b.photos), (seed, name)
            assert a.alarms == 0 < b.alarms


def test_imagery_mode_finds_only_when_the_real_photo_answer_did():
    from app.perception.pool import Shot
    from app.sar.detection import DetectionModel
    from app.sar.grid import Terrain

    class Pool:
        def __init__(self, detected):
            self.detected = detected

        def shot(self, gsd_cm, rng):
            return Shot("0" * 16, (0, 0, 20, 60), 60, self.detected, 0.9, "person")

    table = DetectionModel.load()
    no_canopy = DetectionModel(table.gsd_cm, {t: table.pod_by_terrain[Terrain.OPEN] for t in Terrain}, True)
    scenario = synthetic(25, category="dementia")
    missed = Sim(scenario, BASELINES["lawnmower"](), detection=no_canopy, seed=25, photos=Pool(False)).run()
    found = Sim(scenario, BASELINES["lawnmower"](), detection=no_canopy, seed=25, photos=Pool(True)).run()
    assert missed.found_s is None and missed.in_frame > 0
    assert found.found_s is not None and found.in_frame == 1


def test_real_map_joins_split_boundaries_keeps_holes_and_puts_north_at_row_zero(tmp_path, monkeypatch):
    """A forest whose boundary OSM split into three ways (one drawn backwards), with a clearing cut out of it, in the
    northern half of a 1 km place; a path crosses the south. No network: the download is already on disk."""
    place = terrain.Place("test", 47.0, 7.0, side_m=1000.0)
    b = place.bounds()
    at = lambda north_m, east_m: {"lat": b["north"] - north_m / terrain.M_PER_DEG,
                                  "lon": b["west"] + east_m * (b["east"] - b["west"]) / 1000}
    outer = [[at(100, 100), at(100, 900)], [at(450, 100), at(450, 900), at(100, 900)], [at(450, 100), at(100, 100)]]
    hole = [at(200, 400), at(200, 600), at(350, 600), at(350, 400), at(200, 400)]
    osm = {"elements": [
        {"type": "relation", "tags": {"landuse": "forest"}, "members": [
            *({"role": "outer", "geometry": part} for part in outer), {"role": "inner", "geometry": hole}]},
        {"type": "way", "tags": {"highway": "path"}, "geometry": [at(800, 0), at(800, 1000)]},
    ]}
    monkeypatch.setattr(terrain, "PLACES_DIR", tmp_path)
    monkeypatch.setitem(terrain.PLACES, "test", place)
    (tmp_path / "test.osm.json").write_text(json.dumps(osm), encoding="utf-8")
    terrain.search_map.cache_clear()
    m = terrain.search_map("test")
    assert m.terrain.shape == (40, 40)
    assert m.terrain[6, 6] == Terrain.FOREST and m.terrain[16, 30] == Terrain.FOREST  # either side of the split
    assert m.terrain[11, 20] == Terrain.OPEN  # the clearing
    assert m.terrain[30, 20] == Terrain.OPEN and m.trails[32, 20] and not m.trails[20, 2]
    terrain.search_map.cache_clear()
