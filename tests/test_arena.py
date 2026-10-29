"""The arena: correct statistics, the same subject whichever planner flies, legal routes and detours around no-fly areas."""

import json
import math

import numpy as np
import pytest

from app.agent.interpret import describe
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


def test_fixed_patterns_skip_ground_one_battery_cant_reach_and_never_stall_silently():
    from collections import deque

    from app.sar.interface import FleetSpec
    from app.sar.planners.patterns import take_sortie

    with pytest.raises(ValueError):  # an unreachable next point would ground the pattern for good
        take_sortie(deque([(0.0, 0.0)]), lambda p: math.inf, 1000.0)
    scenario = synthetic(3, category="dementia")
    short = FleetSpec(endurance_s=300.0)  # out and back to at most ~1.4 km, less than the corners of a 3 km map
    for name in ("lawnmower", "spiral", "concentric", "sector"):
        result = Sim(scenario, BASELINES[name](), fleet=short, seed=3, max_s=3600).run()
        assert result.flight_s / result.drone_s > 0.5, name


def test_fixed_patterns_keep_flying_where_a_detour_skims_a_no_fly_corner():
    from app.sar.detection import DetectionModel
    from app.sar.recorder import Recorder

    scenario = synthetic(5, category="dementia")  # these paths pass exactly over a no-fly cell's corner
    blind = DetectionModel(pod_by_terrain={t: (0.0,) * len(DetectionModel().gsd_cm) for t in Terrain})  # runs 4 h
    for name in ("lawnmower", "spiral", "concentric", "sector"):
        rec = Recorder()
        Sim(scenario, BASELINES[name](), detection=blind, seed=5, max_s=4 * 3600, recorder=rec).run()
        assert any(e["type"] == "photo" and e["t_s"] > 3.5 * 3600 for e in rec.events), name


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
        {"type": "node", "tags": {"place": "hamlet", "name": "Hamlet"}, **at(850, 500)},
        {"type": "node", "tags": {"building": "chapel", "name": "Chapel"}, **at(900, 520)},  # 54 m away, other kind
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
    assert [name for name, _, _ in m.landmarks] == ["Hamlet (hamlet)", "Chapel (chapel)"]
    assert "Chapel (chapel): 179°, 0.4 km" in describe(m)  # south, as Spark reads it
    assert place.latlon(500, 500) == pytest.approx((47.0, 7.0))  # mission files: map centre is the place's point
    assert "Named places" not in describe(synthetic(1, size=40, category="dementia").map)  # cached readings stay valid
    terrain.search_map.cache_clear()


def test_a_lost_drone_stops_flying_and_another_drone_takes_over_its_strip():
    from app.sar.detection import DetectionModel
    from app.sar.recorder import Recorder

    scenario = synthetic(3, category="dementia")
    blind = DetectionModel(pod_by_terrain={t: (0.0,) * len(DetectionModel().gsd_cm) for t in Terrain})  # runs 4 h
    rec = Recorder()
    Sim(scenario, BASELINES["lawnmower"](), detection=blind, seed=3, max_s=4 * 3600, recorder=rec,
        losses={1: 300.0}).run()
    photos = [e for e in rec.events if e["type"] == "photo"]
    assert [e["drone"] for e in rec.events if e["type"] == "drone_lost"] == [1]
    assert not [p for p in photos if p["drone"] == 1 and p["t_s"] > 300]
    seen = np.zeros(scenario.map.grid.shape, bool)
    for p in photos:
        r0, r1, c0, c1 = p["cells"]
        seen[r0:r1, c0:c1] = True
    cols = scenario.map.grid.cols
    assert seen[:, cols // 3: 2 * cols // 3].mean() > 0.9  # drone 2's strip (index 1), lost 5 minutes in
    rec = Recorder()  # a drone lost on the ground never takes off
    Sim(scenario, BASELINES["lawnmower"](), detection=blind, seed=3, max_s=1800, recorder=rec, losses={0: 0.0}).run()
    assert not [e for e in rec.events if e["type"] == "photo" and e["drone"] == 0]


def test_a_lost_drones_unflown_route_is_flown_next_in_its_order():
    from app.sar.detection import DetectionModel
    from app.sar.interface import DroneLost, DroneState, FleetSpec
    from app.sar.stats import lost_person_stats

    m = synthetic(3, category="dementia").map
    spiral = BASELINES["spiral"]()
    spiral.reset(m, FleetSpec(), DetectionModel(), lost_person_stats(), 0)
    home = m.grid.centre_m(*m.base)
    first = spiral.plan(0.0, DroneState(0, *home, 60.0, 1500.0, False, ()), ())
    spiral.observe(DroneLost(600.0, 0, tuple(first[10:])))
    nxt = spiral.plan(600.0, DroneState(1, *home, 60.0, 1500.0, False, ()), ())
    assert nxt[:len(first) - 10] == first[10:]


def test_greedy_frees_a_lost_drones_block():
    from app.sar.detection import DetectionModel
    from app.sar.interface import DroneLost, DroneState, FleetSpec
    from app.sar.stats import lost_person_stats

    m = synthetic(3, category="dementia").map
    greedy = BASELINES["greedy"]()
    greedy.reset(m, FleetSpec(), DetectionModel.load(), lost_person_stats(), 0)
    home = m.grid.centre_m(*m.base)

    def at_base(i):
        return DroneState(i, *home, 60.0, 1500.0, False, ())

    first = greedy.plan(0.0, at_base(0), ())
    assert greedy.plan(0.0, at_base(1), ()) != first  # an airborne drone keeps its block...
    greedy.observe(DroneLost(10.0, 0, ()))
    assert greedy.plan(10.0, at_base(2), ()) == first  # ...until it is lost


def test_a_height_in_the_planner_name_flies_that_planner_at_that_height():
    from app.sar.bench import fleet_for, run_seed
    from app.sar.detection import DetectionModel
    from app.sar.interface import FleetSpec

    blind = DetectionModel(pod_by_terrain={t: (0.0,) * len(DetectionModel().gsd_cm) for t in Terrain})  # full hour
    result = run_seed(3, ["spiral", "spiral@120"], FleetSpec(), 3600, "dementia", 120, detection=blind)
    assert result.runs["spiral@120"].photos < result.runs["spiral"].photos  # bigger photos, taken further apart
    with pytest.raises(ValueError):
        fleet_for("spiral@150", FleetSpec())


def test_the_bench_reports_sparks_gain_on_the_same_planner_split_by_the_sighting():
    from app.sar.bench import SeedResult, summarise
    from app.sar.sim import RunResult

    def runs(plain, spark):
        return {name: RunResult(name, "s", found, 0, 0.0, 0, 0.0, 0.0, True)
                for name, found in (("greedy@80", plain), ("greedy+spark@80", spark))}
    results = [SeedResult(0, True, runs(None, 600.0), 0, {"sighting": True}),     # Spark followed a right sighting
               SeedResult(1, True, runs(600.0, None), 0, {"sighting": False}),    # ...and a wrong one
               SeedResult(2, True, runs(600.0, 600.0), 0, {})]
    gain = summarise(results, ["greedy@80", "greedy+spark@80"], "greedy@80", 4 * 3600)["spark_gain"]["greedy+spark@80"]
    assert gain["without"] == "greedy@80" and gain["diff"]["2h"][0] == 0
    split = gain["by_sighting"]
    assert split["at"] == "2h"
    assert [split[k]["diff"][0] for k in ("sighting right", "sighting wrong", "no sighting")] == [1, -1, 0]


def test_an_empty_sortie_is_refused_rather_than_exported_at_a_made_up_height():
    from app.sar.export import qgc_plan

    with pytest.raises(ValueError):
        qgc_plan(None, (0.0, 0.0), [], 5.0)  # checked before the place is used
