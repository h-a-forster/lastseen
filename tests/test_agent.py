"""Checks on Spark's answers, without calling the model: invented evidence is dropped, numbers are clamped,
weights keep a share for "none of these", and shapes point the right way on the map."""

import numpy as np
import pytest

from app.agent.interpret import MIN_REST, check
from app.agent.layers import Shape, shape_mask
from app.meta_api import Answer
from app.sar.grid import Grid, Terrain
from app.sar.scenario import SearchMap

TEXT = "[0 min into the search] A neighbour saw Margaret set off towards the west. Family say Margaret walks slowly."


def flat_map(n=40):
    grid = Grid(n, n, 25.0)
    terrain = np.full((n, n), Terrain.OPEN, np.int8)
    empty = np.zeros((n, n), bool)
    return SearchMap(grid, terrain, empty, empty.copy(), (n // 2, n // 2), (n // 2, n // 2), "dementia")


def hypothesis(name, weight, evidence, **shape):
    fields = dict(direction_deg=None, spread_deg=None, min_km=None, max_km=None, near="any",
                  spot_bearing_deg=None, spot_distance_km=None, spot_radius_km=None)
    return {"name": name, "weight": weight, "reason": "r", "evidence": evidence, **(fields | shape)}


def test_cone_points_west_on_the_map():
    m = flat_map()
    mask = shape_mask(m, Shape(direction_deg=270, spread_deg=60))
    centre = m.ipp[0]
    assert mask[centre, 2] == 1.0  # due west of the house
    assert mask[centre, -3] < 0.01  # due east


def test_invented_evidence_is_dropped_and_rest_keeps_a_minimum_share():
    m = flat_map()
    base = np.full(m.grid.shape, 1 / m.grid.rows**2)
    answer = Answer({"hypotheses": [hypothesis("west", 0.95, ["set off towards the west"], direction_deg=270, spread_deg=60),
                                    hypothesis("chapel", 0.5, ["she loved the chapel"])],
                     "rest_weight": 0.0, "notes": ""}, usd=0.0, cached=True, model="test")
    result = check(m, answer, TEXT, base, 0.0)
    assert [h.name for h in result.hypotheses] == ["west"]
    assert "chapel" in result.dropped[0]
    assert result.rest_weight == pytest.approx(MIN_REST)
    assert result.hypotheses[0].weight + result.rest_weight == pytest.approx(1.0)
    assert result.mixture(base).sum() == pytest.approx(1.0)


def test_out_of_range_numbers_are_clamped():
    m = flat_map()
    base = np.full(m.grid.shape, 1 / m.grid.rows**2)
    answer = Answer({"hypotheses": [hypothesis("slow", 0.5, ["walks slowly"], min_km=5, max_km=-3, spread_deg=500,
                                               direction_deg=-90)],
                     "rest_weight": 0.5, "notes": ""}, usd=0.0, cached=True, model="test")
    shape = check(m, answer, TEXT, base, 0.0).hypotheses[0].shape
    assert shape.direction_deg == 270 and shape.spread_deg == 180
    assert shape.min_km <= shape.max_km


def west_reading(m, t_s=0.0, weight=0.85):
    from app.sar.interface import Hypotheses
    from app.agent.layers import layer

    base = np.full(m.grid.shape, 1.0)
    west = layer(m, base, Shape(direction_deg=270, spread_deg=60))
    return Hypotheses(t_s, ("west",), ("r",), (weight,), west[None], 1 - weight)


def test_greedy_replaces_readings_exactly_even_after_time_passes():
    from app.sar.interface import FleetSpec
    from app.sar.planners.greedy import ProbabilityGreedy
    from app.sar.detection import DetectionModel
    from app.sar.stats import lost_person_stats

    def greedy():
        g = ProbabilityGreedy()
        g.reset(flat_map(), FleetSpec(), DetectionModel(), lost_person_stats(), 0)
        return g

    m = flat_map()
    a, b = greedy(), greedy()
    a.observe(west_reading(m, 0.0, weight=0.85))  # an early reading, then the map spreads for an hour
    a.observe(west_reading(m, 3600.0, weight=0.3))  # the newer reading replaces it
    b.observe(west_reading(m, 3600.0, weight=0.3))  # only the newer reading
    np.testing.assert_allclose(a.probability_map(), b.probability_map())
    assert a.probability_map().sum() == pytest.approx(a.belief.poa.sum())  # the reading only moves probability
    centre = m.ipp[0]
    assert a.probability_map()[centre, 5] > a.probability_map()[centre, -6]  # west outweighs east


def test_advisor_carries_on_when_spark_fails(monkeypatch):
    from app.agent import advisor
    from app.meta_api import ModelCallError
    from app.sar.cues import Report
    from app.sar.interface import FleetSpec
    from app.sar.detection import DetectionModel
    from app.sar.planners.greedy import ProbabilityGreedy
    from app.sar.stats import lost_person_stats

    def fail(*args):
        raise ModelCallError("offline")

    monkeypatch.setattr(advisor, "interpret", fail)
    a = advisor.SparkAdvisor(ProbabilityGreedy())
    a.reset(flat_map(), FleetSpec(), DetectionModel(), lost_person_stats(), 0)
    a.observe(Report(0.0, "A neighbour saw them go west."))
    assert a.failures and not a.readings
    assert a.name == "greedy+spark"


def fake_spark(monkeypatch, data):
    from app.agent import interpret as interpret_module
    monkeypatch.setattr(interpret_module, "spark_json", lambda *a, **k: Answer(data, 0.0, True, "test"))


def test_evidence_must_quote_what_people_said_not_our_time_stamps(monkeypatch):
    from app.agent.interpret import interpret
    from app.sar.cues import Report

    m = flat_map()
    fake_spark(monkeypatch, {"hypotheses": [hypothesis("stamp", 0.9, ["into the search"], direction_deg=90, spread_deg=60)],
                             "rest_weight": 0.1, "notes": ""})
    result = interpret(m, [Report(0.0, "A neighbour saw them go west.")], np.full(m.grid.shape, 1.0))
    assert not result.hypotheses and "stamp" in result.dropped[0]


def test_advisor_carries_on_when_spark_answers_off_schema(monkeypatch):
    from app.agent import advisor
    from app.sar.cues import Report
    from app.sar.interface import FleetSpec
    from app.sar.detection import DetectionModel
    from app.sar.planners.greedy import ProbabilityGreedy
    from app.sar.stats import lost_person_stats

    fake_spark(monkeypatch, {"hypotheses": [{"name": "no other fields"}]})
    a = advisor.SparkAdvisor(ProbabilityGreedy())
    a.reset(flat_map(), FleetSpec(), DetectionModel(), lost_person_stats(), 0)
    a.observe(Report(0.0, "A neighbour saw them go west."))
    assert a.failures and not a.readings


def test_a_spiral_restarts_on_the_readings_most_likely_block_and_only_when_that_moves():
    import math

    from app.agent.layers import layer
    from app.sar.detection import DetectionModel
    from app.sar.interface import DroneState, FleetSpec, Hypotheses
    from app.sar.planners.fixed_paths import ExpandingSquare
    from app.sar.stats import lost_person_stats

    flat = flat_map(120)
    m = SearchMap(flat.grid, flat.terrain, flat.trails, flat.no_fly, (60, 10), (60, 10), "dementia")  # house near the
    house = m.grid.centre_m(*m.ipp)  # west edge, so the statistics' most likely block is east of it
    spot = (house[0], house[1] + 1000)  # a sighting 1 km east of the house
    seen_east = layer(m, np.ones(m.grid.shape), Shape(spot_bearing_deg=90, spot_distance_km=1.0, spot_radius_km=0.15))
    no_news = Hypotheses(0.0, (), (), (), np.zeros((0, *m.grid.shape)), 1.0)
    reading = Hypotheses(1800.0, ("seen east",), ("r",), (0.9,), seen_east[None], 0.1)
    spiral = ExpandingSquare()
    spiral.reset(m, FleetSpec(), DetectionModel(), lost_person_stats(), 0)
    spiral.observe(no_news)  # adds nothing to the statistics, so the pattern stays on the house
    spiral.observe(reading)
    spiral.observe(reading)  # the same reading again moves nothing, so the pattern isn't restarted twice
    route = spiral.plan(1800.0, DroneState(0, *house, 60.0, 1500.0, False, ()), ())
    assert math.dist(route[0][:2], spot) < 300 < 700 < math.dist(route[0][:2], house)
    assert [n["kind"] for n in spiral.drain_notes()] == ["restart"]
