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
