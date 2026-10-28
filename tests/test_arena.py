"""The arena: correct statistics, the same subject whichever planner flies, legal routes and detours around no-fly areas."""

import numpy as np
import pytest

from app.sar import metrics
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
    headings = [c.true for s in range(200) for c in reports_for(synthetic(s, category="dementia"), s)[1] if c.kind == "heading"]
    assert abs(np.mean(headings) - RELIABILITY["heading"]) < 0.1
