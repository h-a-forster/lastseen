"""The arena's statistics and truth: correct intervals and curves, and the same subject whichever planner flies."""

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
