"""Invariants of the search maths: probability is conserved and updates move it the right way."""

import numpy as np
import pytest

from app.sar.belief import Belief
from app.sar.grid import Camera, Grid
from app.sar.prior import ring_prior


def uniform_belief(n=10, row=0.2):
    return Belief(np.full((n, n), (1 - row) / n**2), row)


def test_miss_moves_probability_out_of_the_searched_cells():
    b = uniform_belief()
    cells = (slice(0, 5), slice(0, 5))
    before = b.poa[cells].sum()
    pos = b.observe_miss(cells, np.full((5, 5), 0.6))
    assert pos == pytest.approx(before * 0.6)
    assert b.poa[cells].sum() < before
    assert b.poa[7, 7] > (1 - 0.2) / 100 and b.row > 0.2
    assert b.total() == pytest.approx(1.0)


def test_miss_with_certain_detection_everywhere_is_rejected():
    b = Belief(np.full((2, 2), 0.25), 0.0)
    with pytest.raises(ValueError):
        b.observe_miss((slice(0, 2), slice(0, 2)), np.ones((2, 2)))


def test_predict_conserves_probability_and_leaks_to_rest_of_world_at_edges():
    b = Belief(np.zeros((5, 5)), 0.0)
    b.poa[0, 0] = 1.0
    b.predict(0.5, np.ones((5, 5)))
    assert b.total() == pytest.approx(1.0)
    assert b.row == pytest.approx(0.25)  # two of four neighbours are off the map
    assert b.poa[0, 1] == pytest.approx(0.125)


def test_impassable_cells_receive_nothing():
    passability = np.ones((3, 3))
    passability[1, 1] = 0.0
    b = Belief(np.zeros((3, 3)), 0.0)
    b.poa[0, 1] = 1.0
    b.predict(0.5, passability)
    assert b.poa[1, 1] == 0.0
    assert b.total() == pytest.approx(1.0)


def test_ring_prior_sums_to_one_and_spills_outside_small_areas():
    grid = Grid(40, 40, 25.0)  # 1 km square
    poa, row = ring_prior(grid, (20, 20), (0.3, 0.8, 1.6, 3.5))
    assert poa.sum() + row == pytest.approx(1.0)
    assert row > 0.4  # most of a 3.5 km ring lies outside a 1 km area


def test_footprint_and_gsd():
    cam = Camera()
    width, height = cam.footprint_m(80)
    assert width > height > 0
    assert 1.0 < cam.gsd_cm(80) < 3.0
    rows, cols = Grid(100, 100, 25.0).footprint(1250, 1250, width, height)
    assert (cols.stop - cols.start) * 25 == pytest.approx(width, abs=50)
    assert Grid(10, 10, 25.0).footprint(0, 0, 100, 100) == (slice(0, 2), slice(0, 2))
