"""The team's planner: multi-block sorties that reuse greedy's map stack but pack the battery."""

import math

import numpy as np

from app.agent.layers import Shape, layer
from app.planner import SearchPlanner
from app.planner.hedge import Hedge
from app.sar.detection import DetectionModel
from app.sar.grid import Grid, Terrain
from app.sar.interface import DroneLost, DroneState, FleetSpec, Hypotheses
from app.sar.planners.fixed_paths import ExpandingSquare
from app.sar.planners.greedy import ProbabilityGreedy
from app.sar.planners.patterns import path_length
from app.sar.scenario import SearchMap, synthetic
from app.sar.stats import lost_person_stats


def _at_base(planner, drone_id: int = 0) -> DroneState:
    y, x = planner._home
    return DroneState(drone_id, y, x, planner.fleet.alt_m, planner.fleet.endurance_s, False, ())


def test_ours_packs_multiple_adjacent_blocks_into_one_sortie():
    """Without multi-block packing this fails: a single greedy-style block leaves most of the battery unused."""
    m = synthetic(3, category="dementia").map
    ours = SearchPlanner()
    ours.reset(m, FleetSpec(), DetectionModel.load(), lost_person_stats(), 0)
    route = ours.plan(0.0, _at_base(ours), ())
    assert route, "ours must fly"
    assert len(ours._claimed[0]) >= 2, "sortie should claim a multi-block tour, not one myopic block"


def test_ours_sortie_is_longer_than_greedys_on_the_same_map():
    m = synthetic(3, category="dementia").map
    fleet, stats, det = FleetSpec(), lost_person_stats(), DetectionModel.load()
    ours, greedy = SearchPlanner(), ProbabilityGreedy()
    ours.reset(m, fleet, det, stats, 0)
    greedy.reset(m, fleet, det, stats, 0)
    home = m.grid.centre_m(*m.base)
    ours_route = [(y, x) for y, x, _ in ours.plan(0.0, _at_base(ours), ())]
    greedy_route = [(y, x) for y, x, _ in greedy.plan(0.0, DroneState(0, *home, fleet.alt_m, fleet.endurance_s, False, ()), ())]
    assert path_length([home, *ours_route, home]) > path_length([home, *greedy_route, home])


def test_ours_frees_every_block_when_a_drone_is_lost():
    m = synthetic(3, category="dementia").map
    ours = SearchPlanner()
    ours.reset(m, FleetSpec(), DetectionModel.load(), lost_person_stats(), 0)
    first = ours.plan(0.0, _at_base(ours, 0), ())
    assert ours.plan(0.0, _at_base(ours, 1), ()) != first
    ours.observe(DroneLost(10.0, 0, ()))
    assert ours.plan(10.0, _at_base(ours, 2), ()) == first


def _flat_map() -> SearchMap:
    """3 km of open ground, house in the middle."""
    grid = Grid(120, 120, 25.0)
    empty = np.zeros(grid.shape, bool)
    return SearchMap(grid, np.full(grid.shape, Terrain.OPEN, np.int8), empty, empty.copy(), (60, 60), (60, 60),
                     "dementia")


def _sighting_east(m: SearchMap, t_s: float) -> Hypotheses:
    """A reading that puts the person near a spot 1 km east of the house."""
    seen = layer(m, np.ones(m.grid.shape), Shape(spot_bearing_deg=90, spot_distance_km=1.0, spot_radius_km=0.15))
    return Hypotheses(t_s, ("seen east",), ("r",), (0.9,), seen[None], 0.1)


def _on_the_ground(m: SearchMap, drone: int) -> DroneState:
    return DroneState(drone, *m.grid.centre_m(*m.base), 100.0, 1500.0, False, ())


def _closer_to_the_sighting(m: SearchMap, route) -> bool:
    house = m.grid.centre_m(*m.ipp)
    centre = np.mean([w[:2] for w in route], axis=0)
    return math.dist(centre, (house[0], house[1] + 1000)) < math.dist(centre, house)


def _hedge(m: SearchMap) -> Hedge:
    hedge = Hedge()
    hedge.reset(m, FleetSpec(), DetectionModel(), lost_person_stats(), 0)
    return hedge


def test_hedge_flies_the_spiral_until_a_sighting_even_with_the_brief_read():
    m = _flat_map()
    hedge, spiral = _hedge(m), ExpandingSquare()
    spiral.reset(m, FleetSpec(), DetectionModel(), lost_person_stats(), 0)
    hedge.observe(_sighting_east(m, 0.0))  # the brief
    for drone in range(3):
        assert hedge.plan(0.0, _on_the_ground(m, drone), ()) == spiral.plan(0.0, _on_the_ground(m, drone), ())


def test_a_sighting_sends_one_drone_after_it_at_once_and_the_others_keep_sweeping():
    m = _flat_map()
    hedge = _hedge(m)
    pieces = [hedge.plan(0.0, _on_the_ground(m, drone), ()) for drone in range(3)]
    hedge.observe(_sighting_east(m, 1800.0))
    sweeper = DroneState(0, *pieces[0][2][:2], 100.0, 1000.0, True, tuple(pieces[0][3:]))
    assert hedge.retask(1800.0, sweeper, ()) is None
    route = hedge.retask(1800.0, DroneState(2, *pieces[2][2][:2], 100.0, 1000.0, True, tuple(pieces[2][3:])), ())
    assert _closer_to_the_sighting(m, route)
    assert not _closer_to_the_sighting(m, hedge.plan(1800.0, _on_the_ground(m, 0), ()))


def test_when_the_hunter_is_lost_another_drone_hunts_and_its_route_stays_off_the_pattern():
    m = _flat_map()
    hedge = _hedge(m)
    for drone in range(3):
        hedge.plan(0.0, _on_the_ground(m, drone), ())
    hedge.observe(_sighting_east(m, 1800.0))
    hunt = hedge.plan(1800.0, _on_the_ground(m, 2), ())
    hedge.observe(DroneLost(1900.0, 2, tuple(hunt[1:])))
    assert _closer_to_the_sighting(m, hedge.plan(1900.0, _on_the_ground(m, 1), ()))
    assert hedge.plan(1900.0, _on_the_ground(m, 0), ())[0] != hunt[1]
