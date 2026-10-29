"""The team's planner: multi-block sorties that reuse greedy's map stack but pack the battery."""

from app.planner import SearchPlanner
from app.sar.detection import DetectionModel
from app.sar.interface import DroneLost, DroneState, FleetSpec
from app.sar.planners.greedy import ProbabilityGreedy
from app.sar.planners.patterns import path_length
from app.sar.scenario import synthetic
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
