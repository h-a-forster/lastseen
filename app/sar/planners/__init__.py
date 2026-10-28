"""Baseline planners the team's planner is judged against. Each implements `app.sar.interface.Planner`."""

from app.sar.planners.greedy import ProbabilityGreedy
from app.sar.planners.ipp_patterns import Concentric, ExpandingSquare, Sector
from app.sar.planners.lawnmower import Lawnmower

BASELINES = {p.name: p for p in (Lawnmower, ExpandingSquare, Concentric, Sector, ProbabilityGreedy)}
