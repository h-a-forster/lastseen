"""Baseline planners the team's planner is judged against. Each implements `app.sar.interface.Planner`."""

from app.sar.planners.fixed_paths import Concentric, ExpandingSquare, Lawnmower, Sector
from app.sar.planners.greedy import ProbabilityGreedy

BASELINES = {p.name: p for p in (Lawnmower, ExpandingSquare, Concentric, Sector, ProbabilityGreedy)}
