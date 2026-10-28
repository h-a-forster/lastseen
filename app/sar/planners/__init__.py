"""Search planners. Each turns the current map into the next sortie for one drone (see `app.sar.sim.Planner`)."""

from app.sar.planners.greedy import ProbabilityGreedy
from app.sar.planners.lawnmower import Lawnmower

PLANNERS = {"lawnmower": Lawnmower, "greedy": ProbabilityGreedy}
