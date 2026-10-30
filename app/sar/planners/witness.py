"""A careful volunteer without LastSeen: restart the expanding square where the witness said.

When a sighting comes in, a team flying the expanding square restarts it on the spot the witness named. This baseline
reads that spot straight from the report's words (distance and direction from the house) and restarts the square
there, the same way `spiral+spark` restarts on Spark's reading. It is written for the simulator's report template, so
it reads every sighting exactly and takes the witness at their word; set against `spiral+spark` and the team's
planner, it asks whether Spark's reading does better than that.
"""

import re

from app.agent.layers import Shape, layer
from app.sar.cues import COMPASS
from app.sar.interface import Hypotheses, Observation, Report
from app.sar.planners.fixed_paths import ExpandingSquare

_DIRECTION = "|".join(sorted(COMPASS, key=len, reverse=True))  # "north-east" before "north"
SIGHTING = re.compile(rf"roughly ([\d.]+) km ({_DIRECTION}) of the house")


class WitnessSquare(ExpandingSquare):
    name = "spiral+witness"

    def observe(self, event: Observation) -> None:
        super().observe(event)
        if isinstance(event, Report) and (said := SIGHTING.search(event.text)):
            spot = layer(self.map, self._base_poa, Shape(spot_bearing_deg=45 * COMPASS.index(said[2]),
                                                         spot_distance_km=float(said[1])))
            if spot is not None:  # None: the spot is outside the mapped area, so the square carries on
                super().observe(Hypotheses(event.t_s, ("where the witness said",), ("",), (1.0,), spot[None], 0.0))
