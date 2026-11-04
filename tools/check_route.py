# /// script
# requires-python = ">=3.11"
# dependencies = ["pyulog==1.3.0"]
# ///
"""From a PX4 ULog and a QGroundControl .plan: how close the vehicle came to each waypoint, and in which order.
A flown mission passes every waypoint within a few metres, in sequence."""

import json
import math
import sys
from pathlib import Path

from pyulog import ULog

M_PER_DEG = 111_195.0

log = ULog(sys.argv[1], ["vehicle_global_position"])
pos = log.get_dataset("vehicle_global_position").data
t0 = log.start_timestamp
plan = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
waypoints = [i["params"][4:6] for i in plan["mission"]["items"] if i["command"] == 16]

last_t = -1.0
for n, (lat, lon) in enumerate(waypoints, start=1):
    scale = M_PER_DEG * math.cos(math.radians(lat))
    best = min(range(len(pos["lat"])),
               key=lambda i: math.hypot((pos["lat"][i] - lat) * M_PER_DEG, (pos["lon"][i] - lon) * scale))
    d = math.hypot((pos["lat"][best] - lat) * M_PER_DEG, (pos["lon"][best] - lon) * scale)
    t = (pos["timestamp"][best] - t0) / 1e6
    order = "in order" if t > last_t else "OUT OF ORDER"
    last_t = t
    print(f"waypoint {n:2d}: closest {d:5.1f} m at {t:7.1f} s, {order}")
