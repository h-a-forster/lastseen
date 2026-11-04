# /// script
# requires-python = ">=3.11"
# dependencies = ["pyulog==1.3.0"]
# ///
"""From a PX4 ULog: when the mission on the vehicle changed in flight, and the ground speed around that moment.
A seamless re-plan keeps flying through the switch; a stop or loiter shows as speed near zero."""

import math
import sys
from pathlib import Path

from pyulog import ULog

path = Path(sys.argv[1])
if path.is_dir():  # a log directory: take the newest log in it
    path = max(path.rglob("*.ulg"), key=lambda p: p.stat().st_mtime)
print(path)
log = ULog(str(path), ["mission_result", "vehicle_local_position"])
t0 = log.start_timestamp
data = {d.name: d.data for d in log.data_list}

result = data["mission_result"]
changes = [float(result["timestamp"][i] - t0) / 1e6 for i in range(1, len(result["timestamp"]))
           if result["mission_id"][i] != result["mission_id"][i - 1]]
print("mission changes at (s):", [round(t, 1) for t in changes],
      "item counts:", sorted(set(int(c) for c in result["seq_total"])))

pos = data["vehicle_local_position"]
for t in changes:
    window = [math.hypot(pos["vx"][i], pos["vy"][i]) for i, ts in enumerate(pos["timestamp"])
              if abs((ts - t0) / 1e6 - t) <= 15]
    if window:
        print(f"around {t:.1f} s: ground speed min {min(window):.1f} m/s, max {max(window):.1f} m/s over +-15 s")
