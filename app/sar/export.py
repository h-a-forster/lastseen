"""A sortie as a QGroundControl mission (`.plan`), so a planned block can be flown by any MAVLink drone.

Format: https://docs.qgroundcontrol.com/master/en/qgc-dev-guide/file_formats/plan.html. Waypoints use the
relative-altitude frame (height above the take-off point); the item-level altitude keys are left out, so QGroundControl
reads the frame and altitude from the MAVLink fields themselves. Take off at the base, fly the lanes, return to launch.
"""

from app.sar.interface import Waypoint
from app.sar.terrain import Place

MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_RETURN_TO_LAUNCH, MAV_CMD_NAV_TAKEOFF = 16, 20, 22
MAV_FRAME_MISSION, MAV_FRAME_GLOBAL_RELATIVE_ALT = 2, 3
MAV_AUTOPILOT_PX4, MAV_TYPE_QUADROTOR = 12, 2
ALTITUDE_RELATIVE = 1  # QGroundControl's AltitudeFrameRelative


def qgc_plan(place: Place, base_m: tuple[float, float], route: list[Waypoint], speed_mps: float) -> dict:
    if not route:
        raise ValueError("an empty sortie has no mission to export")
    home = tuple(round(v, 7) for v in place.latlon(*base_m))
    alt = route[0][2]
    items = [_item(MAV_CMD_NAV_TAKEOFF, MAV_FRAME_GLOBAL_RELATIVE_ALT, [0, 0, 0, None, *home, alt])]
    items += [_item(MAV_CMD_NAV_WAYPOINT, MAV_FRAME_GLOBAL_RELATIVE_ALT, [0, 0, 0, None, *place.latlon(y, x), a])
              for y, x, a in route]
    items.append(_item(MAV_CMD_NAV_RETURN_TO_LAUNCH, MAV_FRAME_MISSION, [0, 0, 0, 0, 0, 0, 0]))
    for jump_id, item in enumerate(items, start=1):
        item["doJumpId"] = jump_id
    return {
        "fileType": "Plan", "version": 1, "groundStation": "LastSeen",
        "mission": {"version": 2, "firmwareType": MAV_AUTOPILOT_PX4, "vehicleType": MAV_TYPE_QUADROTOR,
                    "globalPlanAltitudeMode": ALTITUDE_RELATIVE, "cruiseSpeed": speed_mps, "hoverSpeed": speed_mps,
                    "plannedHomePosition": [*home, 0], "items": items},
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "rallyPoints": {"points": [], "version": 2},
    }


def _item(command: int, frame: int, params: list) -> dict:
    params = [round(p, 7) if isinstance(p, float) else p for p in params]
    return {"type": "SimpleItem", "command": command, "frame": frame, "params": params, "autoContinue": True}
