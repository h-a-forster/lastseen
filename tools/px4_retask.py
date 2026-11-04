# /// script
# requires-python = ">=3.11"
# dependencies = ["pymavlink==2.4.50"]
# ///
"""Fly a recorded LastSeen re-plan on a PX4 autopilot: the search pattern, then a new leg sent mid-flight.

From a race replay, the sortie in which the drone was retasked becomes two missions: (a) take off and fly the pattern
up to the point where the drone broke off, (b) the hunt leg it flew after the witness report, then return to launch.
The script uploads (a), starts it, and uploads (b) while the drone is still flying (a), with no landing in between:
how LastSeen sends a new leg mid-flight.

Recorded positions are 30 s apart at a constant speed, so a straight line between two of them cuts any corner the
drone turned in between; corners are recovered from the distance flown. Metres convert to degrees as in
app/sar/terrain.py (`Place.latlon`).

    python px4_retask.py race.json --dry-run --plans out/      # print both missions, write QGroundControl .plan files
    python px4_retask.py race.json --connect udpin:0.0.0.0:14540
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

M_PER_DEG = 111_195.0  # metres per degree of latitude on a 6,371 km sphere, as in the simulator
FRAME_TOL_M = 1.0  # a frame this close to speed x time flew a straight line
CORNER_TOL_M = 10.0  # a recovered corner must explain the distance flown to within this
SIM_STEP_S = 5.0  # the simulator's step: a route that ends mid-step loses the rest of it, so a path can come up short
COLLINEAR_M = 2.0  # a vertex this close to the line through its neighbours is dropped

MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_RETURN_TO_LAUNCH, MAV_CMD_NAV_TAKEOFF = 16, 20, 22
MAV_CMD_DO_CHANGE_SPEED, MAV_CMD_DO_SET_MODE, MAV_CMD_COMPONENT_ARM_DISARM = 178, 176, 400
MAV_FRAME_MISSION, MAV_FRAME_GLOBAL_RELATIVE_ALT, MAV_FRAME_GLOBAL_RELATIVE_ALT_INT = 2, 3, 6
MAV_AUTOPILOT_PX4, MAV_TYPE_QUADROTOR = 12, 2
PX4_MODE_AUTO, PX4_AUTO_MISSION = 4, 4
MAV_LANDED_STATE_ON_GROUND = 1

Point = tuple[float, float]  # (y_m south of the map's north edge, x_m east of its west edge)


@dataclass(frozen=True)
class Frame:
    t_s: float
    pos: Point
    returning: bool


@dataclass(frozen=True)
class Leg:
    name: str
    points: list[Point]


# --- from the replay to two legs ---

def frames(events: list[dict], drone: int) -> list[Frame]:
    return [Frame(e["t_s"], (e["drones"][drone][0], e["drones"][drone][1]), e["drones"][drone][2])
            for e in events if e["type"] == "drones"]


def retask_time(events: list[dict], drone: int) -> float:
    times = [e["t_s"] for e in events if e["type"] == "retask" and e["drone"] == drone]
    if not times:
        raise ValueError(f"drone {drone} was never retasked in this replay")
    return times[0]


def sortie_around(fs: list[Frame], t_s: float) -> list[Frame]:
    """The frames of the sortie in flight at `t_s`: from the last frame on the ground before it to the second frame
    heading home after it, so the turn where the search ended can be recovered."""
    start = max(i for i, f in enumerate(fs) if f.t_s <= t_s and f.returning)
    end = next(i for i, f in enumerate(fs) if f.t_s > t_s and f.returning)
    return fs[start:end + 2]


def path(fs: list[Frame], speed_mps: float) -> list[tuple[float, Point]]:
    """Time-stamped vertices of the flown path: the frames plus the corners recovered between them."""
    out = [(fs[0].t_s, fs[0].pos)]
    for i in range(len(fs) - 1):
        a, b = fs[i], fs[i + 1]
        flown = speed_mps * (b.t_s - a.t_s)
        corners = _corners(fs, i, flown) if _dist(a.pos, b.pos) < flown - FRAME_TOL_M else []
        at = 0.0
        for p in [*corners, b.pos]:
            at += _dist(out[-1][1], p)
            out.append((a.t_s + min(at, flown) / speed_mps, p))
        out[-1] = (b.t_s, b.pos)
    return out


def split(vertices: list[tuple[float, Point]], t_s: float) -> tuple[Leg, Leg]:
    """Break the path at the last vertex reached before `t_s`: where the drone left its pattern."""
    k = max(i for i, (t, _) in enumerate(vertices) if t <= t_s)
    before = [p for _, p in vertices[:k + 1]]
    after = [p for _, p in vertices[k:]]
    return Leg("a-pattern", _simplify(before)), Leg("b-hunt", _simplify(after)[1:])


def _corners(fs: list[Frame], i: int, flown: float) -> list[Point]:
    """The turns between frames i and i+1, from the straight lines flown just before and just after."""
    d_in = _heading(fs, i - 1, flown)
    d_out = _heading(fs, i + 1, flown)
    if d_in is None or d_out is None:
        return []
    p, q = fs[i].pos, fs[i + 1].pos
    cross = d_in[0] * d_out[1] - d_in[1] * d_out[0]
    if abs(cross) > 0.1:
        s = ((q[0] - p[0]) * d_out[1] - (q[1] - p[1]) * d_out[0]) / cross
        corner = (p[0] + s * d_in[0], p[1] + s * d_in[1])
        return [corner] if s >= 0 and _explains(_dist(p, corner) + _dist(corner, q), flown, fs, i) else []
    if d_in[0] * d_out[0] + d_in[1] * d_out[1] < -0.99:
        return _u_turn(p, q, d_in, flown)
    return []


def _explains(length: float, flown: float, fs: list[Frame], i: int) -> bool:
    lost = SIM_STEP_S * flown / (fs[i + 1].t_s - fs[i].t_s)
    return -lost - CORNER_TOL_M < length - flown < CORNER_TOL_M


def _u_turn(p: Point, q: Point, d: Point, flown: float) -> list[Point]:
    """Out along d, across, and back: the turn at the end of a lawnmower lane."""
    rel = (q[0] - p[0], q[1] - p[1])
    along = rel[0] * d[0] + rel[1] * d[1]
    across = (rel[0] - along * d[0], rel[1] - along * d[1])
    out = (flown - math.hypot(*across) + along) / 2
    if out < 0 or out - along < 0:
        return []
    c1 = (p[0] + out * d[0], p[1] + out * d[1])
    return [c1, (c1[0] + across[0], c1[1] + across[1])]


def _heading(fs: list[Frame], i: int, flown: float) -> Point | None:
    """The unit direction of frame gap i if the drone flew it in a straight line, else None."""
    if i < 0 or i + 1 >= len(fs):
        return None
    p, q = fs[i].pos, fs[i + 1].pos
    d = _dist(p, q)
    if d < flown - FRAME_TOL_M:
        return None
    return ((q[0] - p[0]) / d, (q[1] - p[1]) / d)


def _simplify(points: list[Point]) -> list[Point]:
    out = [points[0]]
    for i in range(1, len(points) - 1):
        if _dist(points[i], out[-1]) > COLLINEAR_M and _off_line(points[i], out[-1], points[i + 1]) > COLLINEAR_M:
            out.append(points[i])
    out.append(points[-1])
    return out


def _off_line(p: Point, a: Point, b: Point) -> float:
    ab = _dist(a, b)
    if ab == 0:
        return _dist(p, a)
    return abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1])) / ab


def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


# --- map metres to the autopilot's degrees ---

def to_latlon(bounds: dict[str, float], p: Point) -> tuple[float, float]:
    """As `Place.latlon`: metres south of the north edge and east of the west edge, the map centre's scale."""
    lat0 = (bounds["north"] + bounds["south"]) / 2
    return bounds["north"] - p[0] / M_PER_DEG, bounds["west"] + p[1] / (M_PER_DEG * math.cos(math.radians(lat0)))


# --- missions: (command, frame, params) with params as in a QGroundControl .plan ---

def mission(leg: Leg, bounds: dict, home: Point, alt_m: float, speed_mps: float, take_off: bool) -> list[tuple]:
    items = []
    if take_off:
        items.append((MAV_CMD_NAV_TAKEOFF, MAV_FRAME_GLOBAL_RELATIVE_ALT, [0, 0, 0, None, *to_latlon(bounds, home), alt_m]))
    items.append((MAV_CMD_DO_CHANGE_SPEED, MAV_FRAME_MISSION, [1, speed_mps, -1, 0, 0, 0, 0]))
    items += [(MAV_CMD_NAV_WAYPOINT, MAV_FRAME_GLOBAL_RELATIVE_ALT, [0, 0, 0, None, *to_latlon(bounds, p), alt_m])
              for p in leg.points[1 if take_off else 0:]]
    if not take_off:
        items.append((MAV_CMD_NAV_RETURN_TO_LAUNCH, MAV_FRAME_MISSION, [0, 0, 0, 0, 0, 0, 0]))
    return items


def qgc_plan(items: list[tuple], home_latlon: tuple[float, float], speed_mps: float) -> dict:
    """The same layout as app/sar/export.py writes."""
    simple = [{"type": "SimpleItem", "command": c, "frame": f, "autoContinue": True, "doJumpId": n,
               "params": [round(v, 7) if isinstance(v, float) else v for v in params]}
              for n, (c, f, params) in enumerate(items, start=1)]
    return {
        "fileType": "Plan", "version": 1, "groundStation": "LastSeen",
        "mission": {"version": 2, "firmwareType": MAV_AUTOPILOT_PX4, "vehicleType": MAV_TYPE_QUADROTOR,
                    "globalPlanAltitudeMode": 1, "cruiseSpeed": speed_mps, "hoverSpeed": speed_mps,
                    "plannedHomePosition": [*(round(v, 7) for v in home_latlon), 0], "items": simple},
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "rallyPoints": {"points": [], "version": 2},
    }


# --- flying it over MAVLink ---

def upload(link, items: list[tuple]) -> None:
    """The MAVLink mission protocol: count, then each item on request, then the autopilot's ack."""
    from pymavlink import mavutil
    link.mav.mission_count_send(link.target_system, link.target_component, len(items),
                                mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
    while True:
        msg = link.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"], blocking=True, timeout=10)
        if msg is None:
            raise TimeoutError("the autopilot stopped asking for mission items")
        if msg.get_type() == "MISSION_ACK":
            if msg.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
                raise RuntimeError(f"mission rejected: MAV_MISSION_RESULT {msg.type}")
            return
        command, frame, params = items[msg.seq]
        p = [math.nan if v is None else v for v in params]
        if frame == MAV_FRAME_GLOBAL_RELATIVE_ALT:
            frame, x, y = MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, round(p[4] * 1e7), round(p[5] * 1e7)
        else:
            x, y = int(p[4]), int(p[5])
        link.mav.mission_item_int_send(link.target_system, link.target_component, msg.seq, frame, command,
                                       0, 1, *p[:4], x, y, p[6], mavutil.mavlink.MAV_MISSION_TYPE_MISSION)


def command(link, cmd: int, *params: float) -> None:
    link.mav.command_long_send(link.target_system, link.target_component, cmd, 0, *params, *[0] * (7 - len(params)))
    ack = link.recv_match(type="COMMAND_ACK", condition=f"COMMAND_ACK.command=={cmd}", blocking=True, timeout=10)
    if ack is None or ack.result != 0:
        raise RuntimeError(f"command {cmd} refused: {ack and ack.result}")


def fly(connect: str, a: list[tuple], b: list[tuple], switch_m: float, pause_s: float, timeout_s: float) -> None:
    from pymavlink import mavutil
    link = mavutil.mavlink_connection(connect, source_system=245)
    link.wait_heartbeat(timeout=30)
    print(f"autopilot {link.target_system}: uploading (a), {len(a)} items")
    upload(link, a)
    if pause_s:
        print(f"(a) uploaded; starting in {pause_s:.0f} s", flush=True)
        time.sleep(pause_s)
    link.mav.mission_set_current_send(link.target_system, link.target_component, 0)
    command(link, MAV_CMD_DO_SET_MODE, 1, PX4_MODE_AUTO, PX4_AUTO_MISSION)
    _await_prearm(link)
    command(link, MAV_CMD_COMPONENT_ARM_DISARM, 1)
    print("armed, flying (a)")
    switch_at = _await_switch(link, a, switch_m, timeout_s)
    print(f"retask {switch_at:.0f} m before the break-off point: uploading (b), {len(b)} items, in flight")
    upload(link, b)
    link.mav.mission_set_current_send(link.target_system, link.target_component, 0)
    print("flying (b), then return to launch")
    _await_landed(link, timeout_s)
    print("landed")


def _await_prearm(link, timeout_s: float = 30.0) -> None:
    """Wait until the autopilot reports its arming checks passing, which includes checking the new mission."""
    from pymavlink import mavutil
    prearm = mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = link.recv_match(type="SYS_STATUS", blocking=True, timeout=5)
        if msg is not None and msg.onboard_control_sensors_health & prearm:
            return
    raise TimeoutError("the autopilot's arming checks did not pass")


def _await_switch(link, a: list[tuple], switch_m: float, timeout_s: float) -> float:
    """Wait until the drone is heading for the last waypoint of (a) and within `switch_m` of it."""
    last = len(a) - 1
    lat, lon = a[last][2][4], a[last][2][5]
    current, deadline = 0, time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = link.recv_match(type=["MISSION_CURRENT", "GLOBAL_POSITION_INT"], blocking=True, timeout=5)
        if msg is None:
            continue
        if msg.get_type() == "MISSION_CURRENT":
            current = msg.seq
        elif current == last:
            d = _ground_m(msg.lat / 1e7, msg.lon / 1e7, lat, lon)
            if d < switch_m:
                return d
    raise TimeoutError("the drone never reached the break-off point")


def _await_landed(link, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hb = link.recv_match(type="HEARTBEAT", blocking=True, timeout=5)
        if hb is not None and hb.get_srcSystem() == link.target_system and not hb.base_mode & 128:
            return
    raise TimeoutError("the drone did not land and disarm in time")


def _ground_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return math.hypot((lat2 - lat1) * M_PER_DEG, (lon2 - lon1) * M_PER_DEG * math.cos(math.radians(lat1)))


# --- command line ---

def legs(race: dict, lane: str, drone: int) -> tuple[Leg, Leg, Point]:
    events = race["lanes"][lane]["events"]
    t_retask = retask_time(events, drone)
    sortie = sortie_around(frames(events, drone), t_retask)
    homeward = sortie[-2].t_s
    vertices = [v for v in path(sortie, race["fleet"]["speed_mps"]) if v[0] < homeward]
    a, b = split(vertices, t_retask)
    return a, b, sortie[0].pos


def show(name: str, items: list[tuple]) -> None:
    names = {MAV_CMD_NAV_TAKEOFF: "takeoff", MAV_CMD_NAV_WAYPOINT: "waypoint", MAV_CMD_DO_CHANGE_SPEED: "speed",
             MAV_CMD_NAV_RETURN_TO_LAUNCH: "return to launch"}
    print(f"mission {name}: {len(items)} items")
    for n, (c, _, p) in enumerate(items):
        where = f"{p[4]:.6f}, {p[5]:.6f} at {p[6]:.0f} m" if c in (MAV_CMD_NAV_TAKEOFF, MAV_CMD_NAV_WAYPOINT) else ""
        extra = f"{p[1]:.0f} m/s" if c == MAV_CMD_DO_CHANGE_SPEED else ""
        print(f"  {n:2d} {names[c]:16s} {where}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("race", type=Path, help="a race replay JSON")
    ap.add_argument("--lane", default="hedge+spark")
    ap.add_argument("--drone", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print the missions without flying")
    ap.add_argument("--plans", type=Path, help="also write both missions as QGroundControl .plan files here")
    ap.add_argument("--connect", default="udpin:0.0.0.0:14540", help="pymavlink connection string to the autopilot")
    ap.add_argument("--switch-m", type=float, default=30.0, help="send (b) this close to the break-off point")
    ap.add_argument("--pause-s", type=float, default=0.0, help="wait after uploading (a), e.g. for a ground station")
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    args = ap.parse_args()

    race = json.loads(args.race.read_text(encoding="utf-8"))
    bounds, alt_m, speed = race["place"]["bounds"], race["fleet"]["alt_m"], race["fleet"]["speed_mps"]
    a, b, home = legs(race, args.lane, args.drone)
    mission_a = mission(a, bounds, home, alt_m, speed, take_off=True)
    mission_b = mission(b, bounds, home, alt_m, speed, take_off=False)
    show("(a) pattern", mission_a)
    show("(b) hunt", mission_b)
    if args.plans:
        args.plans.mkdir(parents=True, exist_ok=True)
        for leg, items in ((a, mission_a), (b, mission_b)):
            plan = qgc_plan(items, to_latlon(bounds, home), speed)
            (args.plans / f"retask-{leg.name}.plan").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    if not args.dry_run:
        fly(args.connect, mission_a, mission_b, args.switch_m, args.pause_s, args.timeout_s)


if __name__ == "__main__":
    main()
