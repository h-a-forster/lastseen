# Flying LastSeen's missions on PX4 (software in the loop)

LastSeen's missions were flown by PX4, the open-source autopilot many real drones run, in its simulator: PX4
v1.17.0 SITL with its built-in quadrotor model (`sihsim_quadx`), watched from QGroundControl 5.1.5. Home is the
search base in Saint-Ursanne (`PX4_HOME_LAT=47.36484 PX4_HOME_LON=7.15537 PX4_HOME_ALT=440`), with `MPC_XY_CRUISE=10`
to match LastSeen's 10 m/s.

**A first sortie as exported.** The brief section's QGroundControl download for drone 1 (12 items, 100 m above take-off)
was opened, uploaded and started from QGroundControl. PX4 took off, flew every lane, returned and landed; the flight log
passes all 10 waypoints within 0.5 m, in order.

**The mid-flight re-plan.** In the featured race (Saint-Ursanne, case 19), the team planner's drone breaks off its
expanding square after the dog walker's call. `tools/px4_retask.py` rebuilds that sortie from the replay as two missions:
(a) the square up to where the drone broke off, and (b) the hunt it flew after the call, then return to launch. It
uploads and starts (a), and uploads (b) while the drone is still flying (a), 30 m before the break-off point, the way
LastSeen sends a new leg over the radio link. PX4 switched to (b) without stopping (ground speed 6.6-10 m/s in the 15 s
either side of the switch), passed every waypoint of both missions within 0.6 m, in order, except the break-off point
itself, which (b) replaced before the drone got there, and landed.

To rerun, with PX4 v1.17.0 built (`make px4_sitl sihsim_quadx`, the variables above) and `pymavlink==2.4.50`,
`pyulog==1.3.0`:

    python tools/px4_retask.py race.json --dry-run --plans out/     # print both missions, write them as .plan files
    python tools/px4_retask.py race.json --connect udpin:0.0.0.0:14540
    python tools/check_switch.py log.ulg                             # when the mission changed, and the speed around it
    python tools/check_route.py log.ulg out/retask-b-hunt.plan       # how close the flight came to each waypoint

`race.json` is a recorded race (`/api/runs/<id>` on the page). QGroundControl 5.1.5 shows a mission another program
uploads only after Plan > Download from the vehicle, and that clears its flown track, so the recording of the
re-plan keeps the original square on screen and lets the flown track show the drone leaving it.

This is the autopilot's own mission handling on a simulated airframe: no radio link, wind or real battery, and no
photos coming back. It shows that LastSeen's missions and its mid-flight change are accepted and flown by PX4 as sent.
