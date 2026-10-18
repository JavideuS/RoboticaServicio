# 1. localization (coordinates.py): GPS points of the task -> meters from the boat
# 2. control (controller.py, sensors.py): nose forward drone, state from ROS
# 3. search (planner.py): expanding square around the given point, at 5 m
# 4. faces (detector.py): rotated Haar on the ventral camera, averaged into people
# 5. battery (battery.py): drains in sim time, back to the boat to charge when
#    what is left is just enough to get there, then on from where it stopped
# 6. extras: frontal camera spots what stands out on the way and where, to
#    centre and size the search (spotter.py); battery and flight data over the
#    frontal image (hud.py); live top-down map, also saved for a GIF (trace.py)
#
# Mission, as states:
#   transit  boat -> given point, looking ahead with the spotter; 25 m before
#            it, the search is re-planned around what was spotted
#   search   the expanding square, recording faces
#   home     back to the boat, for charging or because the search is done
#   charge   landed on the boat until the battery is full
#   resume   back to the point where the search was left, then search again
#   done     landed, report every person in local, UTM and GPS coordinates

import utm
import WebGUI
import HAL
import Frequency
import hud
from coordinates import SURVIVORS, BOAT, gps_to_local, local_to_gps
from controller import Drone
from planner import SEARCH, expanding_square, search_area
from detector import Watcher
from battery import Battery
from spotter import Spotter, SPOTTING
from trace import Tracer

RATE = 30   # Hz asked to Frequency.tick, which runs about twice that: it sleeps a whole period every other tick
HOME = (0.0, 0.0)   # the boat


def report(people, spotted):
    e0, n0, zone, letter = utm.from_latlon(BOAT.lat, BOAT.lon)
    print("found %d people (%d candidates spotted ahead on the way):" % (len(people), spotted))
    for i, p in enumerate(people, 1):
        gps = local_to_gps(p.x, p.y)
        print("  %d: x=%6.2f y=%6.2f m | UTM %d%s %.1f E %.1f N | %.6f, %.6f | %d hits"
              % (i, p.x, p.y, zone, letter, e0 + p.x, n0 + p.y, gps.lat, gps.lon, p.hits))


SURVIVORS_XY = gps_to_local(SURVIVORS)  # (40.34, -30.28) m
print("survivors at x=%.2f y=%.2f m from the boat" % SURVIVORS_XY)

centre, radius = SURVIVORS_XY, SEARCH.radius     # until the spotter says otherwise
search = expanding_square(*centre, radius=radius)
planned = False     # re-planned with the spotter's candidates yet

drone = Drone()
watcher = Watcher(drone)
battery = Battery()
spotter = Spotter()
tracer = Tracer(SURVIVORS_XY)
drone.takeoff(SEARCH.height)

state = "transit"
leg = 0     # search waypoint we are flying to
resume = None   # where the search was left to go charging
while True:
    Frequency.tick(RATE)
    drone.update()
    battery.update(drone.sensors.pose_stamp, drone.flying, state == "charge")

    # Battery
    if state in ("transit", "search", "resume") and battery.must_return(drone.distance_to(*HOME)):
        resume = (drone.x, drone.y)
        state = "home"
        print("battery %d%%: back to the boat, search left at (%.1f, %.1f), waypoint %d/%d"
              % (100 * battery.level, *resume, leg, len(search)))

    # Mission
    in_area = max(abs(drone.x - centre[0]), abs(drone.y - centre[1])) < radius + SEARCH.spacing
    watcher.active = state in ("search", "resume") and in_area
    frontal = drone.sensors.image("frontal")

    if state == "transit":
        if not planned and drone.distance_to(*SURVIVORS_XY) > SPOTTING.min_range:
            if frontal is not None:
                spotter.look(frontal[0], drone.sensors.transform)
        elif not planned:
            centre, radius = search_area(spotter.seen, SURVIVORS_XY)
            search = expanding_square(*centre, radius=radius)
            planned = True
            print("%d candidates spotted: search centred at (%.1f, %.1f), radius %.1f m, %d waypoints, lanes %.1f m apart"
                  % (len(spotter.seen), *centre, radius, len(search), SEARCH.spacing))
        if drone.go_to(*search[0], SEARCH.height) and planned:
            leg = 1
            state = "search"
    elif state == "search":
        if drone.go_to(*search[leg], SEARCH.height, v_max=SEARCH.speed):
            leg += 1
            print("waypoint %d/%d, people so far: %d" % (leg, len(search), len(watcher.people.confirmed)))
            if leg == len(search):
                resume = None
                state = "home"
    elif state == "home":
        if drone.go_to(*HOME, SEARCH.height):
            drone.land()
            if resume is not None:
                state = "charge"
            else:
                state = "done"
                report(watcher.people.confirmed, len(spotter.seen))
    elif state == "charge":
        if battery.full:
            print("charged, back to (%.1f, %.1f)" % resume)
            drone.takeoff(SEARCH.height)
            state = "resume"
    elif state == "resume":
        if drone.go_to(*resume, SEARCH.height):
            state = "search"

    tracer.step(drone.sensors.pose_stamp, drone, state, search, leg, watcher.people, spotter.seen, battery)

    # Gui: map with the ventral camera on the right, frontal with the HUD on the left
    WebGUI.showImage(tracer.view(watcher.view))
    if frontal is not None:
        marks = spotter.marks if state == "transit" else ()
        WebGUI.showLeftImage(hud.draw(frontal[0], drone, battery, state, len(watcher.people.confirmed), len(spotter.seen), marks))
