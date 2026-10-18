# Lower layer of the drone: keeps its state and turns "go there" / "look
# there" into HAL commands. Nothing here knows about the mission.
#
# Nose forward: the drone first turns to the target and then flies to it, so
# the frontal camera always looks where we are going.
#
# Yaw is commanded as a rate, not as an angle. With the rate we close the
# loop on the angle ourselves, so callers still ask for angles and we keep
# control of the speed. Height is left to set_cmd_mix, which takes it absolute.
#
# Commands go through HAL, the state comes from sensors.py (HAL's is old).

import math
from dataclasses import dataclass
import HAL
from sensors import Sensors

FLYING = 3  # HAL.get_landed_state(), the exercise page says 2 but the wrapper sends 3


# Settings
@dataclass(frozen=True)
class Flight:
    v_max: float = 6.0  # m/s, top speed, used outside the search where nothing has to be seen
    w_max: float = 1.0  # rad/s
    brake: float = 0.8  # m/s2, what we count on to slow down before the target
    k_v: float = 0.8    # the last meters: m/s per m left
    k_yaw: float = 1.0  # yaw rate per rad of heading error
    turn_start: float = math.radians(30)    # heading error above this -> no forward speed, only turn
    yaw_free: float = 1.5   # m, closer than this the bearing jumps around, so stop turning
    reach_tol: float = 0.3  # m, horizontal
    reach_v: float = 0.3    # m/s, slower than this counts as stopped
    z_tol: float = 0.3  # m
    yaw_tol: float = math.radians(3)


FLIGHT = Flight()


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clip(value, limit):
    return max(-limit, min(limit, value))


# Drone
class Drone:
    def __init__(self):
        self.sensors = Sensors()
        self.x = self.y = self.z = 0.0  # m, from the takeoff point: east, north, up
        self.yaw = 0.0  # rad, 0 = east, counterclockwise
        self.speed = 0.0    # m/s, horizontal
        self.state = None
        self.height = 0.0   # m, the height we are asking for
        self.update()

    # Call once per tick, before anything else
    def update(self):
        if self.sensors.pose is not None:
            self.x, self.y, self.z, self.yaw = self.sensors.pose
        vx, vy, _ = self.sensors.velocity
        self.speed = math.hypot(vx, vy)
        self.state = HAL.get_landed_state()

    @property
    def flying(self):
        return self.state == FLYING

    def distance_to(self, x, y):
        return math.hypot(x - self.x, y - self.y)

    def bearing_to(self, x, y):
        return math.atan2(y - self.y, x - self.x)

    # These two block until done
    def takeoff(self, height):
        self.height = height
        HAL.takeoff(height)
        self.update()

    def land(self):
        HAL.land()
        self.update()

    # Body velocities (forward, left), yaw rate, and the height we keep
    def move(self, forward, left, yaw_rate):
        if not self.flying:     # commands on the ground only confuse the platform
            return
        HAL.set_cmd_mix(float(forward), float(left), float(self.height), float(yaw_rate))    # the platform rejects ints

    def hold(self):
        self.move(0.0, 0.0, 0.0)

    # The next ones are steps: call them every tick until they return True
    def turn_to(self, yaw):
        error = wrap(yaw - self.yaw)
        if abs(error) < FLIGHT.yaw_tol:
            self.hold()
            return True
        self.move(0.0, 0.0, clip(FLIGHT.k_yaw * error, FLIGHT.w_max))
        return False

    def look_at(self, x, y):
        return self.turn_to(self.bearing_to(x, y))

    def go_to(self, x, y, z=None, v_max=FLIGHT.v_max):
        if z is not None:
            self.height = z
        dist = self.distance_to(x, y)
        there = dist < FLIGHT.reach_tol and abs(self.height - self.z) < FLIGHT.z_tol
        if there and self.speed < FLIGHT.reach_v:
            self.hold()
            return True

        error = wrap(self.bearing_to(x, y) - self.yaw)
        # The drone takes its time to stop: sqrt(2*a*d) is the speed it can still
        # brake from in the distance left
        speed = min(v_max, math.sqrt(2 * FLIGHT.brake * dist), FLIGHT.k_v * dist)
        if dist < FLIGHT.yaw_free:
            yaw_rate = 0.0
        else:
            yaw_rate = clip(FLIGHT.k_yaw * error, FLIGHT.w_max)
            speed *= max(0.0, 1.0 - abs(error) / FLIGHT.turn_start)     # full speed only when aligned
        # The velocity points at the target, not at the nose: once aligned it is
        # the same thing, and the small sideways part takes out any drift
        self.move(speed * math.cos(error), speed * math.sin(error), yaw_rate)
        return False
