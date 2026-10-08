# 1. wait for odom, plan the coverage path once with BSA
# 2. follow the path segment by segment: turn in place at corners, then
#    drive along the line between waypoints, steering back onto it if off
# 3. laser as a safety net: if the body would hit something before the
#    waypoint, skip that waypoint instead of crashing
# 4. collision check: the bumper is disabled in this exercise (HAL fakes it
#    from the laser), so "crashed" = told to drive but not getting closer to
#    the waypoint. Then back off a bit and go on with the next waypoint.
#
# Pose and laser are read straight from the ROS topics, not through HAL:
# HAL's spin thread handles 30 callbacks/s but ~60 msgs/s come in, so its
# queues stay full and getPose3d() is ~0.5 s old (and only changes every 4-5
# ticks). This helps a lot with the controller and allows higher velocities + smoother movements

import math
import threading
from dataclasses import dataclass
import cv2
import WebGUI
import HAL
import Frequency
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from bsa_planner import plan, simplify, gui_image, draw_arrow, MapFrame

MAP = "/workspace/code/mapgrannyannie.png"
RATE = 20   # Hz


# Settings
@dataclass(frozen=True)
class Robot:
    # These are given
    diameter: float = 0.35  # m
    laser_x: float = 0.139  # m, laser sits this far ahead of the robot center

    @property
    def radius(self):
        return self.diameter / 2


@dataclass(frozen=True)
class Follow:   # path following
    # After some runs these were the best constants I got
    v_max: float = 0.5  # m/s
    v_min: float = 0.06     # m/s
    w_max: float = 1.5  # rad/s
    w_min: float = 0.15     # rad/s, so the turn doesn't crawl at the end
    k_turn: float = 3.0     # turn-in-place gain
    k_w: float = 2.5    # heading gain while driving
    k_ct: float = 2.0   # how hard to pull back onto the line (rad per m off)
    k_v: float = 2.0    # slow down near the end of the segment (m/s per m left)
    turn_start: float = math.radians(30)    # error above this -> stop and turn in place
    turn_done: float = math.radians(2)  # turning ends below this
    reach_tol: float = 0.02     # m left along the segment
    off_warn: float = 0.05  # m, warn if a waypoint is "reached" this far off the line


@dataclass(frozen=True)
class Safety:   # not crashing, and getting out if we do
    front_min: float = 0.12     # m from the laser, never drive closer than this
    stuck_s: float = 2.0    # s driving without getting closer -> we hit something
    back_s: float = 1.0     # s reversing after that
    back_v: float = 0.15    # m/s


ROBOT = Robot()
FOLLOW = Follow()
SAFETY = Safety()
ARROW_M = 0.40  # m driven between two arrows of the GUI trail

# resolution / origin / axes: same values as map.yaml (made by map_registration.py from wall corners)
CFG = {"resolution": 0.009991, "origin_px": [577.75, 419.05], "axes": [-1, 1],
       "free_thresh": 128, "robot_diameter": ROBOT.diameter, "safety_margin": 0.02, "cell_size": 0.30}
# Sensors
class Sensors(Node):
    # queue depth 1 = always the newest message
    def __init__(self):
        super().__init__("vacuum_fresh_sensors")
        self.odom = None
        self.scan = None
        self.create_subscription(Odometry, "/vacuum_cleaner/odom", self.on_odom, 1)
        self.create_subscription(LaserScan, "/vacuum_cleaner/laser/scan", self.on_scan, 1)
        ex = SingleThreadedExecutor()
        ex.add_node(self)
        threading.Thread(target=ex.spin, daemon=True).start()

    def on_odom(self, msg):
        self.odom = msg

    def on_scan(self, msg):
        self.scan = msg

    def pose(self):     # -> (x, y, yaw) or None before the first message
        if self.odom is None:
            return None
        p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        return p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def front(self):
        if self.scan is None:
            return float("inf")
        vals = [v for v in self.scan.ranges[80:101] if v == v and v > 0.01]     # drop nan / zeros
        return min(vals) if vals else float("inf")


# Path following
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp(v, lim):
    return max(-lim, min(lim, v))


class Follower:
    # Following segments instead of points created better routes and cleaned more areas
    # With only points it may deviate some times
    # Follow the *segment* A->B, not just the point B:
    #   along = how far along the segment we are -> reached when near the end
    #   cross = how far left/right of the line   -> steer back onto it
    # Chasing the point makes the heading jump around near B (zigzag), and
    # "close enough" checks can fire while still off the line.
    #
    # step() is called once per tick with the newest pose and laser reading
    # and returns the (v, w) to send. It never touches the robot itself.

    def __init__(self, waypoints, start):
        self.waypoints = waypoints
        self.i = -1     # waypoint we are driving to
        self.skipped = 0
        self.back_ticks = 0     # > 0 while reversing after a hit
        self.next_segment(start)    # first segment starts at the robot, later ones at planned points

    @property
    def done(self):
        return self.i >= len(self.waypoints)

    def next_segment(self, new_start):
        self.A = new_start
        self.i += 1
        self.turning = True
        self.best = float("inf")    # closest we got to the end of this segment
        self.stuck_ticks = 0    # ticks driving since `best` last improved

    def skip(self, why, here):
        # give up on this waypoint; the next segment starts from where we are
        print(f"waypoint {self.i}: {why}, skipping")
        self.skipped += 1
        self.next_segment(here)

    def step(self, x, y, yaw, front):
        if self.back_ticks > 0:     # reversing after a hit
            self.back_ticks -= 1
            if self.back_ticks == 0:
                self.skip("not moving (hit something)", (x, y))
                return 0.0, 0.0
            return -SAFETY.back_v, 0.0  # we came from behind, so that way is free

        while not self.done:
            A, B = self.A, self.waypoints[self.i]
            sx, sy = B[0] - A[0], B[1] - A[1]
            L = math.hypot(sx, sy)
            if L < 1e-3:
                self.next_segment(B)
                continue
            ux, uy = sx / L, sy / L
            dx, dy = x - A[0], y - A[1]
            along = dx * ux + dy * uy
            cross = ux * dy - uy * dx   # > 0: robot is left of the line
            remaining = L - along
            if remaining >= FOLLOW.reach_tol:
                break
            if abs(cross) > FOLLOW.off_warn:
                print(f"waypoint {self.i} reached {abs(cross):.2f} m off the line")
            self.next_segment(B)    # next segment starts at the planned point
        else:
            return 0.0, 0.0     # no waypoints left

        # heading we want: along the segment, bent back toward the line when off it
        target = math.atan2(uy, ux) - math.atan(FOLLOW.k_ct * cross)
        err = wrap(target - yaw)

        if not self.turning and abs(err) > FOLLOW.turn_start:
            self.turning = True
        if self.turning:
            if abs(err) >= FOLLOW.turn_done:
                w = max(FOLLOW.w_min, min(FOLLOW.w_max, FOLLOW.k_turn * abs(err)))
                return 0.0, math.copysign(w, err)
            self.turning = False

        # blocked = the body would touch before the center gets to the waypoint
        # (the laser is ahead of the center, so compare in laser distances)
        if front < min(SAFETY.front_min, remaining + ROBOT.radius - ROBOT.laser_x - 0.02):
            self.skip(f"laser sees something at {front:.2f} m", (x, y))
            return 0.0, 0.0

        # collision check: driving but not getting any closer
        if remaining < self.best - 0.01:
            self.best, self.stuck_ticks = remaining, 0
        self.stuck_ticks += 1
        if self.stuck_ticks > SAFETY.stuck_s * RATE:
            self.back_ticks = int(SAFETY.back_s * RATE)
            return -SAFETY.back_v, 0.0

        # slow down for the corner at the end of the segment
        v = max(FOLLOW.v_min, min(FOLLOW.v_max, FOLLOW.k_v * remaining)) * max(0.0, math.cos(err))
        return v, clamp(FOLLOW.k_w * err, FOLLOW.w_max)


# Gui
class Gui:
    # green = plan (arrowheads = direction it will be driven),
    # gray = cleaned so far, blue arrows = direction it was actually driven,
    # red arrow = robot
    def __init__(self, res, frame, start):
        self.frame = frame
        self.base = gui_image(res)
        self.swept = self.base * 0  # 1 where the robot body has passed
        self.trail = []     # (row, col, drow, dcol) of the blue arrows
        self.last_px = None     # where the gray was painted up to
        self.last_arrow = start
        WebGUI.showNumpy(self.base)

    def show(self, x, y, yaw):
        r, c = self.frame.to_px(x, y)
        px = (int(round(c)), int(round(r)))
        cv2.line(self.swept, self.last_px or px, px, 1, int(self.frame.m2px(ROBOT.diameter)))
        self.last_px = px
        if math.hypot(x - self.last_arrow[0], y - self.last_arrow[1]) >= ARROW_M:
            self.trail.append((r, c, *self.frame.dir_px(yaw)))
            self.last_arrow = (x, y)
        view = self.base.copy()
        view[(self.swept == 1) & (self.base == 127)] = 90   # only over free floor, the plan stays visible
        for a in self.trail:
            draw_arrow(view, *a, 132)
        draw_arrow(view, r, c, *self.frame.dir_px(yaw), 128, size=13)
        WebGUI.showNumpy(view)


# Main
if not rclpy.ok():
    rclpy.init()
sensors = Sensors()
while sensors.pose() is None:
    Frequency.tick(RATE)
x, y, yaw = sensors.pose()

res = plan(WebGUI.getMap(MAP), CFG, (x, y))
waypoints = simplify(res["waypoints"])
print(f"plan: {len(waypoints)} waypoints, {res['spirals']} spirals, ~{100 * res['coverage']:.0f}% coverage")

gui = Gui(res, MapFrame(CFG), (x, y))
follower = Follower(waypoints, (x, y))
tick = 0
while not follower.done:
    Frequency.tick(RATE)
    x, y, yaw = sensors.pose()
    v, w = follower.step(x, y, yaw, sensors.front())
    HAL.setV(v)
    HAL.setW(w)
    tick += 1
    if tick % (RATE // 4) == 0:
        gui.show(x, y, yaw)

HAL.setV(0)
HAL.setW(0)
gui.show(*sensors.pose())
print(f"done, {follower.skipped} waypoints skipped")
while True:
    Frequency.tick(RATE)
