# Top-down map of the mission: planned route, flown path, what the camera
# covers, every detection and the people they were averaged into, and the
# candidates the frontal camera spotted.
#
# Unibotics has two image panels, so the map goes live on the right one with
# the ventral camera as an inset (view), redrawn 5 times per second of sim
# time. What only grows (flown path, detections) is drawn once on its own
# layer, so a redraw costs the same at the end of the mission as at the start.
# One frame per second also goes to TRACE.folder; turn them into a GIF with
#
#   docker cp unibotics:/tmp/rescue_trace .
#   python3 trace.py rescue_trace mission.gif

import math
import os
import sys
from dataclasses import dataclass
import cv2
import numpy as np
from detector import footprint


# Settings
@dataclass(frozen=True)
class Trace:
    folder: str = "/tmp/rescue_trace"
    live_period: float = 0.2    # s of sim time between redraws of the live map
    save_period: float = 1.0    # s of sim time between saved frames, 0 to save none
    inset: float = 0.5  # size of the ventral image over the map
    px_per_m: float = 8.0
    x_range: tuple = (-6.0, 68.0)   # m, the boat and the whole search area
    y_range: tuple = (-62.0, 8.0)


TRACE = Trace()

# Colors (bgr)
SEA = (120, 80, 30)
GRID = (135, 95, 45)
ROUTE = (170, 150, 130)
PATH_SEARCH = (255, 200, 60)
PATH_OTHER = (200, 200, 200)
RAW = (60, 60, 255)
PERSON = (90, 230, 90)
CANDIDATE = (0, 220, 255)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Tracer:
    def __init__(self, hint):
        self.hint = hint
        self.last_point = None  # (x, y), end of the flown path so far
        self.raw_drawn = 0  # detections already on the trail layer
        self.last_live = self.last_save = self.start = None     # s, sim time
        self.image = None   # newest map
        self.count = 0
        if TRACE.save_period:
            os.makedirs(TRACE.folder, exist_ok=True)
            for name in os.listdir(TRACE.folder):   # frames of an earlier run
                if name.startswith("frame_"):
                    os.remove(os.path.join(TRACE.folder, name))
        w = int((TRACE.x_range[1] - TRACE.x_range[0]) * TRACE.px_per_m)
        h = int((TRACE.y_range[1] - TRACE.y_range[0]) * TRACE.px_per_m)
        self.base = np.full((h, w, 3), SEA, np.uint8)
        for x in range(-10, 80, 10):    # 10 m grid
            cv2.line(self.base, self.px(x, TRACE.y_range[0]), self.px(x, TRACE.y_range[1]), GRID, 1)
        for y in range(-70, 20, 10):
            cv2.line(self.base, self.px(TRACE.x_range[0], y), self.px(TRACE.x_range[1], y), GRID, 1)
        self.trail = np.zeros_like(self.base)   # flown path and detections, drawn once
        self.trail_mask = np.zeros(self.base.shape[:2], np.uint8)

    def px(self, x, y):     # world m -> image px, north up
        return (int((x - TRACE.x_range[0]) * TRACE.px_per_m), int((TRACE.y_range[1] - y) * TRACE.px_per_m))

    def on_trail(self, draw):   # draw on the trail layer and remember which pixels
        draw(self.trail)
        draw(self.trail_mask)

    # Call every tick
    def step(self, now, drone, state, route, leg, people, candidates, battery):
        if not drone.flying:
            self.last_point = None
        elif self.last_point is None or math.hypot(drone.x - self.last_point[0], drone.y - self.last_point[1]) > 0.2:
            if self.last_point is not None:
                a, b = self.px(*self.last_point), self.px(drone.x, drone.y)
                color = PATH_SEARCH if state == "search" else PATH_OTHER
                self.on_trail(lambda img: cv2.line(img, a, b, color if img.ndim == 3 else 255, 2, cv2.LINE_AA))
            self.last_point = (drone.x, drone.y)
        with people.lock:
            new = people.raw[self.raw_drawn:]
        for x, y in new:
            p = self.px(x, y)
            self.on_trail(lambda img: cv2.circle(img, p, 1, RAW if img.ndim == 3 else 255, -1))
        self.raw_drawn += len(new)

        if self.start is None:
            self.start = now
        if self.last_live is None or now - self.last_live >= TRACE.live_period:
            self.last_live = now
            self.image = self.draw(now, drone, state, route, leg, people, candidates, battery)
        if TRACE.save_period and (self.last_save is None or now - self.last_save >= TRACE.save_period):
            self.last_save = now
            cv2.imwrite(os.path.join(TRACE.folder, "frame_%05d.png" % self.count), self.image)
            self.count += 1

    # The newest map with the ventral camera over it, for the GUI
    def view(self, ventral=None):
        image = self.image.copy()
        if ventral is not None:
            small = cv2.resize(ventral, None, fx=TRACE.inset, fy=TRACE.inset)
            h, w = small.shape[:2]
            x0, y0 = image.shape[1] - w - 10, 70    # right, under the text
            image[y0:y0 + h, x0:x0 + w] = small
            cv2.rectangle(image, (x0 - 1, y0 - 1), (x0 + w, y0 + h), WHITE, 1)
        return image

    def draw(self, now, drone, state, route, leg, people, candidates, battery):
        img = self.base.copy()
        s = TRACE.px_per_m

        # Boat and the given point
        cv2.rectangle(img, self.px(-1.5, 1.0), self.px(1.5, -1.0), (60, 90, 140), -1)
        cv2.putText(img, "boat", self.px(-1.5, -2.5), FONT, 0.4, WHITE, 1, cv2.LINE_AA)
        hx, hy = self.px(*self.hint)
        cv2.drawMarker(img, (hx, hy), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, "GPS", (hx + 8, hy - 6), FONT, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

        # Planned route: done part darker
        pts = [self.px(*p) for p in route]
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            cv2.line(img, a, b, ROUTE if i + 1 >= leg else (110, 100, 90), 1, cv2.LINE_AA)

        # Flown path and raw detections
        img[self.trail_mask > 0] = self.trail[self.trail_mask > 0]

        # Frontal candidates, people
        for c in candidates:
            cv2.drawMarker(img, self.px(c.x, c.y), CANDIDATE, cv2.MARKER_TILTED_CROSS, 10, 1)
        with people.lock:
            everyone = [(p.x, p.y, p.hits) for p in people.all]
        n = 0
        for x, y, hits in everyone:
            if hits >= people.confirm_hits:
                n += 1
                cv2.circle(img, self.px(x, y), int(1.0 * s), PERSON, 2, cv2.LINE_AA)
                cv2.putText(img, str(n), self.px(x + 1.3, y + 1.3), FONT, 0.45, PERSON, 1, cv2.LINE_AA)

        # Drone and what its ventral camera sees
        if drone.flying:
            c, sn = math.cos(drone.yaw), math.sin(drone.yaw)
            half_w, half_l = footprint(drone.z) / 2, footprint(drone.z) * 0.75 / 2   # 320 x 240 image
            corners = [(drone.x + c * f - sn * l, drone.y + sn * f + c * l)
                       for f, l in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w))]
            cv2.polylines(img, [np.array([self.px(*p) for p in corners])], True, (230, 230, 230), 1, cv2.LINE_AA)
        nose = (drone.x + 1.6 * math.cos(drone.yaw), drone.y + 1.6 * math.sin(drone.yaw))
        left = (drone.x + 0.9 * math.cos(drone.yaw + 2.5), drone.y + 0.9 * math.sin(drone.yaw + 2.5))
        right = (drone.x + 0.9 * math.cos(drone.yaw - 2.5), drone.y + 0.9 * math.sin(drone.yaw - 2.5))
        cv2.fillPoly(img, [np.array([self.px(*nose), self.px(*left), self.px(*right)])], WHITE)

        # Text
        lines = ["t %3d s   %s" % (now - self.start, state.upper()),
                 "battery %d%%" % round(100 * battery.level),
                 "people %d   candidates %d" % (n, len(candidates))]
        for i, line in enumerate(lines):   # top right, the boat is top left
            cv2.putText(img, line, (img.shape[1] - 210, 20 + 18 * i), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        legend = [("searching", PATH_SEARCH), ("transit", PATH_OTHER), ("detection", RAW),
                  ("person", PERSON), ("frontal candidate", CANDIDATE)]
        for i, (name, color) in enumerate(legend):
            y = img.shape[0] - 12 - 16 * (len(legend) - 1 - i)
            cv2.circle(img, (16, y - 4), 4, color, -1)
            cv2.putText(img, name, (26, y), FONT, 0.4, WHITE, 1, cv2.LINE_AA)
        cv2.line(img, (img.shape[1] - 20 - int(10 * s), img.shape[0] - 16), (img.shape[1] - 20, img.shape[0] - 16), WHITE, 2)
        cv2.putText(img, "10 m", (img.shape[1] - 20 - int(10 * s), img.shape[0] - 22), FONT, 0.4, WHITE, 1, cv2.LINE_AA)
        return img


# Frames -> GIF, run on your own machine
if __name__ == "__main__":
    from PIL import Image
    folder, out = sys.argv[1], sys.argv[2]
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 2     # keep one frame of every N
    names = sorted(n for n in os.listdir(folder) if n.startswith("frame_"))[::every]
    frames = [Image.open(os.path.join(folder, n)).convert("P", palette=Image.ADAPTIVE) for n in names]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=80, loop=0)
    print("%s: %d frames" % (out, len(frames)))
