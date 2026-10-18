# Faces in the ventral camera -> where they are on the water -> list of people.
#
# Haar (Viola-Jones) only finds faces that are close to upright in the image,
# and the people float at any angle, so every frame is also searched rotated.
#
# So 5 m with a rotation every 20 deg (18 per frame). That is ~0.1-0.2 s per
# frame, too slow for the control loop, so it runs in its own thread on the
# newest frame and the loop only reads the results.
#
# Each detection is placed on the water by casting the pixel's ray from the
# camera with the full drone attitude (the camera is fixed to the body, and the
# drone tilts when it speeds up or brakes) and cutting it with the sea, z = 0.
# The same person is seen in many frames, so detections close to each other
# are averaged into one person instead of thrown away as duplicates.
#
# Only frames taken inside the search area count: Haar fires on our own boat
# (its landing pad and frame), and the boat is not someone to rescue.

import math
import threading
import time
from dataclasses import dataclass
import cv2
import numpy as np


# Settings
@dataclass(frozen=True)
class Vision:
    cascade: str = "/resources/exercises/rescue_people/haarcascade_frontalface_default.xml"    # given
    rotation_step: int = 20     # deg, below the narrowest band measured (21 deg at 5 m)
    scale_factor: float = 1.1   # window growth between pyramid levels, OpenCV default
    min_neighbors: int = 3  # overlapping windows needed to call it a face, OpenCV default
    same_face_px: float = 15    # hits closer than this in different rotations are one face
    hfov: float = 1.02974   # rad, camera model (quadrotor_gz.urdf.xacro), 320 x 240 px


@dataclass(frozen=True)
class Tracking:
    merge_radius: float = 2.0   # m, well above our position error, well below the spacing between people
    confirm_hits: int = 3   # a single stray hit on a wave never becomes a person


VISION = Vision()
TRACKING = Tracking()


def footprint(height):
    # Width of water the camera sees across the drone, at this height
    return 2 * height * math.tan(VISION.hfov / 2)


# Faces in one image
class Detector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(VISION.cascade)
        if self.cascade.empty():
            raise RuntimeError("could not load " + VISION.cascade)

    # Pixel centres (u, v) of the faces in the image
    def faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        side = int(math.hypot(w, h)) + 1    # canvas big enough that no corner is cut when rotating
        hits = []
        for angle in range(0, 360, VISION.rotation_step):
            rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rot[:, 2] += ((side - w) / 2, (side - h) / 2)
            back = cv2.invertAffineTransform(rot)
            rotated = cv2.warpAffine(gray, rot, (side, side))
            for x, y, bw, bh in self.cascade.detectMultiScale(rotated, VISION.scale_factor, VISION.min_neighbors):
                hits.append(back @ (x + bw / 2, y + bh / 2, 1.0))
        # One face is usually found in two neighbouring rotations
        faces = []
        for hit in hits:
            for face in faces:
                if np.hypot(*(face[0] / face[1] - hit)) < VISION.same_face_px:
                    face[0] += hit
                    face[1] += 1
                    break
            else:
                faces.append([hit.copy(), 1])
        return [tuple(total / n) for total, n in faces]

    # Where on the water a pixel is, or None if the ray never reaches it
    @staticmethod
    def to_world(u, v, size, transform, camera="ventral"):
        w, h = size
        position, rotation = transform
        f = (w / 2) / math.tan(VISION.hfov / 2)     # px, both cameras are the same model
        right, up = (w / 2 - u) / f, (h / 2 - v) / f    # left and up of the image centre, at 1 m
        if camera == "ventral":     # looking down, image up = forward
            ray = np.array([up, right, -1.0])
        else:   # frontal, looking forward, image up = up
            ray = np.array([1.0, right, up])
        ray = rotation @ ray
        if ray[2] > -1e-3:
            return None
        point = position - position[2] / ray[2] * ray
        return point[0], point[1]


# People found so far
class Person:
    def __init__(self, x, y):
        self.x, self.y, self.hits = x, y, 1

    def add(self, x, y):
        self.hits += 1
        self.x += (x - self.x) / self.hits  # running average
        self.y += (y - self.y) / self.hits


class People:
    def __init__(self, merge_radius=TRACKING.merge_radius, confirm_hits=TRACKING.confirm_hits):
        self.merge_radius, self.confirm_hits = merge_radius, confirm_hits
        self.all = []
        self.raw = []   # every single detection, for the tracer
        self.lock = threading.Lock()

    def add(self, x, y):
        with self.lock:
            self.raw.append((x, y))
            near = [p for p in self.all if math.hypot(p.x - x, p.y - y) < self.merge_radius]
            if near:
                min(near, key=lambda p: math.hypot(p.x - x, p.y - y)).add(x, y)
            else:
                self.all.append(Person(x, y))

    @property
    def confirmed(self):
        with self.lock:
            return [p for p in self.all if p.hits >= self.confirm_hits]


# The thread: newest ventral frame -> faces -> people
class Watcher:
    def __init__(self, drone):
        self.drone = drone
        self.detector = Detector()
        self.people = People()
        self.active = False     # record people only while this is set (the mission sets it)
        self.view = None    # last frame processed, with the faces marked, for the GUI
        self.frame_s = 0.0  # s, how long the last frame took
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        last_stamp = None
        while True:
            frame = self.drone.sensors.image("ventral")
            transform = self.drone.sensors.transform    # taken together with the frame, ~0.02 s apart
            if frame is None or transform is None or not self.drone.flying or frame[1] == last_stamp:
                time.sleep(0.02)
                continue
            image, last_stamp = frame
            start = time.time()
            size = (image.shape[1], image.shape[0])
            for u, v in self.detector.faces(image):
                spot = self.detector.to_world(u, v, size, transform)
                if spot is not None and self.active:
                    self.people.add(*spot)
                cv2.circle(image, (int(u), int(v)), 14, (0, 0, 255), 2)
            self.frame_s = time.time() - start
            self.view = image
