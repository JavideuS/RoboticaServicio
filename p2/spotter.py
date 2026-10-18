# Frontal camera on the way to the search area: find what stands out from the
# sea and where it is, to place the search where the people are. Not a person
# detector: it does not know what a person is, only that something is not water.
#
# Each pixel is compared with a heavy median blur of the image, which is the
# sea around it without small things on it. What differs enough, below the
# sky, is a candidate, and connected pixels are one candidate.
#
# What helps is where they are: each blob is put on the water like the ventral
# detections (ray from the camera cut with the sea), and blobs from many frames
# are averaged into candidates. The planner then centres the square on them and
# covers only as far as they reach (planner.search_area).
#
# Transit flight, 869 frames: the six bodies came out 0.1-0.9 m from where they
# are (700+ hits each), plus one ghost between two close people (558 hits) and a
# tail of weak clusters from body parts (5-102 hits). Keeping candidates seen in
# 20% of the frames leaves the six and the ghost: the square ended up centred
# 1.2 m from the people (the given point is 7 m off) and 16 m instead of 20 m
# wide, about 20% less route. The ghost does no harm, it is between two people.

from dataclasses import dataclass
import cv2
import numpy as np
from detector import Detector, People


# Settings
@dataclass(frozen=True)
class Spotting:
    background_px: int = 21     # median blur size, much bigger than a far person (a few px)
    contrast: int = 18  # grey levels away from the sea around it
    min_area: int = 3   # px, below this it is noise
    sky_grey: int = 205     # the sky in this world is flat light grey, brighter than any water
    min_range: float = 25.0     # m to the search centre, closer than this a body is several blobs
    merge_radius: float = 3.0   # m, at 40 m one pixel row is ~1 m of depth, so blobs of one person spread
    min_share: float = 0.2  # of the frames looked at, a candidate has to be in this many to count


SPOTTING = Spotting()


class Spotter:
    def __init__(self):
        self.candidates = People(SPOTTING.merge_radius)
        self.marks = []     # (u, v) of the blobs in the last frame
        self.frames = 0

    def look(self, image, transform):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, cv2.medianBlur(gray, SPOTTING.background_px))
        sky = cv2.dilate((gray > SPOTTING.sky_grey).astype(np.uint8), np.ones((7, 7), np.uint8))   # and the edge of the sea
        mask = ((diff > SPOTTING.contrast) & (sky == 0)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))   # a far body is a thin horizontal line
        n, _, stats, centres = cv2.connectedComponentsWithStats(mask)
        self.marks = [tuple(centres[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= SPOTTING.min_area]
        self.frames += 1
        size = (image.shape[1], image.shape[0])
        for u, v in self.marks:
            spot = Detector.to_world(u, v, size, transform, "frontal")
            if spot is not None:
                self.candidates.add(*spot)

    @property
    def seen(self):
        with self.candidates.lock:
            return [c for c in self.candidates.all if c.hits >= SPOTTING.min_share * self.frames]
