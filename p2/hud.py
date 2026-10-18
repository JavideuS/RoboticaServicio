# On screen display over the frontal camera, like the one of a real drone app:
# battery, what the drone is doing, speed, height, heading and people found.

import math
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)


def text(image, s, x, y, color=WHITE):
    cv2.putText(image, s, (x, y), FONT, 0.38, (0, 0, 0), 3, cv2.LINE_AA)     # outline, readable on sky and sea
    cv2.putText(image, s, (x, y), FONT, 0.38, color, 1, cv2.LINE_AA)


def draw(image, drone, battery, state, found, spotted, marks=()):
    image = image.copy()
    h, w = image.shape[:2]

    # Battery: bar, green / orange / red against the level needed to get home
    need = battery.threshold(drone.distance_to(0.0, 0.0))
    color = (0, 200, 0) if battery.level > need + 0.15 else (0, 165, 255) if battery.level > need else (0, 0, 255)
    cv2.rectangle(image, (w - 62, 6), (w - 12, 16), WHITE, 1)
    cv2.rectangle(image, (w - 61, 7), (w - 61 + int(48 * battery.level), 15), color, -1)
    cv2.rectangle(image, (w - 12, 9), (w - 10, 13), WHITE, -1)     # the battery tip
    text(image, "%d%%" % round(100 * battery.level), w - 100, 15)
    text(image, state.upper(), 8, 15)

    # Flight data
    heading = (90 - math.degrees(drone.yaw)) % 360    # compass: 0 = north, clockwise
    text(image, "%.1f m/s" % drone.speed, 8, h - 10)
    text(image, "H %.1f m" % drone.z, 85, h - 10)
    text(image, "HDG %03d" % round(heading), 160, h - 10)
    text(image, "FOUND %d" % found + ("   SPOTTED %d" % spotted if spotted else ""), 8, h - 26)

    for u, v in marks:  # candidates from the spotter
        cv2.circle(image, (int(u), int(v)), 6, (0, 255, 255), 1)
    return image
