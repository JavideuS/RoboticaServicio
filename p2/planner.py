# Search pattern: expanding square around the given point.
#
# Over the sea there is nothing to avoid, so this is coverage, not path
# planning: fixed height (set by the detector) and a 2D pattern. The given
# point is approximate, so the people are most likely near it: the square
# starts there and grows outwards, legs d, d, 2d, 2d, 3d, 3d... with d the
# lane spacing. Straight legs and 90 deg corners are what the controller does
# best (turn in place, then fly).
#
# If the frontal camera spotted candidates on the way (spotter.py), the square
# is centred on them instead, and only grows as far as the farthest one plus a
# margin: same sweep, smaller area, placed where the people are. With fewer
# than two candidates it keeps the given point and the full radius.

import math
from dataclasses import dataclass
from detector import footprint


# Settings
@dataclass(frozen=True)
class Search:
    height: float = 5.0     # m, highest at which every face is still detected (see detector.py)
    overlap: float = 0.2    # of the footprint, shared between neighbour lanes: faces are not points, and the drone wobbles
    radius: float = 20.0    # m around the given point that we cover, when nothing was spotted
    margin: float = 5.0     # m beyond the farthest candidate: their position error, and people not spotted next to them
    min_radius: float = 8.0     # m, never smaller than this
    speed: float = 4.0  # m/s, every face stays ~1 s in view: ~7 frames at the detector's rate

    @property
    def spacing(self):
        return footprint(self.height) * (1 - self.overlap)


SEARCH = Search()


def search_area(candidates, hint):
    if len(candidates) < 2:
        return hint, SEARCH.radius
    cx = sum(c.x for c in candidates) / len(candidates)
    cy = sum(c.y for c in candidates) / len(candidates)
    reach = max(math.hypot(c.x - cx, c.y - cy) for c in candidates) + SEARCH.margin
    return (cx, cy), min(SEARCH.radius, max(SEARCH.min_radius, reach))


def expanding_square(cx, cy, spacing=SEARCH.spacing, radius=SEARCH.radius):
    route = [(cx, cy)]
    x, y = cx, cy
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]     # east, north, west, south
    legs = math.ceil(2 * radius / spacing)
    turn = 0
    for n in range(1, legs + 1):
        for _ in range(2 if n < legs else 3):   # the last length three times to close the square
            dx, dy = directions[turn % 4]
            x, y = x + dx * n * spacing, y + dy * n * spacing
            route.append((x, y))
            turn += 1
    return route


if __name__ == "__main__":
    route = expanding_square(0, 0)
    length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
    print("spacing %.2f m, %d waypoints, %.0f m" % (SEARCH.spacing, len(route), length))
