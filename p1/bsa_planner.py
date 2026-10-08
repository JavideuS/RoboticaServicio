# You can load a yaml or just paste the same values into a dict and call plan().

# Steps:
#   1. Inflate the map by the robot radius -> any free pixel is a safe center
#   2. Split it into robot-sized cells, keep cells whose center is safe
#      (grid offset is picked to fit as many cells as possible)
#   3. BSA: spiral keeping walls / cleaned cells on one side, when stuck
#      BFS to the nearest uncleaned cell and spiral again. Stop when nothing
#      reachable is left. One-cell dead ends are swept on the way past.
#   4. Metrics converison: cells -> pixels -> odom waypoints
#
# Run offline:  python bsa_planner.py map_grannyannie.yaml --start -1 1.5

from collections import deque
import numpy as np
import cv2

N, E, S, W = range(4)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # (drow, dcol): up, right, down, left on the image
# Note BSA direction priority is per map heading not robot's own heading


def load_yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:     # tiny fallback, enough for our flat yaml
        import ast
        cfg = {}
        for line in open(path):
            line = line.split("#")[0].strip()
            if ":" in line:
                k, v = line.split(":", 1)
                v = v.strip()
                try:
                    cfg[k.strip()] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    cfg[k.strip()] = v
        return cfg

# Conversion/Coordinates class: odom <-> pixel
class MapFrame:
    def __init__(self, cfg):
        self.res = cfg["resolution"]
        self.c0, self.r0 = cfg["origin_px"]
        self.ax, self.ay = cfg["axes"]

    def to_px(self, x, y):  # -> (row, col) floats
        return self.r0 + self.ay * y / self.res, self.c0 + self.ax * x / self.res

    def to_world(self, row, col):
        return (col - self.c0) * self.res / self.ax, (row - self.r0) * self.res / self.ay

    def m2px(self, m):
        return m / self.res

    def dir_px(self, yaw):  # heading -> (drow, dcol) unit vector on the image
        return self.ay * np.sin(yaw), self.ax * np.cos(yaw)


# map -> grid
def build_grid(img, cfg, frame):
    gray = img if img.ndim == 2 else img[..., :3].mean(-1)
    free = (gray > cfg["free_thresh"]).astype(np.uint8)

    radius = int(np.ceil(frame.m2px(cfg["robot_diameter"] / 2 + cfg["safety_margin"])))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    safe = cv2.erode(free, k)

    cell = int(round(frame.m2px(cfg["cell_size"])))
    half = cell // 2

    # try every grid offset, keep the one with the most usable cells
    best = max(((safe[oy + half::cell, ox + half::cell].sum(), oy, ox)
                for oy in range(cell) for ox in range(cell)))
    _, oy, ox = best
    grid = safe[oy + half::cell, ox + half::cell].astype(bool)

    def center(i, j):
        return oy + half + i * cell, ox + half + j * cell

    # can we drive straight between neighbor cells?
    h, w = grid.shape
    ok = np.zeros((h, w, 4), bool)
    for i in range(h):
        for j in range(w):
            if not grid[i, j]:
                continue
            r1, c1 = center(i, j)
            for d, (di, dj) in enumerate(DIRS):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and grid[ni, nj]:
                    r2, c2 = center(ni, nj)
                    t = np.linspace(0, 1, 8)
                    ok[i, j, d] = safe[np.round(r1 + t * (r2 - r1)).astype(int),
                                       np.round(c1 + t * (c2 - c1)).astype(int)].all()
    return free, safe, grid, ok, center, cell


# BSA
def bsa(grid, ok, start, hand=-1, first=0):
    # hand: which side the spiral keeps walls / cleaned cells on (-1 left, +1 right)
    # first: heading tried first for the very first spiral
    seen = np.zeros_like(grid)
    path = [start]
    seen[start] = True

    def step(c, d):
        return c[0] + DIRS[d][0], c[1] + DIRS[d][1]

    def open_(c, d):    # free and not cleaned yet
        return ok[c][d] and not seen[step(c, d)]

    def n_open(c):
        return sum(open_(c, d) for d in range(4))

    def visit(c):
        seen[c] = True
        path.append(c)

    def pick_heading(c, first=0):
        # spiral wants something blocked on its side, so it hugs walls / cleaned area
        order = [(first + k) % 4 for k in range(4)]
        for d in order:
            if open_(c, d) and not open_(c, (d + hand) % 4):
                return d
        for d in order:
            if open_(c, d):
                return d
        return None

    def sweep_pockets(c):
        # It avoids keeping free dead-ends/corners
        # a neighbor that can only be entered from here would be left behind
        # as a one-cell hole and cost a whole backtrack later: do it now
        # (one step in, one step back)
        for d in range(4):
            if n_open(c) > 1 and open_(c, d) and n_open(step(c, d)) == 0:
                visit(step(c, d))
                path.append(c)

    def spiral(c, first=0):
        d = pick_heading(c, first)
        if d is None:
            return c
        if all(open_(c, x) for x in range(4)):  # open space: go find a wall first
            while open_(c, d):
                c = step(c, d); visit(c)
            d = pick_heading(c)
            if d is None:
                return c
        while True:
            sweep_pockets(c)
            for nd in ((d + hand) % 4, d, (d - hand) % 4, (d + 2) % 4):     # side, front, other side, back
                if open_(c, nd):
                    c, d = step(c, nd), nd
                    visit(c)
                    break
            else:
                return c    # stuck: spiral done

    def nearest_unseen(c):
        # BFS through free cells; first uncleaned cell found is the closest
        prev = {c: None}
        q = deque([c])
        while q:
            cur = q.popleft()
            if not seen[cur]:
                route = []
                while cur != c:
                    route.append(cur); cur = prev[cur]
                return route[::-1]
            for d in range(4):
                if ok[cur][d]:
                    n = step(cur, d)
                    if n not in prev:
                        prev[n] = cur; q.append(n)
        return None

    c, spirals = spiral(start, first), 1
    while True:
        route = nearest_unseen(c)
        if route is None:
            break
        path += route
        seen[route[-1]] = True
        c = spiral(route[-1])
        spirals += 1
    return path, seen, spirals


def n_turns(path):
    return sum((b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1])
               for a, b, c in zip(path, path[1:], path[2:]))


# Execution
def plan(img, cfg, start_xy):
    frame = MapFrame(cfg)
    free, safe, grid, ok, center, cell = build_grid(img, cfg, frame)

    # start = closest usable cell to the robot
    r, c = frame.to_px(*start_xy)
    cells = np.argwhere(grid)
    centers = np.array([center(i, j) for i, j in cells])
    start = tuple(cells[np.argmin(((centers - [r, c]) ** 2).sum(1))])

    # the spiral can hug either side and leave the start in any direction;
    # planning is cheap, so we try them all and keep the cheapest to drive
    # (a turn costs about as much time as two cells)
    path, seen, spirals = min((bsa(grid, ok, start, hand, first) for hand in (-1, 1) for first in range(4)),
                              key=lambda r: len(r[0]) + 2 * n_turns(r[0]))
    px = [center(i, j) for i, j in path]
    world = [frame.to_world(r, c) for r, c in px]

    # rough coverage: sweep a robot-wide brush along the path
    brush = np.zeros_like(free)
    pts = np.array([(c, r) for r, c in px], np.int32)
    cv2.polylines(brush, [pts], False, 1, int(frame.m2px(cfg["robot_diameter"])))
    coverage = (brush & free).sum() / free.sum()

    return dict(path_cells=path, path_px=px, waypoints=world, grid=grid, free=free,
                safe=safe, cell=cell, spirals=spirals, coverage=coverage,
                unreached=int(grid.sum() - seen.sum()))


def simplify(points):
    # keep only the corners, straight runs become one segment.
    if len(points) < 3:
        return list(points)
    out = [points[0]]
    for a, b, c in zip(points, points[1:], points[2:]):
        ux, uy, vx, vy = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]
        straight = abs(ux * vy - uy * vx) < 1e-6 and ux * vx + uy * vy > 0
        if not straight:
            out.append(b)
    out.append(points[-1])
    return out

# Visualization
def draw_arrow(img, r, c, dr, dc, color, size=7):
    # filled arrowhead centered on (r, c) pointing along (drow, dcol)
    n = np.hypot(dr, dc)
    if n < 1e-9:
        return
    u = np.array([dc, dr]) / n  # (x, y) on the image
    p = np.array([-u[1], u[0]])
    o = np.array([c, r])
    tri = [o + size * u, o - 0.6 * size * u + 0.7 * size * p, o - 0.6 * size * u - 0.7 * size * p]
    cv2.fillConvexPoly(img, np.round(tri).astype(np.int32), color)


def gui_image(result, every=60):
    # uint8 for WebGUI.showNumpy: 0-127 gray, 131 green path, 128 red start.
    # Arrowheads along the path (one per `every` px, at least one on any
    # segment long enough to fit it) show which way it will be driven.
    img = (result["free"] * 127).astype(np.uint8)
    pts = np.array([(c, r) for r, c in result["path_px"]], np.int32)
    cv2.polylines(img, [pts], False, 131, 2)
    corners = simplify(result["path_px"])
    for (r1, c1), (r2, c2) in zip(corners, corners[1:]):
        L = np.hypot(r2 - r1, c2 - c1)
        if L < 25:
            continue
        n = max(1, int(L // every))
        for k in range(n):
            t = (k + 0.5) / n
            draw_arrow(img, r1 + t * (r2 - r1), c1 + t * (c2 - c1), r2 - r1, c2 - c1, 131)
    r, c = result["path_px"][0]
    cv2.circle(img, (int(c), int(r)), 8, 128, -1)
    return img


def debug_image(result):
    # colored png for offline checking
    free = result["free"]
    img = np.dstack([free * 255] * 3).astype(np.uint8)
    img[(free == 1) & (result["safe"] == 0)] = (215, 215, 235)  # too close to walls
    for r, c in result["path_px"]:
        cv2.circle(img, (int(c), int(r)), 2, (180, 180, 180), -1)
    pts = np.array([(c, r) for r, c in result["path_px"]], np.int32)
    n = len(pts)
    for k in range(n - 1):  # blue -> red over time
        t = k / max(n - 2, 1)
        cv2.line(img, tuple(map(int, pts[k])), tuple(map(int, pts[k + 1])),
                 (int(255 * (1 - t)), 60, int(255 * t)), 2)
    cv2.circle(img, tuple(map(int, pts[0])), 9, (0, 170, 0), -1)
    return img

# Offline code
if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml")
    ap.add_argument("--start", nargs=2, type=float, default=[-1.0, 1.5], help="odom x y")
    a = ap.parse_args()

    cfg = load_yaml(a.yaml)
    img_path = os.path.join(os.path.dirname(os.path.abspath(a.yaml)), cfg["image"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    res = plan(img, cfg, a.start)
    wp = simplify(res["waypoints"])

    print(f"cell: {res['cell']} px, grid {res['grid'].shape}, usable cells {res['grid'].sum()}")
    print(f"path: {len(res['path_cells'])} cells, {res['spirals']} spirals, "
          f"{len(wp)} waypoints after simplify")
    print(f"unreached cells: {res['unreached']}")
    print(f"estimated coverage: {100 * res['coverage']:.1f}% of free pixels")

    np.savetxt("waypoints.csv", np.array(wp), fmt="%.3f", delimiter=",", header="x,y")
    cv2.imwrite("bsa_plan.png", cv2.cvtColor(debug_image(res), cv2.COLOR_RGB2BGR))
    print("saved waypoints.csv and bsa_plan.png")
