# Offline only: where the CORNERS control points in map_registration.py come from.
#
#   python extract_corners.py model.dae mapgrannyannie.png
#
# model.dae is the house mesh Gazebo loads for this exercise:
#   JdeRobot/RoboticsInfrastructure -> CustomRobots/roomba_robot/models/house_int2/meshes/model.dae
# The world file puts that model at (0, 0, 0) with no rotation, so its
# vertices are already in the same coordinates as the robot's odometry.
#
# Steps:
#   1. wall corners in meters = vertices of the mesh (floor left out)
#   2. corners in pixels = 2x2 blocks of the map with 1 or 3 free pixels
#   3. rough alignment: cut the mesh at laser height to get the wall outline,
#      then try scales / origins / axis flips and keep the one where that
#      outline lies closest to the black/white edges of the map
#      (corners alone are too few for this, many wrong fits look as good)
#   4. pair every world corner with the map corner under it, keep a few
#      that are spread over the whole map

import argparse
import xml.etree.ElementTree as ET
import cv2
import numpy as np

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
FREE_THRESH = 128
MATCH_PX = 3.0  # a world corner and a map corner this close are the same corner
N_POINTS = 12   # control points to keep
SKIP_NODES = ("Plane",)     # the floor
CUT_Z = 0.075   # m, height of the laser
FAR_PX = 8.0    # outline points farther than this from any map edge all count the same
OVERHANG_PX = 40    # the outer walls may stick out of the image by this much


def load_mesh(dae_path):
    # -> list of (N, 3, 3) triangle arrays in world meters, one per part of the house
    root = ET.parse(dae_path).getroot()
    geoms = {}
    for g in root.findall(".//c:geometry", NS):
        mesh = g.find("c:mesh", NS)
        sources = {s.get("id"): s.find("c:float_array", NS) for s in mesh.findall("c:source", NS)}
        pos = [i.get("source")[1:] for i in mesh.find("c:vertices", NS).findall("c:input", NS)
               if i.get("semantic") == "POSITION"][0]
        P = np.array(sources[pos].text.split(), float).reshape(-1, 3)
        tris = []
        for prim in mesh.findall("c:triangles", NS) + mesh.findall("c:polylist", NS):
            inputs = prim.findall("c:input", NS)
            stride = max(int(i.get("offset")) for i in inputs) + 1
            voff = [int(i.get("offset")) for i in inputs if i.get("semantic") == "VERTEX"][0]
            idx = np.array(prim.find("c:p", NS).text.split(), int).reshape(-1, stride)[:, voff]
            vcount = prim.find("c:vcount", NS)
            sizes = [3] * (len(idx) // 3) if vcount is None else map(int, vcount.text.split())
            k = 0
            for n in sizes:     # polygons become triangle fans
                tris += [P[[idx[k], idx[k + j], idx[k + j + 1]]] for j in range(1, n - 1)]
                k += n
        geoms[g.get("id")] = np.array(tris)
    parts = []
    for node in root.findall(".//c:visual_scene/c:node", NS):
        if node.get("name") in SKIP_NODES:
            continue
        M = np.array(node.find("c:matrix", NS).text.split(), float).reshape(4, 4)   # scale + offset of this part
        for inst in node.findall("c:instance_geometry", NS):
            parts.append(geoms[inst.get("url")[1:]] @ M[:3, :3].T + M[:3, 3])
    return parts


def world_corners(parts):
    # every vertex, height dropped
    return np.unique(np.vstack([t.reshape(-1, 3)[:, :2] for t in parts]).round(3), axis=0)


def wall_outline(parts, step=0.02):
    # points every `step` m along the cut of the mesh with the plane z = CUT_Z
    pts = []
    for tris in parts:
        for tri in tris:
            d = tri[:, 2] - CUT_Z
            cut = [tri[i, :2] + d[i] / (d[i] - d[j]) * (tri[j, :2] - tri[i, :2])
                   for i, j in ((0, 1), (1, 2), (2, 0)) if (d[i] < 0) != (d[j] < 0)]
            if len(cut) == 2:
                n = max(2, int(np.linalg.norm(cut[1] - cut[0]) / step))
                pts.append(cut[0] + np.linspace(0, 1, n)[:, None] * (cut[1] - cut[0]))
    return np.vstack(pts)


def load_map(png_path):
    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    return (img[..., :3].mean(-1) > FREE_THRESH).astype(np.uint8)


def map_corners(free):
    f = free.astype(int)
    blocks = f[:-1, :-1] + f[1:, :-1] + f[:-1, 1:] + f[1:, 1:]
    rows, cols = np.where((blocks == 1) | (blocks == 3))
    return np.column_stack([cols + 0.5, rows + 0.5])    # .5 = on the edge between pixels


def rough_alignment(outline, free):
    # distance from every pixel to the nearest black/white edge of the map
    edges = cv2.morphologyEx(free, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    dist = np.minimum(cv2.distanceTransform(1 - edges, cv2.DIST_L2, 5), FAR_PX)
    dist = np.pad(dist, OVERHANG_PX, constant_values=FAR_PX)   # outside the image = far
    h, w = free.shape

    def search(pts, axes, scales, c0s, r0s):
        best = (np.inf, None)
        for ax, ay in axes:
            for scale in scales:    # px per meter
                x = np.round(ax * pts[:, 0] * scale).astype(int)
                y = np.round(ay * pts[:, 1] * scale).astype(int)
                for c0 in c0s:
                    if c0 + x.min() < -OVERHANG_PX or c0 + x.max() >= w + OVERHANG_PX:
                        continue
                    for r0 in r0s:
                        if r0 + y.min() < -OVERHANG_PX or r0 + y.max() >= h + OVERHANG_PX:
                            continue
                        cost = dist[r0 + y + OVERHANG_PX, c0 + x + OVERHANG_PX].mean()
                        if cost < best[0]:
                            best = (cost, (ax, ay, scale, c0, r0))
        return best

    # coarse over everything, then fine around the winner
    _, (ax, ay, scale, c0, r0) = search(outline[::6], [(-1, -1), (-1, 1), (1, -1), (1, 1)],
                                        np.arange(90, 110.01, 1.0), range(0, w, 3), range(0, h, 3))
    return search(outline, [(ax, ay)], np.arange(scale - 1, scale + 1.01, 0.1),
                  range(c0 - 4, c0 + 5), range(r0 - 4, r0 + 5))


def pair_up(world, corners_px, align):
    ax, ay, scale, c0, r0 = align
    proj = np.column_stack([c0 + ax * world[:, 0] * scale, r0 + ay * world[:, 1] * scale])
    pairs = {}
    for (x, y), p in zip(world, proj):
        d = np.hypot(*(corners_px - p).T)
        j = int(d.argmin())
        if d[j] < MATCH_PX and (j not in pairs or d[j] < pairs[j][4]):
            pairs[j] = (x, y, corners_px[j, 0], corners_px[j, 1], d[j])
    return np.array(sorted(pairs.values(), key=lambda p: (p[2], p[3])))


def spread_out(pairs, n):
    # start in one corner of the map, then always add the point farthest from the ones already picked
    picked = [int(np.argmin(pairs[:, 2] + pairs[:, 3]))]
    while len(picked) < min(n, len(pairs)):
        d = np.min([np.hypot(pairs[:, 2] - pairs[k, 2], pairs[:, 3] - pairs[k, 3]) for k in picked], axis=0)
        picked.append(int(d.argmax()))
    return pairs[picked]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dae", help="house mesh (model.dae)")
    ap.add_argument("png", help="map image")
    a = ap.parse_args()

    parts = load_mesh(a.dae)
    world = world_corners(parts)
    free = load_map(a.png)
    corners_px = map_corners(free)
    print(f"{len(world)} corners in the mesh, {len(corners_px)} corners in the map")

    cost, align = rough_alignment(wall_outline(parts), free)
    ax, ay, scale, c0, r0 = align
    print(f"rough alignment: axes [{ax}, {ay}], {scale:.1f} px/m, odom (0,0) near col {c0}, row {r0}")

    pairs = pair_up(world, corners_px, align)
    print(f"{len(pairs)} corners paired within {MATCH_PX} px, keeping {min(N_POINTS, len(pairs))}:\n")
    print("# x, y (odom, m)     col, row (png, px)")
    for x, y, col, row, _ in spread_out(pairs, N_POINTS):
        print(f"    [{x:6.3f}, {y:6.3f},  {col:5.1f}, {row:5.1f}],")
