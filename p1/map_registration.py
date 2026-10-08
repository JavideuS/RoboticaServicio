# Run this once to get the odom -> map pixel transform and save it as a yaml
# for bsa_planner.py (the same numbers go in CFG in the main vacuum script).
# No need to run it again unless the map or the points change.
#
#   python map_registration.py                      # just print the results
#   python map_registration.py --yaml map.yaml      # also save a yaml for bsa_planner.py
#   python map_registration.py --clicked            # first attempt, to compare
#
# How it works:
#   1. Control points = places known both in odom (x, y) and as a pixel
#      (col, row) on the PNG.
#        CORNERS: wall corners. World coordinates from the Gazebo house model,
#                 pixel = the corner of the white area on the map. (default)
#        CLICKED: first attempt. Drove the robot to a few spots, wrote down
#                 odom and the pixel where it seemed to be. Guessing the pixel
#                 by eye is the weak part: 17 px rms, vs ~1.5 px with corners.
#   2. Map isn't rotated vs the world, just flipped, so each axis is fit on
#      its own:  col = c0 + a*x,  row = r0 + e*y
#      (also tried a free-angle fit and swapped axes to double check.)
#   3. Some clicks were copied to keep them lined up (same x -> same col).
#      Copying shares one click's error between points, so those get +-3 px
#      of random noise and the fit is repeated many times. The average is the
#      final M, and the spread shows how much those clicks matter.
#      (only done for CLICKED: corners that share a column are on the same wall)

import argparse
import numpy as np

# x, y (odom, m)     col, row (png, px)
# .5 = the corner sits on the edge between two pixels
CORNERS = np.array([
    [ 2.895,  0.700,  288.5, 489.5],
    [-2.702,  1.728,  846.5, 592.5],
    [ 5.313,  5.730,   47.5, 992.5],
    [-0.300, -1.060,  606.5, 312.5],
    [ 2.895,  4.301,  287.5, 850.5],
    [ 1.099,  1.926,  468.5, 612.5],
    [-0.900,  0.740,  669.5, 494.5],
    [ 3.095,  2.885,  266.5, 707.5],
    [ 1.100, -1.050,  468.5, 312.5],
    [ 2.895,  1.885,  288.5, 606.5],
    [ 1.899,  1.925,  386.5, 612.5],
    [-0.900, -1.060,  669.5, 313.5],
])

CLICKED = np.array([
    [-1.000,  1.500,  663, 556],
    [-3.598,  1.500,  956, 584],
    [ 5.011,  1.500,   58, 535],
    [ 5.011, -3.198,   58,  85],
    [ 5.011,  5.301,   58, 980],
    [-2.468,  5.301,  824, 940],
    [ 0.479,  3.622,  528, 782],
    [ 1.790,  2.236,  402, 636],
    [ 2.360,  0.786,  338, 502],
    [ 0.315, -0.852,  536, 335],
])

EXCLUDE = []    # drop bad points here after looking at the errors
NOISE_PX = 3
TRIALS = 1000

ap = argparse.ArgumentParser()
ap.add_argument("--yaml", help="save the result as a yaml for bsa_planner.py")
ap.add_argument("--image", default="mapgrannyannie.png", help="map png name written in the yaml")
ap.add_argument("--clicked", action="store_true", help="use the first-attempt clicked points")
args = ap.parse_args()

DATA = CLICKED if args.clicked else CORNERS
keep = [i for i in range(len(DATA)) if i not in EXCLUDE]
world, pixel = DATA[keep, :2], DATA[keep, 2:]
rng = np.random.default_rng(0)


def fit(world, pixel, swap=False):
    # straight line per axis, swap=True tries x->row / y->col
    wc, wr = (1, 0) if swap else (0, 1)
    ones = np.ones(len(world))
    (c0, a), *_ = np.linalg.lstsq(np.column_stack([ones, world[:, wc]]), pixel[:, 0], rcond=None)
    (r0, e), *_ = np.linalg.lstsq(np.column_stack([ones, world[:, wr]]), pixel[:, 1], rcond=None)
    M = np.zeros((2, 3))
    M[0, wc], M[0, 2] = a, c0
    M[1, wr], M[1, 2] = e, r0
    return M


def fit_free_angle(world, pixel):
    # rotation + scale + shift, only used to check the angle
    W, P = world - world.mean(0), pixel - pixel.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ W)
    R = U @ Vt
    return np.degrees(np.arctan2(R[1, 0], R[0, 0])), np.linalg.det(R) < 0


def errors(M, world, pixel):
    pred = (M @ np.column_stack([world, np.ones(len(world))]).T).T
    return np.linalg.norm(pred - pixel, axis=1)


def rms(e):
    return np.sqrt((e ** 2).mean())


# Quick checks
rms_normal = rms(errors(fit(world, pixel), world, pixel))
rms_swap = rms(errors(fit(world, pixel, swap=True), world, pixel))
angle, flipped = fit_free_angle(world, pixel)
swap = rms_swap < rms_normal
print(f"x->col fit: {rms_normal:.1f} px   x->row fit: {rms_swap:.1f} px   -> swap={swap}")
print(f"free angle: {angle:+.1f} deg (flipped={flipped}), ~0 or ~180 means no real rotation")

# Noise on copied clicks
# a coord counts as copied if 2+ points share both the world value and the pixel value
copied = np.zeros_like(pixel, dtype=bool)
for w in (0, 1):
    for p in (0, 1):
        for i in range(len(world)):
            same = (np.abs(world[:, w] - world[i, w]) < 0.01) & (pixel[:, p] == pixel[i, p])
            if same.sum() > 1:
                copied[same, p] = True
if not args.clicked:
    copied[:] = False   # corners on the same wall really do share a column / row
print(f"copied clicks (point, axis): {[(keep[i], 'col' if j == 0 else 'row') for i, j in zip(*np.where(copied))]}")

Ms = np.array([
    fit(world, pixel + copied * rng.integers(-NOISE_PX, NOISE_PX + 1, pixel.shape), swap=swap)
    for _ in range(TRIALS)
])
M = Ms.mean(0)
spread = Ms.std(0)

# Results
px_per_m = np.sqrt(abs(np.linalg.det(M[:, :2])))
err = errors(M, world, pixel)

np.set_printoptions(suppress=True, precision=3)
print("\nM =\n", M)
print(f"spread from noise (px): a {spread[0, 0]:.2f}, c0 {spread[0, 2]:.2f}, e {spread[1, 1]:.2f}, r0 {spread[1, 2]:.2f}")
print(f"resolution: {1 / px_per_m:.4f} m/px ({px_per_m:.1f} px/m)")
print(f"odom (0,0) is at col {M[0, 2]:.0f}, row {M[1, 2]:.0f}")
print(f"robot 0.35 m = {0.35 * px_per_m:.1f} px (exercise says 35)")
print(f"error per point (px): {np.round(err, 1)}")
print(f"rms: {rms(err):.1f} px = {rms(err) / px_per_m:.2f} m")


# Copy these into the main script
M_inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))

def world_to_pixel(x, y):
    col, row = M @ [x, y, 1]
    return int(round(row)), int(round(col))     # [row, col] for numpy

def pixel_to_world(row, col):
    x, y, _ = M_inv @ [col, row, 1]
    return x, y


# Save yaml for bsa_planner
def save_yaml(path, M, image):
    # bsa_planner uses one resolution for both axes, so the two fitted
    # scales get averaged (they differ by ~0.2% here)
    if M[0, 0] == 0:
        raise SystemExit("axes look swapped (x->row), bsa_planner doesn't handle that")
    with open(path, "w") as f:
        f.write(f"""# made by map_registration.py ({len(world)} {'clicked points' if args.clicked else 'wall corners'}, rms {rms(err):.1f} px)
# pixel <-> odom:  col = origin_col + axes[0] * x / resolution
#                  row = origin_row + axes[1] * y / resolution
image: {image}
resolution: {1 / px_per_m:.6f}        # m/px
origin_px: [{M[0, 2]:.2f}, {M[1, 2]:.2f}]   # where odom (0,0) lands
axes: [{int(np.sign(M[0, 0]))}, {int(np.sign(M[1, 1]))}]
free_thresh: 128

robot_diameter: 0.35
safety_margin: 0.02
cell_size: 0.30
""")
    print(f"\nsaved {path}")


if args.yaml:
    save_yaml(args.yaml, M, args.image)
