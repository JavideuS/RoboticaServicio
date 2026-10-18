# The task gives GPS points, the drone works in local meters.
# GPS -> UTM (meters) -> offset from the boat, which is where the drone
# starts and so the origin of its position frame

from dataclasses import dataclass
import utm

# Degree, minutes, seconds
def dms(deg, minutes, seconds):
    return deg + minutes / 60 + seconds / 3600


@dataclass(frozen=True)
class Gps:  # degrees, north and east positive
    lat: float
    lon: float


# These are given
BOAT = Gps(dms(40, 16, 48.2), -dms(3, 49, 3.5))    # 40º16'48.2" N, 3º49'03.5" W
SURVIVORS = Gps(dms(40, 16, 47.23), -dms(3, 49, 1.78))  # 40º16'47.23" N, 3º49'01.78" W, approximate


def gps_to_local(point, origin=BOAT):
    # The exercise page says position control is north (+x), west (+y), but the
    # world is the usual ROS one: east (+x), north (+y), same axes as UTM.
    # With the page's axes the survivors would end up 60 m away from where they really are
    e0, n0, zone, _ = utm.from_latlon(origin.lat, origin.lon)
    e, n, _, _ = utm.from_latlon(point.lat, point.lon, force_zone_number=zone)
    return e - e0, n - n0


def local_to_gps(x, y, origin=BOAT):
    # The way back, to report where each person is
    e0, n0, zone, letter = utm.from_latlon(origin.lat, origin.lon)
    lat, lon = utm.to_latlon(e0 + x, n0 + y, zone, letter)
    return Gps(lat, lon)


if __name__ == "__main__":
    x, y = gps_to_local(SURVIVORS)
    print("survivors at x=%.2f y=%.2f m from the boat" % (x, y))
    print("and back:", local_to_gps(x, y), "given:", SURVIVORS)
