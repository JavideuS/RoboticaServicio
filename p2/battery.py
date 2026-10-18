# Simulated battery, in sim time (pose stamps), so it does not depend on how
# fast the simulator runs.
#
# It drains while flying and charges while landed on the boat. The return
# threshold is not a fixed percentage: it is what the trip home costs from
# where the drone is, plus a reserve.#
from dataclasses import dataclass


# Settings
@dataclass(frozen=True)
class Power:
    endurance: float = 180.0    # s of flight from full to empty (scaled, see above)
    reserve: float = 0.15   # never land with less than this, for wind or a missed landing
    charge_time: float = 10.0   # s from empty to full: on a real boat, a battery swap
    home_speed: float = 3.0     # m/s, average on the way back, braking and turning included
    landing: float = 5.0   # s that landing takes


POWER = Power()


class Battery:
    def __init__(self):
        self.level = 1. # Battery model between 0-1
        self.last = None    # s, sim time of the last update
        self.was_flying = False

    def update(self, now, flying, charging):
        if self.last is not None:
            dt = max(0.0, now - self.last)
            # Takeoff and landing block the loop, so one dt can cover all of it:
            # if it started or ended in the air, it was spent flying
            if flying or self.was_flying:
                self.level = max(0.0, self.level - dt / POWER.endurance)
            elif charging:
                self.level = min(1.0, self.level + dt / POWER.charge_time)
        self.last = now
        self.was_flying = flying

    # What going home from this distance costs, plus the reserve
    def threshold(self, distance_home):
        trip = distance_home / POWER.home_speed + POWER.landing
        return trip / POWER.endurance + POWER.reserve

    def must_return(self, distance_home):
        return self.level <= self.threshold(distance_home)

    @property
    def full(self):
        return self.level >= 1.0
