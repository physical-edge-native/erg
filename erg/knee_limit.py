"""The knee servo's torque limit near the straight end of the knee range (Issue #81), written once for the simulation
(erg/servo.py) and the leg firmware (firmware/leg-esp32/main/leg.c, with constants generated into obs_layout.h by
scripts/gen_firmware_layout.py).

The knee servo drives the shank through a parallelogram push rod (body L1; docs/design/2026-11-27-leg-print.md): a
crank of radius CRANK on the knee servo, the same crank on the shank, both at angle knee + CRANK_PHASE from the
thigh's axis, and a rod between them along the thigh. The parallelogram passes the torque on one to one; the rod's
force is torque / (CRANK * |sin(knee + CRANK_PHASE)|), which grows without bound as the cranks line up with the rod.
The rod is a carbon-fibre tube (the maintainer's choice of 2026-11-27); its allowable load is its Euler buckling
load divided by ROD_SAFETY. The rule:

    torque_limit(q) = min(stall torque, ALLOWABLE * CRANK * |sin(q + CRANK_PHASE)|)

keeps the rod's force under the allowable load at every knee angle. With the numbers below it is below the stall
torque only for knee angles from about -0.23 rad to the straight limit -0.2 rad (and beyond -2.91 rad on the bent
side, outside the range).

The sine is erg.exactmath's (additions, multiplications, divisions and square roots only), so that C and Python give
the same bits (tests/test_firmware_host.py). The servo's torque limit register holds tenths of a percent of the
servo's maximum torque, taken as the stall torque; the register value is rounded down, so the servo's limit is never
above the rule's.
"""

from math import floor, pi

from erg import design, exactmath

# --- Push rod (docs/design/2026-11-27-leg-print.md; the maintainer's choice of 2026-11-27) -------------------------
CRANK = 0.018  # m, crank radius at both ends of the parallelogram
CRANK_PHASE = 0.0  # rad, direction of both cranks at zero knee angle, from the thigh's axis
TUBE_OD, TUBE_ID = 4.0, 3.0  # mm, carbon-fibre tube (assumed 4 x 3 mm; check on the real tube)
TUBE_E = 100e3  # N/mm^2 (100 GPa), the low end of 100-130 GPa for pultruded carbon tubes
ROD_LENGTH = 77.0  # mm, between the ball centres (thigh 95 mm minus the knee servo's 18 mm below the hip shaft)
ROD_SAFETY = 2.0  # safety factor on the buckling load

STALL = design.SERVO_STALL_TORQUE  # N*m, the servo's maximum torque (register value 1000)
REGISTER_FULL = 1000  # torque limit register: tenths of a percent of the maximum torque
WRITE_STEP = 10  # the firmware raises the register only by at least this much (1 %), to save bus time


def rod_buckling_load() -> float:
    """Euler buckling load (N) of the tube, pinned at both ball centres."""
    second_moment = pi / 64 * (TUBE_OD**4 - TUBE_ID**4)
    return pi**2 * TUBE_E * second_moment / ROD_LENGTH**2


ALLOWABLE = rod_buckling_load() / ROD_SAFETY  # N
ARM = ALLOWABLE * CRANK  # N*m per unit |sin|: the torque at which the rod carries ALLOWABLE


def rod_force(knee: float, torque: float = STALL) -> float:
    """The rod's force (N) when the knee servo exerts `torque` (N*m) at knee angle `knee` (rad)."""
    return torque / (CRANK * abs(exactmath.sin(knee + CRANK_PHASE)))


def torque_limit(knee: float) -> float:
    """The knee servo's torque limit (N*m) at knee angle `knee` (rad): the rule above, operation by operation as in
    leg_knee_torque_limit (leg.c)."""
    a = ARM * abs(exactmath.sin(knee + CRANK_PHASE))
    return STALL if a > STALL else a


def register_value(knee: float) -> int:
    """The torque limit register's value (0..1000) for knee angle `knee`, rounded down."""
    v = floor(torque_limit(knee) / STALL * 1000.0)
    return min(max(v, 0), REGISTER_FULL)


def should_write(last: int, value: int) -> bool:
    """Whether the firmware writes `value` to the register that holds `last` (-1: not written yet): at once when it
    lowers the limit, when it raises it by WRITE_STEP or more, and when it goes back to full."""
    return last < 0 or value < last or value >= last + WRITE_STEP or (value == REGISTER_FULL and last != value)


# Knee angles at which torque_limit is the stall torque for certain (|sin| above STALL / ARM with a wide margin), so
# that the simulation can skip the sine there; tests/test_knee_limit.py checks the band's ends.
FULL_BAND = (-2.8, -0.3)
