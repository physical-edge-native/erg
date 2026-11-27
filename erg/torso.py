"""Printed torso frame of the whole robot (Issue #53), and the torso's mass properties for the MuJoCo model.

Torso frame: origin at the MuJoCo torso body's origin, x forward, y left, z up; millimetres. The four abduction
shafts are at HIPS (the hip positions of the model), along x, with the horns forward.

The torso is one printed part (PLA, 100 % infill), a flat **deck** with four **end caps** under it:

- each abduction servo stands upright over its shaft, as on the bench stand (erg.frame.stand_part), and its end cap
  (erg.frame.end_cap) is part of the deck, so each leg is held the way it was tested on the bench;
- the deck, 3 mm thick, joins the four end caps 35.2 mm above the abduction shafts, out of reach of the hip yokes,
  whose bases sweep up to about 31 mm around the shafts over the abduction range;
- the battery hangs under the deck between the legs, held by a strap through four slots; each leg's electronics
  (microcontroller, regulator, servo bus adapter) sit on the deck inwards of its hip, and its IMU in a shallow recess right
  above the hip (glued or taped, so that it moves with the torso).

How to print: the deck's top face on the bed (it is flat), the end caps pointing up; 100 % infill.

tests/test_torso.py checks that no part of the torso, legs, servos and battery overlap over the joint ranges.

Masses that are not printed or servos (battery, electronics, wiring) are estimates, to be replaced with weighed
values after building.
"""

from build123d import Box, Location, Plane, Pos, Solid, mirror

from erg import frame

HIPS = {"FL": (90.0, 50.0), "FR": (90.0, -50.0), "HL": (-90.0, 50.0), "HR": (-90.0, -50.0)}  # x, y of each shaft
# The abduction servos stand upright over their shafts, as on the bench stand (erg.frame.leg_locations): each one's
# end cap wall is at DECK_BOTTOM, and the deck joins the four end caps. Lying the servos across the robot (the first
# idea, with a spine between them) does not work: the hip yoke's base sweeps into the end cap when the leg abducts
# inwards.
DECK_BOTTOM = frame.END_CAP_REACH  # 35.2 above the abduction shafts
DECK_THICKNESS = frame.WALL
DECK_TOP = DECK_BOTTOM + DECK_THICKNESS
CAP_HALF_LENGTH = frame.SERVO_FACE + frame.WALL  # an end cap reaches this far from its servo's mid-plane along x
DECK_HALF_LENGTH = 90.0 + CAP_HALF_LENGTH - frame.MID  # 108.4: to the outer ends of the end caps
DECK_HALF_WIDTH = 50.0 + frame.SERVO_WIDTH / 2  # 62.4: to the outer sides of the servos
# Battery: a 2S 2200 mAh hard-case LiPo, assumed at most 110 x 50 x 30 mm (the exact size depends on the model bought),
# hung under the deck between the legs by a 20 mm hook-and-loop strap through four slots.
BATTERY_SIZE = (110.0, 50.0, 30.0)
BATTERY_MASS = 0.13  # kg, an estimate
STRAP_SLOT = (22.0, 4.0)  # along x, along y
STRAP_SLOTS = [(sx * 25.0, sy * 28.0) for sx in (-1, 1) for sy in (-1, 1)]  # just outside the battery
# Each leg's electronics (microcontroller, regulator, servo bus adapter) on top of the deck, inwards of its hip.
ELECTRONICS_SIZE = (40.0, 24.0, 12.0)
ELECTRONICS_MASS = 0.015  # kg per leg, an estimate
ELECTRONICS_AT = {leg: (45.0 if x > 0 else -45.0, 30.0 if y > 0 else -30.0) for leg, (x, y) in HIPS.items()}
WIRING_MASS = 0.04  # kg, an estimate (servo cables, power wires, connectors, screws), spread over the deck
# IMU recesses: one per leg, in the deck's top face right above its hip, to place the board (Adafruit 4438,
# 25.4 x 17.8 x 1.6 mm, its long side across the robot); the deck's top face stays flat for printing it face down.
# The IMU's site in the MuJoCo model is on the board's top face.
IMU_RECESS = (18.4, 26.0, 0.6)
IMU_BOARD_THICKNESS = 1.6
IMU_SITES = {leg: (x, y, DECK_TOP - IMU_RECESS[2] + IMU_BOARD_THICKNESS) for leg, (x, y) in HIPS.items()}
# Holes in the deck for the servo cables, inwards of each servo, and to lighten the deck.
CABLE_HOLE = (10.0, 6.0)
LIGHTENING = [((sx * 25.0, 0.0), (30.0, 30.0)) for sx in (-1, 1)]


def abd_servo_location(leg: str) -> Location:
    """Location of a leg's abduction servo frame in the torso frame: the bench stand's orientation (erg.frame), moved
    to the hip. The same for right legs: the servo is symmetric across its own xz plane (y across the servo, here
    across the robot), so it is its own mirror image."""
    x, y = HIPS[leg]
    return Pos(x, y, 0) * frame.leg_locations()["abd_servo"]


def end_cap_location(leg: str) -> Location:
    return abd_servo_location(leg)


def torso_solid() -> Solid:
    """The printed torso frame: the deck and the four end caps of the abduction servos."""
    zc = DECK_BOTTOM + DECK_THICKNESS / 2
    solid = Pos(0, 0, zc) * Box(2 * DECK_HALF_LENGTH, 2 * DECK_HALF_WIDTH, DECK_THICKNESS)
    for x, y in STRAP_SLOTS:
        solid -= Pos(x, y, zc) * Box(*STRAP_SLOT, DECK_THICKNESS + 0.2)
    for (x, y), (lx, ly) in LIGHTENING:
        solid -= Pos(x, y, zc) * Box(lx, ly, DECK_THICKNESS + 0.2)
    for leg, (x, y) in HIPS.items():
        solid -= Pos(x, y - (19.0 if y > 0 else -19.0), zc) * Box(*CABLE_HOLE, DECK_THICKNESS + 0.2)
        solid += end_cap_location(leg) * frame.end_cap()
    for x, y, _ in IMU_SITES.values():
        solid -= Pos(x, y, DECK_TOP - IMU_RECESS[2] / 2 + 0.05) * Box(IMU_RECESS[0], IMU_RECESS[1], IMU_RECESS[2] + 0.1)
    return solid


def torso_part() -> frame.Part:
    return frame.Part("torso", torso_solid())


def battery_solid() -> Solid:
    return Pos(0, 0, DECK_BOTTOM - BATTERY_SIZE[2] / 2) * Box(*BATTERY_SIZE)


def electronics_solids() -> dict[str, Solid]:
    return {leg: Pos(x, y, DECK_TOP + ELECTRONICS_SIZE[2] / 2) * Box(*ELECTRONICS_SIZE)
            for leg, (x, y) in ELECTRONICS_AT.items()}


def leg_solids_in_torso(leg: str, abd: float = 0.0, flex: float = 0.0, knee: float = 0.0) -> dict[str, Solid]:
    """A leg's printed parts and servos placed in the torso frame (the bench stand is left out). Right legs are the
    mirror images of left legs; their joint angles are the model's (abduction about +x, flexion and knee about +y)."""
    x, y = HIPS[leg]
    left = leg.endswith("L")
    # Left legs: the leg frame (erg.frame) is the torso frame moved to the hip. Right legs: the mirror image of the
    # left leg with the abduction angle reversed (a rotation about x changes sign in the mirror image).
    solids = frame.leg_solids(abd if left else -abd, flex, knee)
    solids.pop("stand")
    solids.pop("abd_servo")  # placed by abd_servo_location instead
    out = {}
    for name, s in solids.items():
        s = s if left else mirror(s, Plane.XZ)
        out[name] = Pos(x, y, 0) * s
    out["abd_servo"] = abd_servo_location(leg) * frame.servo_envelope()
    return out


def inertial() -> tuple[float, tuple, tuple]:
    """Mass (kg), centre of mass (m, torso frame) and full inertia about it (kg m^2, MJCF fullinertia order) of the
    torso body: the printed frame, the four abduction servos, the battery, the electronics and the wiring (the wiring
    spread uniformly over the deck)."""
    import numpy as np

    servo = frame.servo_envelope()
    items = [(torso_solid(), frame.MATERIAL_DENSITY)]
    items += [(abd_servo_location(leg) * servo, frame.SERVO_MASS / servo.volume) for leg in HIPS]
    b = battery_solid()
    items.append((b, BATTERY_MASS / b.volume))
    for e in electronics_solids().values():
        items.append((e, ELECTRONICS_MASS / e.volume))
    deck = Pos(0, 0, DECK_BOTTOM + DECK_THICKNESS / 2) * Box(2 * DECK_HALF_LENGTH, 2 * DECK_HALF_WIDTH, DECK_THICKNESS)
    items.append((deck, WIRING_MASS / deck.volume))
    masses = [s.volume * rho for s, rho in items]
    centres = [np.array(tuple(s.center())) for s, _ in items]
    m = sum(masses)
    c = sum(mi * ci for mi, ci in zip(masses, centres)) / m
    inertia = np.zeros((3, 3))
    for (s, rho), mi, ci in zip(items, masses, centres):
        d = ci - c
        inertia += rho * np.array(s.matrix_of_inertia) + mi * (d @ d * np.eye(3) - np.outer(d, d))
    inertia *= 1e-6
    full = (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
    return m, tuple(c * 1e-3), full
