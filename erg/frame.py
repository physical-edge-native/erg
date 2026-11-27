"""Printed frame of one leg (Issue #25), as solids built with build123d.

The leg is built from U-shaped brackets ("yokes"), the usual way of joining hobby serial bus servos: a yoke holds
the output horn of one servo on one side and the idler (the free bearing on the opposite face) on the other side,
and carries the next servo by an end cap screwed to that servo's top and bottom faces at its far end.

Printed parts of one leg (PLA, 100 % infill, on a Bambu Lab X2D fused-filament printer; decision of 2026-11-25):

- ``stand``: end cap of the abduction servo with a flat foot, to fix one leg on a bench (stage 1 of the parts list);
- ``hip``: yoke of the abduction joint + end cap of the hip flexion servo;
- ``thigh``: yoke of the hip flexion joint + end cap of the knee servo;
- ``shank``: yoke of the knee joint, ending in the foot.

Servo dimensions are those of the Feetech STS3215 CAD model in the SO-ARM100 project
(https://github.com/TheRobotStudio/SO-ARM100, file STEP/SO100/STS3215_03a.step, Apache-2.0), read on 2026-11-25.
Only measured values are used here; the file itself is not copied. They are to be checked with calipers on a real
servo before the parts are ordered.

Units: millimetres. The servo frame: body centred on the origin, length along x, output shaft along z at x = 12.5
with the horn on +z.
"""

from dataclasses import dataclass
from math import pi

from build123d import Box, Cylinder, Location, Plane, Pos, Rot, Solid, Sphere, mirror

# --- Feetech STS3215, from the SO-ARM100 CAD model ------------------------------------------------------------------
SERVO_LENGTH = 45.4  # x
SERVO_WIDTH = 24.8  # y
SERVO_FACE = 15.9  # top and bottom faces at z = +-15.9
AXIS_X = 12.5  # output shaft, from the body centre
HORN_RADIUS = 10.0
HORN_TOP = 18.7  # mounting face of the output horn (z)
IDLER_BOTTOM = -17.7  # mounting face of the idler on the bottom (z)
HORN_HOLE_OFFSET = 4.9  # four holes at (+-4.9, +-4.9) around the shaft, on horn and idler
HORN_HOLE_DIAMETER = 2.5
# Threaded holes for frame screws in the top and bottom faces (pilot diameter 1.5 mm), at the far end from the shaft.
TOP_FRAME_HOLES = ((-16.5, 10.2), (-16.5, -10.2))
BOTTOM_FRAME_HOLES = ((-20.3, 10.2), (-20.3, -10.2))
# Parts of the case that stand out of the bottom face: the cable connectors and a raised plate.
BOTTOM_CONNECTORS = ((-4.6, 1.4), (-8.6, 8.6), (-19.4, -SERVO_FACE))  # x range, y range, z range
BOTTOM_PLATE = ((-17.5, -7.5), (-10.0, 10.0), (-17.7, -SERVO_FACE))
SERVO_MASS = 0.055  # kg, design record

# --- Frame ----------------------------------------------------------------------------------------------------------
MATERIAL_DENSITY = 1.24e-6  # kg/mm^3 (PLA, 1.24 g/cm^3, printed with 100 % infill)
# Holes are drawn 0.3 mm larger than the screw's usual clearance hole, because fused-filament printing makes holes
# come out smaller than drawn.
CLEARANCE_HOLE = 3.0  # for the horn screws (to be matched to the screws that come with the servo)
FRAME_SCREW_HOLE = 2.5  # for M2 self-tapping screws into the servo's 1.5 mm pilot holes
WOOD_SCREW_HOLE = 3.7  # for M3 wood screws through the stand
WALL = 3.0
BASE = 4.0
ARM = 4.0
BOSS = 3.0  # spacer between a horn (or idler) and the arm, so that arms clear the next servo's end cap
THIGH_BOSS = 7.5  # thigh arms sit outside the shank arms so that the folded knee does not hit them
ARM_HUB = 12.0  # radius of an arm around its shaft

MID = (HORN_TOP + IDLER_BOTTOM) / 2  # mid-plane between horn and idler faces, in the servo frame
HALF_SPAN = HORN_TOP - MID  # from the mid-plane to the horn (and idler) face
END_CAP_REACH = SERVO_LENGTH / 2 + AXIS_X  # from the far end of the servo to its shaft: 35.2

# Leg dimensions (the MuJoCo model follows these; tests/test_frame.py checks it).
HIP_YOKE_RADIUS = 19.0  # inner face of the hip yoke's base, from the abduction shaft
HIP_DROP = HIP_YOKE_RADIUS + BASE + WALL + END_CAP_REACH  # abduction shaft to hip flexion shaft: 61.2
THIGH_LENGTH = 90.0  # hip flexion shaft to knee shaft
THIGH_YOKE_RADIUS = THIGH_LENGTH - BASE - WALL - END_CAP_REACH
SHANK_LENGTH = 90.0  # knee shaft to the centre of the foot
FOOT_RADIUS = 12.0

# --- Foot switch: Omron SS-5GLD (Issue #53) ---------------------------------------------------------------------------
# Body size from the Omron SS-5 series datasheet (12.8 x 5.8 x 6.5 mm), to be checked with calipers on a real switch,
# with a test fit of one printed shank, before printing all four. The switch lies in a pocket inside the foot, its
# actuator facing the sole; the lever comes out through a slot in the sole, so the switch closes when the foot carries
# weight. It is slid in from the side of the foot and fixed with a drop of glue; its wires leave through a channel
# along the shank.
SWITCH_LENGTH = 12.8  # along the lever
SWITCH_THICKNESS = 5.8  # across
SWITCH_HEIGHT = 6.5  # from the back to the actuator face
SWITCH_CLEARANCE = 0.2  # each side, in the pocket
SWITCH_MASS = 0.001  # kg, an estimate (to be weighed)
SOLE_GAP = 2.0  # from the actuator face to the bottom of the foot, for the lever
LEVER_SLOT_WIDTH = 4.0
WIRE_CHANNEL = 3.0  # diameter


@dataclass
class Part:
    name: str
    solid: Solid

    @property
    def mass(self) -> float:
        return self.solid.volume * MATERIAL_DENSITY


def servo_envelope() -> Solid:
    """The space a servo takes, in its own frame (for interference checks)."""
    body = Box(SERVO_LENGTH, SERVO_WIDTH, 2 * SERVO_FACE)
    horn = Pos(AXIS_X, 0, (SERVO_FACE + HORN_TOP) / 2) * Cylinder(HORN_RADIUS, HORN_TOP - SERVO_FACE)
    idler = Pos(AXIS_X, 0, (IDLER_BOTTOM - SERVO_FACE) / 2) * Cylinder(HORN_RADIUS, -SERVO_FACE - IDLER_BOTTOM)
    parts = [body, horn, idler]
    for (x0, x1), (y0, y1), (z0, z1) in (BOTTOM_CONNECTORS, BOTTOM_PLATE):
        parts.append(Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0))
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def _arm(z0: float, thickness: float, reach: float, width: float, boss: float, up: bool) -> Solid:
    """One arm of a yoke in the yoke frame (shaft along z at the origin, base along +x): a plate from the shaft to
    the base, with a boss that bears on the horn or idler, and the four screw holes and a centre hole."""
    sign = 1 if up else -1
    hub = Cylinder(ARM_HUB, thickness)
    bar = Pos(reach / 2, 0, 0) * Box(reach, width, thickness)
    plate = Pos(0, 0, sign * (z0 + boss + thickness / 2)) * (hub + bar)
    spacer = Pos(0, 0, sign * (z0 + boss / 2)) * Cylinder(HORN_RADIUS, boss)
    solid = plate + spacer
    depth = boss + thickness
    for dx in (-HORN_HOLE_OFFSET, HORN_HOLE_OFFSET):
        for dy in (-HORN_HOLE_OFFSET, HORN_HOLE_OFFSET):
            solid -= Pos(dx, dy, sign * (z0 + depth / 2)) * Cylinder(CLEARANCE_HOLE / 2, depth + 0.2)
    solid -= Pos(0, 0, sign * (z0 + depth / 2)) * Cylinder(2.5, depth + 0.2)
    return solid


def yoke(reach: float, width: float, boss: float, base: bool = True) -> Solid:
    """A yoke in its frame: shaft along z through the origin, the mid-plane between horn and idler at z = 0,
    base plate (normal x) with its inner face at x = reach."""
    outer = HALF_SPAN + boss + ARM
    solid = _arm(HALF_SPAN, ARM, reach + (BASE if base else 0), width, boss, up=True)
    solid += _arm(HALF_SPAN, ARM, reach + (BASE if base else 0), width, boss, up=False)
    if base:
        solid += Pos(reach + BASE / 2, 0, 0) * Box(BASE, width, 2 * outer)
    return solid


def end_cap() -> Solid:
    """End cap of a servo, in the servo frame: an end wall at the far end (x = -22.7) with plates on the top and
    bottom faces, screwed into the servo's frame holes."""
    x_end = -SERVO_LENGTH / 2
    top_len = 13.0
    bottom_len = 4.7  # stops before the raised plate on the bottom face
    wall = Pos(x_end - WALL / 2, 0, 0) * Box(WALL, SERVO_WIDTH, 2 * (SERVO_FACE + WALL))
    top = Pos(x_end - WALL + (top_len + WALL) / 2, 0, SERVO_FACE + WALL / 2) * Box(top_len + WALL, SERVO_WIDTH, WALL)
    bottom = Pos(x_end - WALL + (bottom_len + WALL) / 2, 0, -SERVO_FACE - WALL / 2) * Box(
        bottom_len + WALL, SERVO_WIDTH, WALL
    )
    solid = wall + top + bottom
    for x, y in TOP_FRAME_HOLES:
        solid -= Pos(x, y, SERVO_FACE + WALL / 2) * Cylinder(FRAME_SCREW_HOLE / 2, WALL + 0.2)
    for x, y in BOTTOM_FRAME_HOLES:
        solid -= Pos(x, y, -SERVO_FACE - WALL / 2) * Cylinder(FRAME_SCREW_HOLE / 2, WALL + 0.2)
    return solid


# Where the next servo sits in a yoke frame: its long axis along the yoke's x, its far end on the base, its own shaft
# parallel to the yoke's shaft (thigh, shank) or turned by 90 degrees about x (hip: flexion shaft across abduction).
def next_servo_location(reach: float, turned: bool) -> Location:
    x0 = reach + BASE + WALL + SERVO_LENGTH / 2
    rot = Rot(-90, 0, 0) if turned else Rot(0, 0, 0)
    # put the next servo's mid-plane (z = MID in its frame) on the yoke's mid-plane
    return Pos(x0, 0, 0) * rot * Pos(0, 0, -MID)


def stand_part() -> Part:
    """End cap of the abduction servo with a flat foot drilled for four wood screws (M3), in the servo frame."""
    cap = end_cap()
    foot = Pos(-SERVO_LENGTH / 2 - WALL - BASE / 2, 0, 0) * Box(BASE, 60.0, 60.0)
    solid = cap + foot
    for y in (-22.0, 22.0):
        for z in (-22.0, 22.0):
            solid -= Pos(-SERVO_LENGTH / 2 - WALL - BASE / 2, y, z) * (Rot(0, 90, 0) * Cylinder(WOOD_SCREW_HOLE / 2, BASE + 0.2))
    return Part("stand", solid)


def hip_part() -> Part:
    """In the abduction yoke frame."""
    width = 2 * (SERVO_FACE + WALL) + 4.0  # holds the turned flexion servo's end cap, top to bottom
    solid = yoke(HIP_YOKE_RADIUS, width, BOSS) + next_servo_location(HIP_YOKE_RADIUS, turned=True) * end_cap()
    return Part("hip", solid)


def thigh_part() -> Part:
    """In the hip flexion yoke frame."""
    solid = yoke(THIGH_YOKE_RADIUS, SERVO_WIDTH + 2.0, THIGH_BOSS)
    solid += next_servo_location(THIGH_YOKE_RADIUS, turned=False) * end_cap()
    return Part("thigh", solid)


def shank_part() -> Part:
    """In the knee yoke frame: arms to a block that joins them, ending in the foot (a sphere) at SHANK_LENGTH, with a
    pocket for the foot switch (Issue #53)."""
    reach = SHANK_LENGTH - FOOT_RADIUS - 6.0
    width = 20.0
    solid = yoke(reach, width, BOSS, base=False)
    outer = HALF_SPAN + BOSS + ARM
    block_len = SHANK_LENGTH - reach
    solid += Pos(reach + block_len / 2, 0, 0) * Box(block_len, width, 2 * outer)
    solid += Pos(SHANK_LENGTH, 0, 0) * Sphere(FOOT_RADIUS)
    return Part("shank", solid - switch_pocket())


def switch_body() -> Solid:
    """The switch body in its pocket, in the knee yoke frame (for mass and interference checks)."""
    top = SHANK_LENGTH + FOOT_RADIUS - SOLE_GAP - SWITCH_CLEARANCE
    return Pos(top - SWITCH_HEIGHT / 2, 0, 0) * Box(SWITCH_HEIGHT, SWITCH_LENGTH, SWITCH_THICKNESS)


def switch_pocket() -> Solid:
    """The space cut from the foot for the switch, in the knee yoke frame (the sole is at +x): the pocket, open
    towards +z so that the switch slides in from the side; the lever slot through the sole; the wire channel."""
    c = SWITCH_CLEARANCE
    top = SHANK_LENGTH + FOOT_RADIUS - SOLE_GAP  # actuator face
    h, length, t = SWITCH_HEIGHT + 2 * c, SWITCH_LENGTH + 2 * c, SWITCH_THICKNESS + 2 * c
    z_open = FOOT_RADIUS + 1.0
    pocket = Pos(top - h / 2, 0, (z_open - t / 2) / 2) * Box(h, length, z_open + t / 2)
    slot = Pos(top + SOLE_GAP, 0, 0) * Box(2 * SOLE_GAP + 0.2, length, LEVER_SLOT_WIDTH)
    reach = 20.0
    channel = Pos(top - h - reach / 2 + 1.0, 0, 0) * (Rot(0, 90, 0) * Cylinder(WIRE_CHANNEL / 2, reach))
    return pocket + slot + channel


def parts() -> list[Part]:
    """The printed parts of a left leg, and the stand for the bench."""
    return [stand_part(), hip_part(), thigh_part(), shank_part()]


# The right legs are mirror images of the left ones (y -> -y in the leg frame): the hip in its yoke frame is
# mirrored across its xz plane (the yoke's y is the leg's y), the thigh and shank across their xy plane (their shaft,
# z, is along the leg's y). The servos need no mirror image: an STS3215 is symmetric across its own xz plane, so its
# mirror image is the same servo turned half a turn about its long axis, horn and idler swapped.
MIRROR_PLANES = {"hip": Plane.XZ, "thigh": Plane.XY, "shank": Plane.XY}


def right_parts() -> list[Part]:
    """The printed parts of a right leg."""
    return [Part(f"{p.name}_right", mirror(p.solid, MIRROR_PLANES[p.name])) for p in parts() if p.name in MIRROR_PLANES]


# --- Assembly ---------------------------------------------------------------------------------------------------------
# Leg frame: origin on the abduction shaft, on the hip yoke's mid-plane; x forward, y outward, z up, as in the MuJoCo
# model. Joint angles follow the MuJoCo joints: abduction about x, flexion and knee about y; zero = leg straight down.

DEG = 180 / pi


def leg_locations(abd: float = 0.0, flex: float = 0.0, knee: float = 0.0) -> dict[str, Location]:
    """Location of each part's and each servo's frame in the leg frame, for the given joint angles (radians)."""
    # Abduction servo, fixed: shaft along leg x with the horn forward, body upwards, mid-plane at leg x = 0.
    abd_servo = Location(Plane(origin=(0, 0, 0), x_dir=(0, 0, -1), z_dir=(1, 0, 0))) * Pos(-AXIS_X, 0, -MID)
    # Hip yoke: shaft along leg x, base straight down at abd = 0.
    hip = Rot(abd * DEG, 0, 0) * Location(Plane(origin=(0, 0, 0), x_dir=(0, 0, -1), z_dir=(1, 0, 0)))
    flex_servo = hip * next_servo_location(HIP_YOKE_RADIUS, turned=True)
    # Thigh yoke: shaft along the flexion shaft (leg y), base straight down at flex = 0.
    thigh = hip * Location(Plane(origin=(HIP_DROP, 0, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0))) * Rot(0, 0, flex * DEG)
    knee_servo = thigh * next_servo_location(THIGH_YOKE_RADIUS, turned=False)
    shank = thigh * Pos(THIGH_LENGTH, 0, 0) * Rot(0, 0, knee * DEG)
    return {
        "stand": abd_servo,
        "abd_servo": abd_servo,
        "hip": hip,
        "flex_servo": flex_servo,
        "thigh": thigh,
        "knee_servo": knee_servo,
        "shank": shank,
    }


def leg_solids(abd: float = 0.0, flex: float = 0.0, knee: float = 0.0) -> dict[str, Solid]:
    """Printed parts and servo envelopes placed in the leg frame."""
    loc = leg_locations(abd, flex, knee)
    servo = servo_envelope()
    out = {p.name: loc[p.name] * p.solid for p in parts()}
    for name in ("abd_servo", "flex_servo", "knee_servo"):
        out[name] = loc[name] * servo
    out["switch"] = loc["shank"] * switch_body()
    return out


def foot_position(abd: float = 0.0, flex: float = 0.0, knee: float = 0.0):
    """Centre of the foot in the leg frame."""
    return (leg_locations(abd, flex, knee)["shank"] * Pos(SHANK_LENGTH, 0, 0)).position


# --- Mass properties for the MuJoCo model -----------------------------------------------------------------------------
# Each MuJoCo body of a leg moves with one joint: "hip" (abduction) carries the hip part and the flexion servo,
# "thigh" (flexion) the thigh part and the knee servo, "shank" (knee) the shank part. A servo counts as a solid of
# uniform density over its envelope with SERVO_MASS.

BODY_CONTENTS = {"hip": ("hip", "flex_servo"), "thigh": ("thigh", "knee_servo"), "shank": ("shank", "switch")}
# Origin of each MuJoCo body in the leg frame at zero angles (on its joint's axis).
BODY_ORIGINS = {"hip": (0.0, 0.0, 0.0), "thigh": (0.0, 0.0, -HIP_DROP), "shank": (0.0, 0.0, -HIP_DROP - THIGH_LENGTH)}


def body_inertials() -> dict[str, tuple[float, tuple, tuple]]:
    """Per MuJoCo body: mass (kg), centre of mass (m, in the body frame) and full inertia about the centre of mass
    (kg m^2, as MJCF fullinertia: ixx iyy izz ixy ixz iyz), at zero joint angles."""
    import numpy as np

    solids = leg_solids()
    servo_density = SERVO_MASS / servo_envelope().volume
    out = {}
    for body, names in BODY_CONTENTS.items():
        density = {n: servo_density if n.endswith("_servo") else MATERIAL_DENSITY for n in names}
        if "switch" in density:
            density["switch"] = SWITCH_MASS / solids["switch"].volume
        items = [(solids[n], density[n]) for n in names]
        masses = [s.volume * rho for s, rho in items]
        centres = [np.array(tuple(s.center())) for s, _ in items]
        m = sum(masses)
        c = sum(mi * ci for mi, ci in zip(masses, centres)) / m
        inertia = np.zeros((3, 3))
        for (s, rho), mi, ci in zip(items, masses, centres):
            d = ci - c
            inertia += rho * np.array(s.matrix_of_inertia) + mi * (d @ d * np.eye(3) - np.outer(d, d))
        inertia *= 1e-6  # kg mm^2 -> kg m^2
        com = (c - np.array(BODY_ORIGINS[body])) * 1e-3
        full = (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
        out[body] = (m, tuple(com), full)
    return out
