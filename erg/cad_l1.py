"""Layout-level code-CAD of body L1 (Issue #59), with build123d.

This first CAD of the redesign fixes **where every part is and how much space it takes**, so that the body can be
checked for collisions over the joint ranges and its printed masses estimated before anything is ordered. It is not
yet print-ready: screw holes, bearing pins, the push rod's ball joints and the switch pocket are drawn on these
placements in erg/cad_leg.py (Issue #72, one leg for a test print).

Parts, for the front-left leg (the others are mirror images), all PLA at full infill:

- **torso**: a floor and a deck joined by side walls, with a cradle for each abduction servo (shaft along x, horn at
  X_FACE), the battery lying across between the servos, and the electronics on the deck;
- **shoulder**: a plate on the abduction horn, in front of the torso, that holds the hip flexion servo by its far
  (upper) end;
- **thigh**: a plate on the hip flexion horn, a cap holding the knee servo by its far (upper) end, and a beam under the
  knee servo out to the knee pin in the leg's plane;
- **shank**: from the knee pin to the foot, with the crank that the push rod drives;
- **crank and rod**: a crank on the knee servo's horn and a rod parallel to the thigh to the shank's crank (a
  parallelogram, so the knee turns by the knee servo's angle).

Units: millimetres. Body frame as in erg/design.py (x forward, y left, z up, origin on the abduction shafts' plane).
"""

from math import degrees

from build123d import Box, Cylinder, Location, Plane, Pos, Rot, Solid, mirror

from erg import design

MM = 1000.0
SERVO = (45.2, 24.7, 31.8)  # body length (x), width (y), height between the faces (z), in the servo's frame
HORN_R, HORN_H = 10.0, design.SERVO_HORN * MM - SERVO[2] / 2  # horn disc radius; how far it stands off the face
IDLER_H = 1.8
DENSITY = 1.24e-6  # kg/mm^3, PLA at full infill
WALL = 2.5

HIP_X, HIP_Y = design.HIP_X * MM, design.HIP_Y * MM
X_FACE = design.X_FACE * MM
HIP_HORN_Y = design.HIP_HORN_Y * MM
KNEE_HORN_Y = design.KNEE_HORN_Y * MM
LEG_Y = (design.HIP_Y + design.LEG_OFFSET) * MM  # the leg's plane: knee pin, shank and foot
THIGH, SHANK = design.THIGH * MM, design.SHANK * MM
KNEE_DOWN = design.KNEE_SERVO_DOWN * MM
SHAFT = design.SERVO_SHAFT * MM
FOOT_R = design.FOOT_RADIUS * MM
CRANK = 18.0  # crank radius, both ends of the parallelogram (12 mm until the maintainer's choice of 2026-11-27, #72)


def servo() -> Solid:
    """A servo's space in its own frame: body centred on the origin, shaft along z at x = SHAFT, horn on +z."""
    body = Box(*SERVO)
    horn = Pos(SHAFT, 0, SERVO[2] / 2 + HORN_H / 2) * Cylinder(HORN_R, HORN_H)
    idler = Pos(SHAFT, 0, -SERVO[2] / 2 - IDLER_H / 2) * Cylinder(HORN_R, IDLER_H)
    return body + horn + idler


def end_cap(margin: float = 0.4) -> Solid:
    """A cap around the far end of a servo (x = -SERVO[0]/2), in the servo's frame, that holds it by four screws."""
    x0 = -SERVO[0] / 2
    outer = Pos(x0 + 6 - WALL, 0, 0) * Box(12 + 2 * WALL, SERVO[1] + 2 * WALL, SERVO[2] + 2 * WALL)
    inner = Pos(x0 + 6 + 0.5, 0, 0) * Box(13, SERVO[1] + 2 * margin, SERVO[2] + 2 * margin)
    return outer - inner


# --- Placements (front-left), as build123d Locations in the body frame -------------------------------------------

def abd_servo_loc() -> Location:
    """Shaft along +x with the horn face at X_FACE, body inwards along y, width along z."""
    return Location(Plane(origin=(X_FACE - design.SERVO_HORN * MM, HIP_Y - SHAFT, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)))


def hip_servo_loc() -> Location:
    """In the hip frame at zero abduction (= body frame): shaft along +y at (HIP_X, *, 0), horn face at HIP_HORN_Y,
    body upwards."""
    return Location(Plane(origin=(HIP_X, HIP_HORN_Y - design.SERVO_HORN * MM, SHAFT), x_dir=(0, 0, -1), z_dir=(0, 1, 0)))


def knee_servo_loc() -> Location:
    """In the thigh frame (origin on the hip flexion shaft, axes as the body's at zero angles): shaft along +y,
    KNEE_DOWN below the hip flexion shaft, horn face at KNEE_HORN_Y, body upwards."""
    y = KNEE_HORN_Y - design.SERVO_HORN * MM - LEG_Y
    return Location(Plane(origin=(0, y, -KNEE_DOWN + SHAFT), x_dir=(0, 0, -1), z_dir=(0, 1, 0)))


# --- Parts -------------------------------------------------------------------------------------------------------

def torso() -> Solid:
    """Floor and deck (2 mm, with windows) joined by side walls, and a cap for each abduction servo."""
    t = 2.0
    length = 2 * (X_FACE - 0.5)
    width = 2 * (HIP_Y + 11.0)  # the abduction servos' outer sides at HIP_Y + 10.2
    h = SERVO[1] + 0.8
    plates = Pos(0, 0, -h / 2 - t / 2) * Box(length, width, t) + Pos(0, 0, h / 2 + t / 2) * Box(length, width, t)
    for z in (-h / 2 - t / 2, h / 2 + t / 2):
        plates -= Pos(0, 0, z) * Box(length - 70, width - 16, t + 0.2)
    walls = Pos(0, width / 2 - 1, 0) * Box(length, 2, h) + Pos(0, -width / 2 + 1, 0) * Box(length, 2, h)
    solid = plates + walls
    cap = abd_servo_loc() * end_cap()
    for c in (cap, mirror(cap, Plane.XZ), mirror(cap, Plane.YZ), mirror(mirror(cap, Plane.XZ), Plane.YZ)):
        solid += c
    return solid


def shoulder() -> Solid:
    """On the abduction horn (in the hip frame = body frame at zero abduction): a plate in front of X_FACE around the
    abduction shaft, rising to the cap that holds the hip flexion servo's upper end."""
    x = X_FACE + HORN_H + 2.0  # just in front of the horn
    y0, y1 = HIP_Y - 14.0, HIP_HORN_Y - 2.0  # stays inside the thigh's inner plate
    plate = Pos(x + WALL / 2, (y0 + y1) / 2, 12) * Box(WALL, y1 - y0, 60)
    cap = hip_servo_loc() * end_cap()
    bridge = Pos((x + HIP_X - SERVO[1] / 2) / 2 + 1, HIP_HORN_Y - design.SERVO_HORN * MM, 12 + SERVO[0] / 2 + 2) * Box(
        HIP_X - SERVO[1] / 2 - x + 2, SERVO[2] + 2 * WALL, 2 * WALL)
    return plate + cap + bridge


def thigh() -> Solid:
    """In the thigh frame: a plate on the hip flexion horn down to the knee, the cap on the knee servo's upper end,
    and a bar at the knee from the plate out to the knee pin in the leg's plane."""
    y_plate = HIP_HORN_Y + 1.0 - LEG_Y  # inner plate's inner face, relative to the leg's plane
    plate = Pos(0, y_plate + WALL / 2, -THIGH / 2 + 6) * Box(16, WALL, THIGH + 12)
    cap = knee_servo_loc() * end_cap()
    span = -y_plate - WALL  # from the inner plate out to the knee hub, under the knee servo
    bar = Pos(0, y_plate + WALL + span / 2, -THIGH) * Box(10, span, 8)
    hub = Pos(0, -3, -THIGH) * (Rot(90, 0, 0) * Cylinder(7, 6))
    return plate + cap + bar + hub


def shank() -> Solid:
    """In the shank frame (origin on the knee pin, in the leg's plane): a link to the foot sphere, and the crank."""
    link = Pos(0, 3, -SHANK / 2) * Box(10, 5, SHANK)
    hub = Pos(0, 3, 0) * (Rot(90, 0, 0) * Cylinder(7, 6))
    foot = Pos(0, 3, -SHANK) * Solid.make_sphere(FOOT_R)
    crank = Pos(0, 7, CRANK / 2) * Box(8, 4, CRANK + 6)
    return link + hub + foot + crank


def crank_and_rod(knee_angle: float = 0.0) -> Solid:
    """In the thigh frame: the crank on the knee servo's horn and the rod down to the shank's crank, parallel to the
    thigh."""
    y = KNEE_HORN_Y + 3.0 - LEG_Y
    crank = Pos(0, y + 2, -KNEE_DOWN + CRANK / 2) * Box(8, 4, CRANK + 6)
    rod = Pos(0, y + 6, -(KNEE_DOWN + THIGH) / 2 + CRANK) * Box(5, 4, THIGH - KNEE_DOWN)
    return crank + rod


def mass(solid: Solid) -> float:
    return solid.volume * DENSITY


# --- Assembly ----------------------------------------------------------------------------------------------------

def leg_solids(abd: float = 0.0, flex: float = 0.0, knee: float = 0.0) -> dict[str, Solid]:
    """The front-left leg's parts and servos placed in the body frame at the given joint angles (radians)."""
    hip = Pos(0, HIP_Y, 0) * Rot(degrees(abd), 0, 0) * Pos(0, -HIP_Y, 0)
    thigh_f = hip * Pos(HIP_X, LEG_Y, 0) * Rot(0, degrees(flex), 0)
    shank_f = thigh_f * Pos(0, 0, -THIGH) * Rot(0, degrees(knee), 0)
    return {
        "shoulder": hip * shoulder(),
        "hip_servo": hip * (hip_servo_loc() * servo()),
        "thigh": thigh_f * thigh(),
        "knee_servo": thigh_f * (knee_servo_loc() * servo()),
        "rod": thigh_f * crank_and_rod(),
        "shank": shank_f * shank(),
    }


def body_solids() -> dict[str, Solid]:
    return {"torso": torso(), "abd_servo": abd_servo_loc() * servo()}
