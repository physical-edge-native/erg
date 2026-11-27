"""Print-ready parts of the front-left leg of body L1 and its abduction servo's cradle (Issue #72), with build123d.

The layout CAD (erg/cad_l1.py, Issue #59) fixes where each part is and how much space it takes. This module draws the
details needed to print one leg for a test fit, on the same placements:

- **servo interfaces**: the four frame screw holes of each end cap (M2, countersunk so that no head stands out) with a
  notch for the raised plate on the servo's bottom face, and the horn screw patterns (M2.5, countersunk) with a centre
  hole on every part screwed to a horn;
- **knee**: an M3 screw through a 3 x 5 mm bronze sleeve in the shank, into a nut held in the thigh, so that the shank
  turns on the sleeve and the screw clamps only the sleeve;
- **push rod**: a crank on the knee servo's horn and a crank on the shank, each with a hole at radius CRANK for an M2
  ball stud (nut in a hexagonal pocket on the back); a printed rod between two M2 ball links, outside the knee screw's
  head, so that it never meets the knee hubs over the whole knee range;
- **foot**: a separate printed foot pushed onto the end of the shank and held by one M2 screw, with a pocket for the
  Omron SS-5GLD foot switch (slid in from the outer side, lever out through a slot in the bottom), a cylindrical seat
  with a groove for a rubber cap over the 12 mm-radius sphere, and a hole for the switch's wires;
- **cable passages**: pairs of slots for cable ties on the thigh (ties go round the shank), and the opening in the
  torso's side wall around the abduction servo's near end;
- **print orientation**: every part has a flat face to lie on (PRINT_ORIENTATION), with no overhang larger than a
  short bridge.

Every dimension of the servo interface is a named parameter and marked "check on the real servo": the values are
those of the Feetech STS3215 (from the SO-ARM100 project's CAD model, read on 2026-11-25, as in the first body's
erg/frame.py), whose case the 12 V ST3215 is said to share; the ST3215's drawing is to be checked when it arrives.

Holes are drawn larger than the screw by the usual clearance plus 0.3 mm, because holes printed by fused-filament
printers come out smaller than drawn (as in the first body).

Units: millimetres. Frames as in erg/cad_l1.py.
"""

from math import cos, degrees, sin
from pathlib import Path

from build123d import Box, Cone, Cylinder, Location, Plane, Pos, Rot, Solid, export_step, export_stl, mirror

from erg import cad_l1 as L
from erg import design

# --- Servo interface (STS3215 case; check on the real servo) -------------------------------------------------------
# In the servo's frame (erg/cad_l1.servo): body centred on the origin, length along x, shaft along z at x = SHAFT,
# horn on +z ("top"), idler on -z ("bottom").
SERVO_LENGTH, SERVO_WIDTH, SERVO_HEIGHT = L.SERVO  # check on the real servo
SERVO_FACE = SERVO_HEIGHT / 2  # top and bottom faces at z = +-SERVO_FACE
HORN_HOLE_OFFSET = 4.9  # four horn screw holes at (+-4.9, +-4.9) around the shaft; check on the real servo
HORN_CENTRE_HOLE = 5.0  # clears the horn's centre screw head; check on the real servo
# Threaded (1.5 mm pilot) holes for the frame screws, in the top and bottom faces; check on the real servo.
TOP_FRAME_HOLES = ((-16.5, 10.2), (-16.5, -10.2))
BOTTOM_FRAME_HOLES = ((-20.3, 10.2), (-20.3, -10.2))
# Parts of the case standing out of the bottom face (x range, y range, z range); check on the real servo.
BOTTOM_PLATE = ((-17.5, -7.5), (-10.0, 10.0), (-17.7, -SERVO_FACE))
BOTTOM_CONNECTORS = ((-4.6, 1.4), (-8.6, 8.6), (-19.4, -SERVO_FACE))

# --- Screws and fits ---------------------------------------------------------------------------------------------
M2_HOLE = 2.5  # clearance for M2 (frame screws into the servo's pilot holes, ball studs, foot screw)
M2_CSK = 4.2  # countersink for an M2 flat head (3.8 mm head)
M2_PILOT = 1.6  # for an M2 thread cut into the printed rod's ends
M2_NUT = 4.3  # across flats of the pocket for an M2 nut (4.0 mm)
M2_NUT_T = 1.8  # pocket depth (nut 1.6 mm)
HORN_SCREW_HOLE = 3.0  # clearance for the horn screws (M2.5); check on the real servo's horn screws
HORN_SCREW_CSK = 5.2  # countersink for an M2.5 flat head (5.0 mm head)
M3_HOLE = 3.5  # clearance for the knee screw (M3)
M3_NUT = 5.8  # across flats of the slot for the knee screw's M3 nut (5.5 mm)
M3_NUT_T = 2.7  # slot thickness (nut 2.4 mm)
KNEE_HEAD_D = 6.4  # counterbore for the knee screw's low head (DIN 7984 M3: 5.5 mm across, 2.0 mm high)
SLEEVE_OD, SLEEVE_LENGTH = 5.0, 5.0  # bronze sleeve 3 x 5 x 5 mm, the knee's bearing
SLEEVE_BORE = SLEEVE_OD + 0.4  # running fit on the sleeve, allowing for printed holes coming out small
KNEE_GAP = 0.3  # axial play between the shank and the thigh's knee hub (the sleeve is this much longer)

# --- Leg geometry (leg plane y = 0 in the thigh and shank frames, at body y = cad_l1.LEG_Y) ------------------------
CRANK = L.CRANK  # crank radius, both ends of the parallelogram
CRANK_PHASE = 0.0  # direction of both cranks at zero knee angle, from the thigh's +z (rad)
BALL_Y = 7.0  # plane of the ball centres (outside the knee screw's head, which is flush at y = SHANK_Y1)
BALL_HEIGHT = 4.0  # ball centre above the face the stud is screwed to; check on the real ball studs
BALL_LINK_R, BALL_LINK_W = 3.5, 6.0  # socket of an M2 ball link: radius around the ball, width along the stud
BALL_LINK_REACH = 9.0  # ball centre to the end of the link's threaded neck; check on the real ball links
ROD_SIDE = 5.6  # printed rod's square section (2 mm of wall around the pilot holes)
ROD_THREAD_DEPTH = 7.0  # M2 pilot holes in the rod's ends, for threaded studs into the ball links
SHANK_Y0, SHANK_Y1 = -4.0, 3.0  # the shank's faces (it is printed lying on the y = SHANK_Y0 face)
KNEE_HUB_R = 7.5  # shank's knee hub
THIGH_HUB_R = 7.0  # thigh's knee hub
THIGH_HUB_Y0 = -13.3  # thigh's knee hub from here to SHANK_Y0 - KNEE_GAP
SHANK_W = 9.0  # shank link, fore and aft (its thickness across is SHANK_Y1 - SHANK_Y0)
HORN_PLATE_R = L.HORN_R + 1.0  # discs screwed to a horn
WALL = L.WALL  # end caps' wall (cad_l1); every other wall is at least MIN_WALL
MIN_WALL = 2.0

# --- Foot: Omron SS-5GLD (check with calipers on the real switch) ----------------------------------------------------
SWITCH_LENGTH, SWITCH_THICKNESS, SWITCH_HEIGHT = 12.8, 5.8, 6.5  # along the lever, across, back to actuator face
SWITCH_CLEARANCE = 0.2
SOLE_GAP = 5.2  # from the actuator face to the bottom of the sphere, for the lever (keeps 2 mm of wall at the
# pocket's corners); check on the real switch that the lever's tip stands about 1 mm out of the sphere when free
LEVER_SLOT_WIDTH = 4.0
WIRE_HOLE = 3.0
FOOT_R = L.FOOT_R
FOOT_SEAT = 10.0  # cylindrical seat for the rubber cap, from the sphere's equator up
CAP_GROOVE = (1.5, 0.6)  # groove in the seat (height, depth) for the cap's rim or a cable tie around it
FOOT_SOCKET_DEPTH = 7.0  # the shank's end goes this deep into the foot
FOOT_FIT = 0.2  # each side, between the shank's end and the foot's socket

TIE_SLOT = (1.8, 3.0)  # cable-tie slots, in pairs: width along the part, length across

KNEE_Z = -L.THIGH  # knee pin in the thigh frame
KNEE_SERVO_Z = -L.KNEE_DOWN  # knee servo shaft in the thigh frame
ROD_LENGTH = L.THIGH - L.KNEE_DOWN  # between the ball centres (the parallelogram's long side)


# --- Helpers -----------------------------------------------------------------------------------------------------

def screw_cut(d: float, depth: float, csk: float | None = None) -> Solid:
    """A hole of diameter d from a face at z = 0 down to z = -depth (with 0.2 mm overlap at both ends), and a 90-degree
    countersink of diameter csk at the face."""
    cut = Pos(0, 0, -depth / 2) * Cylinder(d / 2, depth + 0.4)
    if csk:
        h = (csk - d) / 2
        cut += Pos(0, 0, -h / 2 + 0.1) * Cone(d / 2, csk / 2 + 0.1, h + 0.2)
    return cut


def hex_prism(across_flats: float, height: float) -> Solid:
    """A hexagonal prism along z, centred on the origin, with two flats normal to x."""
    big = 4 * across_flats
    out = Box(across_flats, big, height)
    for a in (60, 120):
        out = out & (Rot(0, 0, a) * Box(across_flats, big, height))
    return out


def horn_cut(thickness: float, csk: bool = True) -> Solid:
    """The horn screw pattern through a plate on a horn, in the horn's frame (shaft along z, horn face at z = 0, plate
    from z = 0 to z = thickness): four screw holes, countersunk on the far face, and the centre hole."""
    cut = Pos(0, 0, thickness / 2) * Cylinder(HORN_CENTRE_HOLE / 2, thickness + 0.4)
    for dx in (-HORN_HOLE_OFFSET, HORN_HOLE_OFFSET):
        for dy in (-HORN_HOLE_OFFSET, HORN_HOLE_OFFSET):
            cut += Pos(dx, dy, thickness) * screw_cut(HORN_SCREW_HOLE, thickness, HORN_SCREW_CSK if csk else None)
    return cut


def tie_slots(centre: tuple[float, float, float], pitch: float, depth: float) -> Solid:
    """A pair of cable-tie slots through a plate normal to y, `pitch` apart along x."""
    width, length = TIE_SLOT
    x, y, z = centre
    slot = Box(length, depth + 0.4, width)
    return Pos(x - pitch / 2, y, z) * slot + Pos(x + pitch / 2, y, z) * slot


def along_y(r: float, y0: float, y1: float, x: float = 0.0, z: float = 0.0) -> Solid:
    """A cylinder of radius r along y from y0 to y1, its axis at (x, z)."""
    return Pos(x, (y0 + y1) / 2, z) * (Rot(90, 0, 0) * Cylinder(r, abs(y1 - y0)))


def mass(solid: Solid) -> float:
    return solid.volume * L.DENSITY


# --- Servo ---------------------------------------------------------------------------------------------------------

def servo() -> Solid:
    """A servo's space in its own frame, with the raised plate and the cable connectors on its bottom face."""
    s = L.servo()
    for (x0, x1), (y0, y1), (z0, z1) in (BOTTOM_PLATE, BOTTOM_CONNECTORS):
        s += Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)
    return s


def end_cap_cuts() -> Solid:
    """What is cut from cad_l1.end_cap in the servo frame: the four frame screw holes, countersunk on the outside, and
    the notch in the bottom plate for the raised plate on the servo's bottom face."""
    outer = SERVO_FACE + WALL  # the cap's outer faces (cad_l1.end_cap)
    cut = None
    for x, y in TOP_FRAME_HOLES:
        c = Pos(x, y, outer) * screw_cut(M2_HOLE, WALL, M2_CSK)
        cut = c if cut is None else cut + c
    for x, y in BOTTOM_FRAME_HOLES:
        cut += Pos(x, y, -outer) * Rot(180, 0, 0) * screw_cut(M2_HOLE, WALL, M2_CSK)
    (x0, x1), (y0, y1), _ = BOTTOM_PLATE
    x0, y0, y1 = x0 - 0.3, y0 - 0.3, y1 + 0.3
    cut += Pos((x0 - 9.0) / 2, 0, -outer + WALL / 2) * Box(-9.0 - x0, y1 - y0, WALL + 1.0)
    return cut


def end_cap() -> Solid:
    """cad_l1's end cap with its screw holes."""
    return L.end_cap() - end_cap_cuts()


# --- Torso and its front-left cradle -------------------------------------------------------------------------------

def _mirrors(s: Solid) -> list[Solid]:
    return [s, mirror(s, Plane.XZ), mirror(s, Plane.YZ), mirror(mirror(s, Plane.XZ), Plane.YZ)]


def torso() -> Solid:
    """cad_l1's torso with the screw holes and notches in its four cradles (end caps of the abduction servos)."""
    h = L.SERVO[1] / 2 + 0.4 + 2.0  # the floor's and deck's outer faces (cad_l1.torso); the cradles stand 0.1 mm out
    out = L.torso() & Box(400, 400, 2 * h)
    for c in _mirrors(L.abd_servo_loc() * (end_cap_cuts() + side_window())):
        out -= c
    return out


def side_window() -> Solid:
    """In the abduction servo's frame: the opening in the torso's side wall for the servo's near (shaft) end and for
    its horn and idler, which reach into cad_l1's side wall (the servo's end face is at HIP_Y + 10.1 mm and the horn's
    rim at HIP_Y + 10, the wall from HIP_Y + 9 to HIP_Y + 11). The horn turns, so it is given 0.5 mm all round."""
    m = 0.4
    x0 = SERVO_LENGTH / 2 - 2.0
    z0, z1 = -SERVO_FACE - L.IDLER_H - 0.5, SERVO_FACE + L.HORN_H + 0.5
    return Pos(x0 + 3.0, 0, (z0 + z1) / 2) * Box(6.0, SERVO_WIDTH + 2 * m, z1 - z0)


CRADLE_BOX = ((25.0, 66.0), (-0.5, 52.0), (-16.0, 16.0))  # the part of the torso printed for the test fit


def torso_cradle() -> Solid:
    """The front-left corner of the torso, around the abduction servo's cradle, to print for a test fit."""
    (x0, x1), (y0, y1), (z0, z1) = CRADLE_BOX
    out = torso() & (Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0))
    # cad_l1's floor, deck and walls end 0.2 mm behind the cradle's front face; bring them flush, so that the piece
    # lies flat on its front face
    front, cap_front = L.X_FACE - 0.5, L.X_FACE - 0.3
    edge = out & (Pos(front - 0.1, (y0 + y1) / 2, 0) * Box(0.2, y1 - y0, z1 - z0))
    return out + Pos(cap_front - front, 0, 0) * edge


# --- Shoulder (hip frame = body frame at zero abduction) ------------------------------------------------------------

SH_PLATE_X0 = L.X_FACE + 3.8  # the plate stands off the abduction horn by a boss of this length
SH_PLATE_X1 = L.X_FACE + 6.8  # and stays 0.25 mm behind the hip flexion servo's back face
SH_WINDOW = ((28.5, 48.0), (13.5, 33.0))  # lightening window in the plate (y range, z range)


def shoulder() -> Solid:
    """A boss and a plate on the abduction horn, rising to the cap on the hip flexion servo's upper end (as in
    cad_l1.shoulder), with the abduction horn's screw pattern and a lightening window."""
    hip_y = L.HIP_Y
    boss_length = SH_PLATE_X0 - L.X_FACE
    boss = Pos((L.X_FACE + SH_PLATE_X0) / 2, hip_y, 0) * (Rot(0, 90, 0) * Cylinder(HORN_PLATE_R, boss_length))
    y0, y1 = L.HIP_Y - 14.0, L.HIP_HORN_Y - 2.0
    t = SH_PLATE_X1 - SH_PLATE_X0
    top = L.SHAFT + SERVO_LENGTH / 2 + 5.0  # the cap's end wall's outer face (cad_l1.end_cap)
    bottom = -HORN_PLATE_R - 1.0
    plate = Pos((SH_PLATE_X0 + SH_PLATE_X1) / 2, (y0 + y1) / 2, (top + bottom) / 2) * Box(t, y1 - y0, top - bottom)
    (wy0, wy1), (wz0, wz1) = SH_WINDOW
    plate -= Pos((SH_PLATE_X0 + SH_PLATE_X1) / 2, (wy0 + wy1) / 2, (wz0 + wz1) / 2) * Box(t + 1, wy1 - wy0, wz1 - wz0)
    # the cap's back wall lies inside the plate, up to the cap's top, so the plate holds the cap without cad_l1's
    # bridge (which reached 1 mm into the servo)
    cap = L.hip_servo_loc() * end_cap()
    solid = boss + plate + cap
    horn = Location(Plane(origin=(L.X_FACE, hip_y, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)))
    return solid - horn * horn_cut(SH_PLATE_X1 - L.X_FACE)


# --- Thigh (thigh frame: origin on the hip flexion shaft, y = 0 in the leg's plane) ---------------------------------

TH_HORN_Y = L.HIP_HORN_Y - L.LEG_Y  # hip flexion horn face
TH_PLATE_Y0 = TH_HORN_Y + 1.0  # the plate's inner face (cad_l1.thigh), a 1 mm boss off the horn
TH_PLATE_Y1 = TH_PLATE_Y0 + L.WALL
TH_PLATE_W = 16.0
KNEE_CAP_Y0 = L.KNEE_HORN_Y - design.SERVO_HORN * L.MM - L.LEG_Y - SERVO_FACE - WALL  # knee cap's inner face
TH_BAR = (10.0, 8.0)  # the bar under the knee servo out to the knee hub (along x, along z)
TH_WINDOWS = ((-34.0, -14.0), (-68.0, -46.0))  # lightening slots in the plate (z ranges), 8 mm wide


def thigh() -> Solid:
    """A disc on the hip flexion horn and a plate down to the knee (as cad_l1.thigh), a web joining the plate to the cap
    on the knee servo's upper end, the bar out to the knee hub with the knee screw's nut slot, and cable-tie slots."""
    disc = along_y(HORN_PLATE_R, TH_PLATE_Y0, TH_PLATE_Y1)
    z_top, z_bot = 0.0, KNEE_Z - TH_BAR[1] / 2
    plate = Pos(0, (TH_PLATE_Y0 + TH_PLATE_Y1) / 2, (z_top + z_bot) / 2) * Box(TH_PLATE_W, L.WALL, z_top - z_bot)
    cap = L.knee_servo_loc() * end_cap()
    # the web: from the plate's outer face to the knee cap's inner face, up the cap's length
    cap_z0 = -L.KNEE_DOWN + L.SHAFT + SERVO_LENGTH / 2 - 12.0
    cap_z1 = -L.KNEE_DOWN + L.SHAFT + SERVO_LENGTH / 2 + 5.0
    web = Pos(0, (TH_PLATE_Y0 + KNEE_CAP_Y0 + 0.2) / 2, (cap_z0 + cap_z1) / 2) * Box(
        SERVO_WIDTH + 2 * WALL, KNEE_CAP_Y0 + 0.2 - TH_PLATE_Y0, cap_z1 - cap_z0)
    web_low = Pos(0, (TH_PLATE_Y0 + TH_PLATE_Y1) / 2, (cap_z0 + z_top) / 2) * Box(TH_PLATE_W, L.WALL, cap_z0 - z_top)
    hub_y1 = SHANK_Y0 - KNEE_GAP
    bar = Pos(0, (TH_PLATE_Y0 + THIGH_HUB_Y0) / 2, KNEE_Z) * Box(TH_BAR[0], THIGH_HUB_Y0 - TH_PLATE_Y0, TH_BAR[1])
    hub = along_y(THIGH_HUB_R, THIGH_HUB_Y0, hub_y1, z=KNEE_Z)
    solid = disc + plate + cap + web + web_low + bar + hub
    hip_horn = Location(Plane(origin=(0, TH_HORN_Y, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0)))
    solid -= hip_horn * horn_cut(TH_PLATE_Y1 - TH_HORN_Y)  # TH_HORN_Y to TH_PLATE_Y0 is the spacer
    # knee screw: clearance through the hub and into the bar, the nut slot in the bar, open downwards
    pin = along_y(M3_HOLE / 2, THIGH_HUB_Y0 - 6.0, hub_y1 + 0.2, z=KNEE_Z)
    ny = THIGH_HUB_Y0 - 1.0 - M3_NUT_T / 2
    nut = Pos(0, ny, KNEE_Z) * (Rot(90, 0, 0) * hex_prism(M3_NUT, M3_NUT_T))
    nut += Pos(0, ny, KNEE_Z - TH_BAR[1] / 2) * Box(M3_NUT, M3_NUT_T, TH_BAR[1])
    solid -= pin + nut
    # the knee cap's bottom screws go through the web and the plate, from the plate's inner face
    solid -= L.knee_servo_loc() * knee_cap_bottom_screws()
    # lightening slots in the plate, behind the knee servo and lower down; cable-tie slots for the foot switch's wires
    for z0, z1 in TH_WINDOWS:
        solid -= Pos(0, (TH_PLATE_Y0 + TH_PLATE_Y1) / 2, (z0 + z1) / 2) * Box(TH_PLATE_W - 2 * 4.0, L.WALL + 1, z1 - z0)
    for z in (-40.0, -75.0):
        solid -= tie_slots((0, (TH_PLATE_Y0 + TH_PLATE_Y1) / 2, z), 9.0, L.WALL)
    return solid


def knee_cap_bottom_screws() -> Solid:
    """In the knee servo's frame: the bottom frame screws' holes from the thigh plate's inner face to the servo."""
    z_face = TH_PLATE_Y0 - (KNEE_CAP_Y0 + WALL + SERVO_FACE)  # the plate's inner face, in the servo's frame
    out = None
    for x, y in BOTTOM_FRAME_HOLES:
        c = Pos(x, y, z_face) * Rot(180, 0, 0) * screw_cut(M2_HOLE, -SERVO_FACE - z_face, M2_CSK)
        out = c if out is None else out + c
    return out


def thigh_spacer() -> Solid:
    """The 1 mm spacer between the hip flexion horn and the thigh, printed separately so that the thigh lies flat on
    its inner face."""
    disc = along_y(HORN_PLATE_R, TH_HORN_Y, TH_PLATE_Y0)
    hip_horn = Location(Plane(origin=(0, TH_HORN_Y, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0)))
    return disc - hip_horn * horn_cut(TH_PLATE_Y0 - TH_HORN_Y, csk=False)


# --- Crank, rod and shank -------------------------------------------------------------------------------------------

def _crank_dir(knee: float) -> tuple[float, float]:
    a = CRANK_PHASE + knee
    return sin(a), cos(a)


def _ball_stud_cut(face_y: float, back_y: float) -> Solid:
    """At the crank's tip (x = 0, z = CRANK; the crank points +z): the M2 ball stud's hole from face_y through to
    back_y and the nut pocket at back_y."""
    lo, hi = min(face_y, back_y), max(face_y, back_y)
    hole = Pos(0, (lo + hi) / 2, CRANK) * (Rot(90, 0, 0) * Cylinder(M2_HOLE / 2, hi - lo + 0.4))
    s = 1 if back_y > face_y else -1
    nut = Pos(0, back_y - s * (M2_NUT_T / 2 - 0.1), CRANK) * (Rot(90, 0, 0) * hex_prism(M2_NUT, M2_NUT_T + 0.2))
    return hole + nut


CRANK_Y0 = L.KNEE_HORN_Y - L.LEG_Y  # the knee servo's horn face, in the thigh frame
CRANK_DISC_T = 3.0
CRANK_FACE_Y = BALL_Y - BALL_HEIGHT  # the ball studs are screwed to faces in this plane, on both cranks
STUD_BOSS_R = 4.5  # 2 mm of wall around the ball stud's nut pocket


def crank() -> Solid:
    """The crank on the knee servo's horn, in its own frame (shaft along y at the origin, horn face at y = CRANK_Y0,
    pointing +z): a disc on the horn and an arm to a boss at radius CRANK whose face carries the ball stud."""
    y0, y1 = CRANK_Y0, CRANK_Y0 + CRANK_DISC_T
    disc = Pos(0, (y0 + y1) / 2, 0) * (Rot(90, 0, 0) * Cylinder(HORN_PLATE_R - 1.0, y1 - y0))
    arm = Pos(0, (y0 + y1) / 2, CRANK / 2) * Box(2 * STUD_BOSS_R, y1 - y0, CRANK)
    boss = Pos(0, (y0 + CRANK_FACE_Y) / 2, CRANK) * (Rot(90, 0, 0) * Cylinder(STUD_BOSS_R, CRANK_FACE_Y - y0))
    solid = disc + arm + boss
    horn = Location(Plane(origin=(0, y0, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0)))
    return solid - horn * horn_cut(y1 - y0) - _ball_stud_cut(CRANK_FACE_Y, y0)


def rod() -> Solid:
    """The printed rod, in its own frame (along z, the upper ball centre at the origin, the lower at -ROD_LENGTH):
    a square bar between the ends of the two ball links' necks, with M2 pilot holes in its ends."""
    z0, z1 = -ROD_LENGTH + BALL_LINK_REACH, -BALL_LINK_REACH
    bar = Pos(0, 0, (z0 + z1) / 2) * Box(ROD_SIDE, ROD_SIDE, z1 - z0)
    for z, s in ((z1, -1), (z0, 1)):
        bar -= Pos(0, 0, z + s * ROD_THREAD_DEPTH / 2) * Cylinder(M2_PILOT / 2, ROD_THREAD_DEPTH + 0.2)
    return bar


def ball_links() -> Solid:
    """The two ball links' space (socket and neck), in the rod's frame (for the overlap checks)."""
    out = None
    for zc, s in ((0.0, -1), (-ROD_LENGTH, 1)):
        socket = Pos(0, 0, zc) * (Rot(90, 0, 0) * Cylinder(BALL_LINK_R, BALL_LINK_W))
        neck = Pos(0, 0, zc + s * BALL_LINK_REACH / 2) * Cylinder(2.5, BALL_LINK_REACH)
        out = socket + neck if out is None else out + socket + neck
    return out


def shank() -> Solid:
    """In the shank frame (origin on the knee pin): the knee hub with the sleeve's bore and the screw head's
    counterbore (flush with the outer face), the link down to the foot (the switch's wires are held to it by cable
    ties around it) and the crank pointing +z (at CRANK_PHASE) with the ball stud's hole. Printed lying on its inner
    face (y = SHANK_Y0)."""
    t = SHANK_Y1 - SHANK_Y0
    yc = (SHANK_Y0 + SHANK_Y1) / 2
    hub = Pos(0, yc, 0) * (Rot(90, 0, 0) * Cylinder(KNEE_HUB_R, t))
    z_end = foot_top() - FOOT_SOCKET_DEPTH + FOOT_FIT  # the link's end, inside the foot's socket
    link = Pos(0, yc, z_end / 2) * Box(SHANK_W, t, -z_end)
    arm = Pos(0, yc, CRANK / 2) * Box(2 * STUD_BOSS_R, t, CRANK)
    tip = Pos(0, yc, CRANK) * (Rot(90, 0, 0) * Cylinder(STUD_BOSS_R, t))
    crank_part = Rot(0, degrees(CRANK_PHASE), 0) * (arm + tip - _ball_stud_cut(SHANK_Y1, SHANK_Y0))
    solid = hub + link + crank_part
    # knee: the sleeve's bore, and the counterbore for the screw head from the outer face; the sleeve stands 0.2 mm
    # proud of the counterbore's floor so that the head clamps only the sleeve
    sleeve_top = SHANK_Y0 - KNEE_GAP + SLEEVE_LENGTH
    floor = sleeve_top - 0.2
    solid -= along_y(SLEEVE_BORE / 2, SHANK_Y0 - 0.2, SHANK_Y1 + 0.2)
    solid -= along_y(KNEE_HEAD_D / 2, floor, SHANK_Y1 + 0.2)
    # the foot's screw across the link's end
    solid -= Pos(0, yc, FOOT_SCREW_Z) * (Rot(0, 90, 0) * Cylinder(M2_HOLE / 2, SHANK_W + 1))
    return solid


FOOT_SCREW_Z = -L.SHANK + FOOT_SEAT - 3.5  # the M2 screw that holds the foot, across x


def foot_top() -> float:
    return -L.SHANK + FOOT_SEAT


def switch_body() -> Solid:
    """The switch in its pocket, in the shank frame (for the overlap checks)."""
    face = -L.SHANK - FOOT_R + SOLE_GAP + SWITCH_CLEARANCE  # actuator face
    return Pos(0, 0, face + SWITCH_HEIGHT / 2) * Box(SWITCH_LENGTH, SWITCH_THICKNESS, SWITCH_HEIGHT)


def foot() -> Solid:
    """In the shank frame: the sphere of radius FOOT_R at the shank's end and a cylindrical seat above it for the
    rubber cap (with a groove), a socket for the shank's end, the switch pocket (open on the outer side, +y, so that
    the switch slides in; the rubber cap closes it), the lever slot through the bottom and the wires' hole up to the
    top. Printed upside down, on its flat top."""
    zc = -L.SHANK
    top = foot_top()
    solid = Pos(0, 0, zc) * Solid.make_sphere(FOOT_R)
    solid += Pos(0, 0, (zc + top) / 2) * Cylinder(FOOT_R, top - zc)
    solid &= Pos(0, 0, top - 50) * Box(100, 100, 100)  # the sphere's top is inside the seat; cut at the seat's top
    gz = zc + 2.5
    ring = Pos(0, 0, gz) * (Cylinder(FOOT_R + 1, CAP_GROOVE[0]) - Cylinder(FOOT_R - CAP_GROOVE[1], CAP_GROOVE[0]))
    solid -= ring
    # socket for the shank's end
    t = SHANK_Y1 - SHANK_Y0
    sd = FOOT_SOCKET_DEPTH
    socket = Box(SHANK_W + 2 * FOOT_FIT, t + 2 * FOOT_FIT, sd + 0.2)
    solid -= Pos(0, (SHANK_Y0 + SHANK_Y1) / 2, top - sd / 2 + 0.1) * socket
    solid -= Pos(0, 0, FOOT_SCREW_Z) * (Rot(0, 90, 0) * Cylinder(M2_HOLE / 2, 2 * FOOT_R + 2))
    # switch pocket, lever slot, wires
    c = SWITCH_CLEARANCE
    face = zc - FOOT_R + SOLE_GAP
    h, length, w = SWITCH_HEIGHT + 2 * c, SWITCH_LENGTH + 2 * c, SWITCH_THICKNESS + 2 * c
    y_open = FOOT_R + 1.0
    solid -= Pos(0, (y_open - w / 2) / 2, face + h / 2) * Box(length, y_open + w / 2, h)
    solid -= Pos(0, 0, face - SOLE_GAP / 2 - 0.5) * Box(length, LEVER_SLOT_WIDTH, SOLE_GAP + 1.2)
    wy = SHANK_Y1 + FOOT_FIT + MIN_WALL + WIRE_HOLE / 2
    solid -= Pos(0, wy, (face + h + top) / 2) * Cylinder(WIRE_HOLE / 2, top - face - h + 1.0)
    return solid


# --- Assembly ----------------------------------------------------------------------------------------------------

def leg_solids(abd: float = 0.0, flex: float = 0.0, knee: float = 0.0) -> dict[str, Solid]:
    """The front-left leg's parts, servos, ball links and switch, placed in the body frame at the given joint angles
    (radians), on cad_l1's placements."""
    hip = Pos(0, L.HIP_Y, 0) * Rot(degrees(abd), 0, 0) * Pos(0, -L.HIP_Y, 0)
    thigh_f = hip * Pos(L.HIP_X, L.LEG_Y, 0) * Rot(0, degrees(flex), 0)
    shank_f = thigh_f * Pos(0, 0, KNEE_Z) * Rot(0, degrees(knee), 0)
    sx, sz = _crank_dir(knee)
    upper = (CRANK * sx, BALL_Y, KNEE_SERVO_Z + CRANK * sz)
    rod_f = thigh_f * Pos(*upper)
    return {
        "shoulder": hip * shoulder(),
        "hip_servo": hip * (L.hip_servo_loc() * servo()),
        "thigh": thigh_f * (thigh() + thigh_spacer()),
        "knee_servo": thigh_f * (L.knee_servo_loc() * servo()),
        "crank": thigh_f * Pos(0, 0, KNEE_SERVO_Z) * Rot(0, degrees(knee + CRANK_PHASE), 0) * crank(),
        "rod": rod_f * (rod() + ball_links()),
        "shank": shank_f * shank(),
        "foot": shank_f * foot(),
        "switch": shank_f * switch_body(),
    }


def body_solids() -> dict[str, Solid]:
    return {"torso": torso(), "abd_servo": L.abd_servo_loc() * servo()}


# --- Print parts and export -----------------------------------------------------------------------------------------

def print_parts() -> dict[str, Solid]:
    """The printed parts of the test print, each in its own design frame."""
    return {"torso_cradle_fl": torso_cradle(), "shoulder": shoulder(), "thigh": thigh(), "thigh_spacer": thigh_spacer(),
            "crank": crank(),
            "rod": rod(), "shank": shank(), "foot": foot()}


# Rotation that puts each part's chosen flat face on the bed (z = 0), as (rx, ry, rz) in degrees applied in that order.
PRINT_ORIENTATION = {
    "torso_cradle_fl": (0, 90, 0),  # front face on the bed: floor, deck, walls and the cradle's sleeve stand up
    "shoulder": (180, 0, 0),  # upside down, on the top of the cap, the bridge and the plate
    "thigh": (90, 0, 0),  # the plate's inner face on the bed; the knee hub stands up
    "thigh_spacer": (90, 0, 0),
    "crank": (90, 0, 0),  # the face on the horn on the bed
    "rod": (0, 90, 0),  # lying on a side
    "shank": (90, 0, 0),  # lying on its inner face
    "foot": (180, 0, 0),  # upside down, on its flat top
}


def oriented(name: str, solid: Solid) -> Solid:
    """The part turned to its print orientation and moved onto the bed (lowest point at z = 0)."""
    rx, ry, rz = PRINT_ORIENTATION[name]
    s = Rot(0, 0, rz) * (Rot(0, ry, 0) * (Rot(rx, 0, 0) * solid))
    b = s.bounding_box()
    return Pos(-(b.min.X + b.max.X) / 2, -(b.min.Y + b.max.Y) / 2, -b.min.Z) * s


def export(out: Path) -> dict[str, tuple[Path, Path, float]]:
    """Write each printed part as STEP and STL, in its print orientation; returns {name: (step, stl, mass in kg)}."""
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, solid in print_parts().items():
        s = oriented(name, solid)
        step, stl = out / f"{name}.step", out / f"{name}.stl"
        export_step(s, str(step))
        export_stl(s, str(stl), tolerance=0.01, angular_tolerance=0.1)
        result[name] = (step, stl, mass(solid))
    return result
