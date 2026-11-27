"""Body layout L1 of the redesign (Issues #58, #59): its dimensions and masses, and the MuJoCo model made from them.

Layout L1 (docs/design/2026-11-27-redesign.md), with the knee servo in the thigh (chosen 2026-11-27, #59): the
abduction servo lies in the torso; the hip flexion servo sits on a shoulder plate in front of the torso; the knee servo
sits inside the thigh just below the hip flexion shaft and drives the shank through a parallelogram push rod, so the
knee servo's angle is the knee angle; the feet are spread outward of the abduction shafts; the body is low; the battery
lies across the bottom of the torso; a head mount is at the front.

This module is the single source of the body's numbers until the printed parts are drawn (the code-CAD follows the
same numbers). Every component is a box or a cylinder of known mass at a known place; the inertia of each MuJoCo body
is summed from them. Masses of parts that are not bought yet are estimates and are marked so.

Collisions: every robot geom touches the floor; thighs, shanks and feet also touch the torso (contype 3 against the
torso's conaffinity 2), because the wide abduction range used to get up can swing a leg over the body; legs do not
touch each other, and the shoulder blocks sit next to the torso.

The push-rod knee is modelled as a knee joint with its own actuator: the knee servo is fixed in the thigh and the
parallelogram makes the shank turn against the thigh by the servo's angle, which is what a knee joint actuator does.

Units: metres, kilograms. Body frame: x forward, y left, z up; origin on the abduction shafts' plane.
"""

from dataclasses import dataclass

import numpy as np

LEGS = ("FL", "FR", "HL", "HR")

# --- Servo: 12 V class serial bus servo (decided 2026-11-27). Values from the seller's listing of the Waveshare
# ST3215 (12 V, 30 kg*cm, 0.222 s per 60 degrees); unverified until checked on a real servo. Size and mass taken as
# the STS3215's (45.2 x 24.7 x 35 mm, 55 g), to be checked.
SERVO_STALL_TORQUE = 2.94  # N*m (30 kg*cm)
SERVO_NO_LOAD_SPEED = 4.72  # rad/s (0.222 s per 60 degrees)
SERVO_SIZE = (0.0452, 0.0247, 0.035)
SERVO_MASS = 0.055

# --- Geometry ---
HIP_X = 0.085  # abduction shafts at x = +-HIP_X
HIP_Y = 0.040  # and y = +-HIP_Y (the torso's sides stay inside the thighs' inner plates)
LEG_OFFSET = 0.060  # from the abduction shaft out to the leg's plane (hip flexion and knee joints)
THIGH = 0.095  # hip flexion shaft to knee
SHANK = 0.100  # knee to the centre of the foot
FOOT_RADIUS = 0.012
STAND_DEPTH = 0.140  # standing: foot centre below the hip flexion shaft
STAND_FOOT_X = -0.008  # standing: foot centre behind the hip flexion shaft, to put the support centre under the centre of mass

# Joint ranges (rad). Abduction wide enough to roll the body over from its side.
ABD_RANGE = (-2.6, 2.6)  # wide enough to push the body over from its back (self-righting, #59)
FLEX_RANGE = (-1.8, 1.8)
KNEE_RANGE = (-2.6, -0.2)

# --- Masses (kg) ---
PLA_TORSO = 0.087  # printed torso frame (layout CAD, erg/cad_l1.py)
ELECTRONICS = 0.015  # per leg: microcontroller, IMU, regulator, servo bus adapter; estimate
BATTERY = 0.120  # 3S LiPo of 1300-1500 mAh class; estimate
BATTERY_SIZE = (0.035, 0.075, 0.025)  # x, y, z: lies across the torso, between the abduction servos
BATTERY_X = 0.015  # forward of the centre, to balance the legs' mass, which sits behind the hips with the knees bent backwards
WIRING = 0.040  # estimate
SHOULDER = 0.014  # printed shoulder plate holding the hip flexion servo, per leg (layout CAD)
THIGH_LINK = 0.021  # printed thigh with the knee servo's cap, per leg (layout CAD)
PUSH_ROD = 0.003  # rod and crank, per leg (layout CAD)
SHANK_LINK = 0.016  # printed shank with the foot, per leg (layout CAD)
FOOT = 0.004  # foot switch and soft foot cap, per leg; estimate

# --- Servo placement (layout L1 with the knee servo in the thigh, chosen 2026-11-27, #59) ---
# All in metres, for the front-left leg; other legs are mirrored in x and y.
# The abduction servo lies in the torso, shaft along x, its horn face at X_FACE; everything that turns with abduction
# is in front of X_FACE, so that it clears the torso over the whole abduction range.
SERVO_HORN = 0.0187  # servo centre to horn face, along the shaft
SERVO_SHAFT = 0.0125  # servo centre to shaft, along its length
BOSS = 0.003  # spacer between a horn and the part it drives
PLATE = 0.004  # printed plate thickness
X_FACE = 0.0656  # abduction horn face (HIP_X - 19.4 mm: room for the shoulder plate in front of it)
HIP_HORN_Y = 0.0524  # hip flexion horn face (thigh's inner plate starts here)
KNEE_HORN_Y = 0.094  # knee servo horn face (crank plane just outside)
KNEE_SERVO_DOWN = 0.018  # knee servo shaft below the hip flexion shaft, along the thigh


TORSO_SIZE = (2 * X_FACE, 2 * 0.051, 0.033)  # collision box of the torso
TORSO_Z = 0.0  # centre of the torso box


@dataclass
class Part:
    mass: float
    pos: tuple  # centre of mass in the body frame
    size: tuple  # box size (x, y, z)


def box_inertia(p: Part) -> np.ndarray:
    a, b, c = p.size
    return p.mass / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


def combine(parts: list[Part]) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, centre of mass and inertia (about the centre of mass) of boxes."""
    m = sum(p.mass for p in parts)
    c = sum(p.mass * np.array(p.pos) for p in parts) / m
    inertia = np.zeros((3, 3))
    for p in parts:
        d = np.array(p.pos) - c
        inertia += box_inertia(p) + p.mass * (d @ d * np.eye(3) - np.outer(d, d))
    return m, c, inertia


def leg_sign(leg: str) -> tuple[int, int]:
    return (1 if leg[0] == "F" else -1), (1 if leg[1] == "L" else -1)


def torso_parts() -> list[Part]:
    parts = [
        Part(PLA_TORSO, (0.0, 0.0, 0.0), TORSO_SIZE),
        Part(BATTERY, (BATTERY_X, 0.0, -0.025 + BATTERY_SIZE[2] / 2), BATTERY_SIZE),  # lies across the torso
        Part(WIRING, (0.0, 0.0, 0.0), TORSO_SIZE),
    ]
    for leg in LEGS:
        sx, sy = leg_sign(leg)
        # abduction servo: shaft along x with its horn face at X_FACE, body inwards along y
        parts.append(Part(SERVO_MASS, (sx * (X_FACE - SERVO_HORN), sy * (HIP_Y - SERVO_SHAFT), 0.0), (0.035, 0.0452, 0.0247)))
        parts.append(Part(ELECTRONICS, (sx * 0.035, sy * 0.025, 0.031), (0.04, 0.024, 0.012)))  # on the deck
    return parts


def hip_parts(leg: str) -> list[Part]:
    """In the hip body (turns with abduction; origin on the abduction shaft at HIP_X): the shoulder plate on the
    abduction horn and the hip flexion servo, shaft along y, body above its shaft."""
    sx, sy = leg_sign(leg)
    hip_servo_y = HIP_HORN_Y - SERVO_HORN - HIP_Y  # relative to the hip body
    return [
        Part(SHOULDER, (sx * (X_FACE + BOSS + PLATE / 2 - HIP_X), sy * hip_servo_y / 2, 0.01), (PLATE, 0.03, 0.05)),
        Part(SERVO_MASS, (0.0, sy * hip_servo_y, SERVO_SHAFT), (0.0247, 0.035, 0.0452)),
    ]


def thigh_parts(leg: str) -> list[Part]:
    """In the thigh body (origin on the hip flexion shaft, in the leg's plane): the thigh bracket around the knee
    servo, whose shaft is KNEE_SERVO_DOWN below the hip flexion shaft and whose body points up along the thigh, past the
    hip flexion shaft (the two servos are in different planes, so they never meet); the push rod to the shank."""
    sx, sy = leg_sign(leg)
    knee_servo_y = KNEE_HORN_Y - SERVO_HORN - (HIP_Y + LEG_OFFSET)
    z = -(KNEE_SERVO_DOWN - SERVO_SHAFT)  # body upwards from its shaft, close to the hip flexion shaft
    return [
        Part(THIGH_LINK, (0.0, sy * knee_servo_y / 2, -THIGH / 2), (0.03, 0.045, THIGH)),
        Part(SERVO_MASS, (0.0, sy * knee_servo_y, z), (0.0247, 0.035, 0.0452)),
        Part(PUSH_ROD, (0.0, sy * 0.006, -(KNEE_SERVO_DOWN + THIGH) / 2), (0.006, 0.004, THIGH - KNEE_SERVO_DOWN)),
    ]


def shank_parts() -> list[Part]:
    return [Part(SHANK_LINK, (0.0, 0.0, -SHANK / 2), (0.014, 0.012, SHANK)),
            Part(FOOT, (0.0, 0.0, -SHANK), (0.02, 0.02, 0.02))]


def stand_angles() -> tuple[float, float]:
    """Hip flexion and knee angles that put the foot STAND_DEPTH below and STAND_FOOT_X ahead of the hip flexion shaft
    (knee bent backwards)."""
    x, z = STAND_FOOT_X, -STAND_DEPTH
    c = (x * x + z * z - THIGH**2 - SHANK**2) / (2 * THIGH * SHANK)
    knee = -float(np.arccos(c))
    flex = float(np.arctan2(-x, -z)) - float(np.arctan2(SHANK * np.sin(knee), THIGH + SHANK * np.cos(knee)))
    return flex, knee


def _inertial(parts: list[Part], mirror_y: bool = False) -> str:
    m, c, inertia = combine(parts)
    f = lambda v: f"{v:.6g}"  # noqa: E731
    return (f'<inertial pos="{f(c[0])} {f(c[1])} {f(c[2])}" mass="{f(m)}" fullinertia="{f(inertia[0, 0])} '
            f'{f(inertia[1, 1])} {f(inertia[2, 2])} {f(inertia[0, 1])} {f(inertia[0, 2])} {f(inertia[1, 2])}"/>')


def mjcf() -> str:
    """The MuJoCo model of the body (models/quadruped.xml is this text; tests check it)."""
    flex, knee = stand_angles()
    torso_z = STAND_DEPTH + FOOT_RADIUS + 0.005
    lines = [
        "<!-- Generated by scripts/make_model.py from erg/design.py (body layout L1, Issue #59); do not edit. -->",
        '<mujoco model="erg_quadruped">',
        '  <compiler angle="radian" autolimits="true"/>',
        '  <option timestep="0.002" integrator="implicitfast"/>',
        "  <default>",
        f'    <joint damping="{SERVO_STALL_TORQUE / SERVO_NO_LOAD_SPEED:.4g}" armature="0.01" frictionloss="0.05"/>',
        f'    <position kp="10" forcerange="{-SERVO_STALL_TORQUE} {SERVO_STALL_TORQUE}"/>',
        '    <geom contype="1" conaffinity="0" friction="0.8 0.02 0.0001"/>',
        f'    <default class="abd"><joint axis="1 0 0" range="{ABD_RANGE[0]} {ABD_RANGE[1]}"/>'
        f'<position ctrlrange="{ABD_RANGE[0]} {ABD_RANGE[1]}"/></default>',
        f'    <default class="flex"><joint axis="0 1 0" range="{FLEX_RANGE[0]} {FLEX_RANGE[1]}"/>'
        f'<position ctrlrange="{FLEX_RANGE[0]} {FLEX_RANGE[1]}"/></default>',
        f'    <default class="knee"><joint axis="0 1 0" range="{KNEE_RANGE[0]} {KNEE_RANGE[1]}"/>'
        f'<position ctrlrange="{KNEE_RANGE[0]} {KNEE_RANGE[1]}"/></default>',
        "  </default>",
        "  <worldbody>",
        '    <geom name="floor" type="plane" size="5 5 0.1" contype="0" conaffinity="1"/>',
        f'    <body name="torso" pos="0 0 {torso_z:.4g}">',
        '      <freejoint name="root"/>',
        "      " + _inertial(torso_parts()),
        f'      <geom name="torso" type="box" conaffinity="2" pos="0 0 {TORSO_Z}" size="{TORSO_SIZE[0] / 2:.4g} '
        f'{TORSO_SIZE[1] / 2:.4g} {TORSO_SIZE[2] / 2:.4g}"/>',
    ]
    for leg in LEGS:
        sx, sy = leg_sign(leg)
        lines.append(f'      <site name="{leg}_imu" pos="{sx * 0.05:.4g} {sy * 0.03:.4g} {TORSO_SIZE[2] / 2:.4g}"/>')
    for leg in LEGS:
        sx, sy = leg_sign(leg)
        y = sy * (LEG_OFFSET - 0.018)
        lines += [
            f"      <!-- {leg} -->",
            f'      <body name="{leg}_hip" pos="{sx * HIP_X:.4g} {sy * HIP_Y:.4g} 0">',
            "        " + _inertial(hip_parts(leg)),
            f'        <joint name="{leg}_hip_abd" class="abd"/>',
            f'        <geom type="box" pos="0 {sy * (HIP_HORN_Y - SERVO_HORN - HIP_Y):.4g} {SERVO_SHAFT}" '
            f'size="0.01235 0.0175 0.0226"/>',  # hip flexion servo
            f'        <body name="{leg}_thigh" pos="0 {sy * LEG_OFFSET:.4g} 0">',
            "          " + _inertial(thigh_parts(leg)),
            f'          <joint name="{leg}_hip_flex" class="flex"/>',
            f'          <geom type="capsule" fromto="0 0 0 0 0 {-THIGH:.4g}" size="0.009" contype="3"/>',
            f'          <geom type="box" pos="0 {sy * (KNEE_HORN_Y - SERVO_HORN - HIP_Y - LEG_OFFSET):.4g} '
            f'{-(KNEE_SERVO_DOWN - SERVO_SHAFT):.4g}" size="0.01235 0.0175 0.0226" contype="3"/>',  # knee servo
            f'          <geom type="box" pos="0 {sy * (HIP_HORN_Y + 0.001 + 0.00125 - HIP_Y - LEG_OFFSET):.4g} '
            f'{-THIGH / 2 + 0.006:.4g}" size="0.008 0.00125 {THIGH / 2 + 0.006:.4g}" contype="3"/>',  # thigh's inner plate
            f'          <body name="{leg}_shank" pos="0 0 {-THIGH:.4g}">',
            "            " + _inertial(shank_parts()),
            f'            <joint name="{leg}_knee" class="knee"/>',
            f'            <geom type="capsule" fromto="0 0 0 0 0 {-(SHANK - FOOT_RADIUS):.4g}" size="0.007" contype="3"/>',
            f'            <geom name="{leg}_foot" type="sphere" pos="0 0 {-SHANK:.4g}" size="{FOOT_RADIUS}" contype="3"/>',
            f'            <site name="{leg}_foot" pos="0 0 {-SHANK:.4g}" size="{FOOT_RADIUS + 0.002:.4g}"/>',
            "          </body>",
            "        </body>",
            "      </body>",
        ]
    lines += ["    </body>", "  </worldbody>", "  <actuator>"]
    for leg in LEGS:
        for j, cls in (("hip_abd", "abd"), ("hip_flex", "flex"), ("knee", "knee")):
            lines.append(f'    <position name="{leg}_{j}" joint="{leg}_{j}" class="{cls}"/>')
    lines += ["  </actuator>", "  <sensor>"]
    for leg in LEGS:
        for kind, tag in (("pos", "jointpos"), ("vel", "jointvel")):
            for j in ("hip_abd", "hip_flex", "knee"):
                lines.append(f'    <{tag} name="{leg}_{j}_{kind}" joint="{leg}_{j}"/>')
        for j in ("hip_abd", "hip_flex", "knee"):
            lines.append(f'    <actuatorfrc name="{leg}_{j}_load" actuator="{leg}_{j}"/>')
        lines += [f'    <accelerometer name="{leg}_acc" site="{leg}_imu"/>',
                  f'    <gyro name="{leg}_gyro" site="{leg}_imu"/>',
                  f'    <touch name="{leg}_touch" site="{leg}_foot"/>']
    lines += ["  </sensor>", "</mujoco>"]
    return "\n".join(lines) + "\n"
