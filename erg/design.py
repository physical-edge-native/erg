"""Body layout L1 of the redesign (Issues #58, #59): its dimensions and masses, and the MuJoCo model made from them.

Layout L1 (docs/design/2026-11-27-redesign.md): the three servos of a leg sit at the shoulder; the knee is driven by
a push rod from the knee servo, which sits next to the hip flexion servo; the feet are spread outward of the abduction
shafts; the body is low; the battery is at the bottom centre of the torso; a head mount is at the front.

This module is the single source of the body's numbers until the printed parts are drawn (the code-CAD follows the
same numbers). Every component is a box or a cylinder of known mass at a known place; the inertia of each MuJoCo body
is summed from them. Masses of parts that are not bought yet are estimates and are marked so.

Collisions: every robot geom touches the floor; thighs, shanks and feet also touch the torso (contype 3 against the
torso's conaffinity 2), because the wide abduction range used to get up can swing a leg over the body; legs do not
touch each other, and the shoulder blocks sit next to the torso.

The push-rod knee is modelled as a knee joint with its own actuator, as before. With a parallelogram push rod the knee
servo, fixed to the shoulder, turns the shank against the shoulder, not against the thigh; driving the knee joint
directly instead adds the knee's torque to the hip flexion servo in the simulation. The hip torques found here are
therefore an upper bound.

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
HIP_Y = 0.045  # and y = +-HIP_Y
LEG_OFFSET = 0.055  # from the abduction shaft out to the leg's plane (hip flexion and knee joints)
THIGH = 0.095  # hip flexion shaft to knee
SHANK = 0.100  # knee to the centre of the foot
FOOT_RADIUS = 0.012
STAND_DEPTH = 0.140  # standing: foot centre below the hip flexion shaft

# Joint ranges (rad). Abduction wide enough to roll the body over from its side.
ABD_RANGE = (-2.6, 2.6)  # wide enough to push the body over from its back (self-righting, #59)
FLEX_RANGE = (-1.8, 1.8)
KNEE_RANGE = (-2.6, -0.2)

# --- Masses (kg) ---
PLA_TORSO = 0.080  # printed torso frame, estimate
ELECTRONICS = 0.015  # per leg: microcontroller, IMU, regulator, servo bus adapter; estimate
BATTERY = 0.120  # 3S LiPo of 1300-1500 mAh class; estimate
BATTERY_SIZE = (0.075, 0.035, 0.025)
BATTERY_X = 0.015  # forward of the centre, to balance the legs' mass, which sits behind the hips with the knees bent backwards
WIRING = 0.040  # estimate
SHOULDER = 0.015  # printed shoulder block holding the hip flexion and knee servos, per leg; estimate
THIGH_LINK = 0.010  # printed thigh, per leg; estimate
PUSH_ROD = 0.005  # rod and crank, per leg; estimate
SHANK_LINK = 0.015  # printed shank with the switch pocket, per leg; estimate
FOOT = 0.004  # foot switch and soft foot cap, per leg; estimate

TORSO_SIZE = (2 * HIP_X + 0.03, 2 * HIP_Y + 0.0, 0.05)  # collision box of the torso
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
        Part(BATTERY, (BATTERY_X, 0.0, -0.025 + BATTERY_SIZE[2] / 2), BATTERY_SIZE),
        Part(WIRING, (0.0, 0.0, 0.0), TORSO_SIZE),
    ]
    for leg in LEGS:
        sx, sy = leg_sign(leg)
        # abduction servo: shaft along x at the hip, body towards the centre of the torso
        parts.append(Part(SERVO_MASS, (sx * (HIP_X - 0.020), sy * HIP_Y, 0.0), (0.035, 0.0247, 0.0452)))
        parts.append(Part(ELECTRONICS, (sx * 0.035, sy * 0.025, 0.020), (0.04, 0.024, 0.012)))
    return parts


def hip_parts(leg: str) -> list[Part]:
    """In the hip body (turns with abduction): shoulder block, hip flexion servo and knee servo, side by side along x
    with their shafts along y, just inside the leg's plane."""
    sx, sy = leg_sign(leg)
    y = sy * (LEG_OFFSET - 0.018)
    return [
        Part(SHOULDER, (0.0, y, 0.0), (0.05, 0.03, 0.04)),
        Part(SERVO_MASS, (0.0, y, 0.0), (0.0452, 0.035, 0.0247)),  # hip flexion servo, shaft along y
        Part(SERVO_MASS, (-sx * 0.030, y, 0.012), (0.0452, 0.035, 0.0247)),  # knee servo, towards the body centre
    ]


def thigh_parts() -> list[Part]:
    return [Part(THIGH_LINK + PUSH_ROD, (0.0, 0.0, -THIGH / 2), (0.016, 0.012, THIGH))]


def shank_parts() -> list[Part]:
    return [Part(SHANK_LINK, (0.0, 0.0, -SHANK / 2), (0.014, 0.012, SHANK)),
            Part(FOOT, (0.0, 0.0, -SHANK), (0.02, 0.02, 0.02))]


def stand_angles() -> tuple[float, float]:
    """Hip flexion and knee angles that put the foot STAND_DEPTH straight below the hip flexion shaft (knee bent
    backwards)."""
    c = (STAND_DEPTH**2 - THIGH**2 - SHANK**2) / (2 * THIGH * SHANK)
    knee = -float(np.arccos(c))
    flex = -float(np.arctan2(SHANK * np.sin(knee), THIGH + SHANK * np.cos(knee)))
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
        lines.append(f'      <site name="{leg}_imu" pos="{sx * HIP_X:.4g} {sy * HIP_Y:.4g} {TORSO_SIZE[2] / 2:.4g}"/>')
    for leg in LEGS:
        sx, sy = leg_sign(leg)
        y = sy * (LEG_OFFSET - 0.018)
        lines += [
            f"      <!-- {leg} -->",
            f'      <body name="{leg}_hip" pos="{sx * HIP_X:.4g} {sy * HIP_Y:.4g} 0">',
            "        " + _inertial(hip_parts(leg)),
            f'        <joint name="{leg}_hip_abd" class="abd"/>',
            f'        <geom type="box" pos="{-sx * 0.015:.4g} {y:.4g} 0.004" size="0.04 0.0175 0.02"/>',
            f'        <body name="{leg}_thigh" pos="0 {sy * LEG_OFFSET:.4g} 0">',
            "          " + _inertial(thigh_parts()),
            f'          <joint name="{leg}_hip_flex" class="flex"/>',
            f'          <geom type="capsule" fromto="0 0 0 0 0 {-THIGH:.4g}" size="0.009" contype="3"/>',
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
