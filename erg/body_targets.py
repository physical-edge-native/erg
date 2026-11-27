"""Body targets of the redesign (Issue #58), computed from the MuJoCo model.

docs/design/2026-11-27-redesign.md sets targets that the robot's body must meet in simulation before parts are
ordered. This module computes them for any model that follows erg's names (robot.LEGS, the `{leg}_foot` geoms, the
`{leg}_hip` and `{leg}_thigh` bodies):

- total mass;
- standing: centre-of-mass height above the ground, the support rectangle of the four feet, the centre of mass's
  offset from the support centre, and its height divided by half of the support width (the static sideways tipping
  angle is atan(half width / height));
- leg mass below the hip flexion joint (the thigh and shank bodies), as a fraction of the total;
- the hand-made trot of erg.gait at a forward speed: joint torque relative to the stall torque, and the joint speed
  asked of the servos (the change of the joint targets per control period) relative to the no-load speed; for each
  joint its 95th percentile over the walk after a settling time, the largest of these over the joints, and the
  maximum torque;
- standing up from a crouch (body resting low, legs folded) to the standing posture in STAND_UP_SECONDS: the largest
  joint torque relative to the servo limit. This is the last part of getting up after a fall; rolling over from
  the side is added with the new body (Issue #59).

All quantities are deterministic: the simulation starts from the standing posture with no noise.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from erg import gait, robot

SETTLE_STEPS = 500  # 1 s at 2 ms, standing still, before measuring the standing posture
TROT_SPEED = 0.2  # m/s, forward
TROT_SECONDS = 10.0
TROT_SKIP_SECONDS = 2.0  # left out of the statistics: the start of the walk
CROUCH = np.array([0.0, 1.3, -2.4] * len(robot.LEGS))  # legs folded under the body
STAND_UP_SECONDS = 2.0


@dataclass
class Targets:
    mass: float  # kg
    com_height: float  # m above the ground, standing
    support_length: float  # m, front feet to hind feet
    support_width: float  # m, left feet to right feet
    com_offset_x: float  # m, from the support centre, forward positive
    com_offset_y: float  # m, left positive
    height_over_half_width: float
    tipping_angle_deg: float  # static sideways tipping angle
    leg_mass_fraction: float  # thigh and shank bodies of all legs / total mass
    trot_torque_p95: float  # of the stall torque: the largest over joints of each joint's 95th percentile in time
    trot_torque_max: float
    trot_speed_p95: float  # commanded joint speed (target change per control period) / no-load speed, the same way
    stand_up_torque_max: float


def _settle(m, d, targets, steps):
    for _ in range(steps):
        d.ctrl[:] = targets
        mujoco.mj_step(m, d)


def standing(m: mujoco.MjModel) -> dict:
    d = mujoco.MjData(m)
    robot.reset_standing(m, d)
    _settle(m, d, robot.STAND, SETTLE_STEPS)
    feet = np.array([d.geom(f"{leg}_foot").xpos for leg in robot.LEGS])
    radius = m.geom(f"{robot.LEGS[0]}_foot").size[0]
    ground = feet[:, 2].min() - radius
    com = d.subtree_com[1]
    centre = feet[:, :2].mean(axis=0)
    length = feet[:, 0].max() - feet[:, 0].min()
    width = feet[:, 1].max() - feet[:, 1].min()
    height = com[2] - ground
    return {
        "com_height": float(height),
        "support_length": float(length),
        "support_width": float(width),
        "com_offset_x": float(com[0] - centre[0]),
        "com_offset_y": float(com[1] - centre[1]),
        "height_over_half_width": float(height / (width / 2)),
        "tipping_angle_deg": float(np.degrees(np.arctan2(width / 2, height))),
    }


def leg_mass_fraction(m: mujoco.MjModel) -> float:
    legs = sum(m.body(f"{leg}_thigh").subtreemass[0] for leg in robot.LEGS)  # the thigh with the shank below it
    return float(legs / m.body_mass.sum())


def trot(m: mujoco.MjModel, speed: float = TROT_SPEED) -> tuple[float, float, float]:
    d = mujoco.MjData(m)
    robot.reset_standing(m, d)
    substeps = round(robot.CONTROL_PERIOD / m.opt.timestep)
    torque = []
    t = 0.0
    for k in range(round(TROT_SECONDS / robot.CONTROL_PERIOD)):
        d.ctrl[:] = gait.trot_targets(t, speed)
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        t += robot.CONTROL_PERIOD
        if t > TROT_SKIP_SECONDS:
            torque.append(np.abs(d.actuator_force) / robot.SERVO_STALL_TORQUE)
    torque = np.array(torque)
    # the joint speed asked of the servos: the change of the joint targets per control period. The measured joint
    # velocity also holds the passive vibration after each touch-down, which the servo does not have to drive.
    ts = np.arange(round(TROT_SECONDS / robot.CONTROL_PERIOD)) * robot.CONTROL_PERIOD
    targets = np.array([gait.trot_targets(x, speed) for x in ts if x > TROT_SKIP_SECONDS])
    joint_speed = np.abs(np.diff(targets, axis=0)) / robot.CONTROL_PERIOD / robot.SERVO_NO_LOAD_SPEED
    # the worst joint's 95th percentile: one joint near its limit matters even if the others are idle
    return (float(np.percentile(torque, 95, axis=0).max()), float(torque.max()),
            float(np.percentile(joint_speed, 95, axis=0).max()))


def stand_up(m: mujoco.MjModel) -> float:
    """Largest joint torque / stall torque while going from the crouch to the standing posture."""
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.12
    d.qpos[7:] = CROUCH
    mujoco.mj_forward(m, d)
    _settle(m, d, CROUCH, SETTLE_STEPS)  # rests on the ground with the legs folded
    steps = round(STAND_UP_SECONDS / m.opt.timestep)
    worst = 0.0
    for k in range(steps):
        a = (k + 1) / steps
        d.ctrl[:] = (1 - a) * CROUCH + a * robot.STAND
        mujoco.mj_step(m, d)
        worst = max(worst, float(np.max(np.abs(d.actuator_force)) / robot.SERVO_STALL_TORQUE))
    return worst


def compute(m: mujoco.MjModel | None = None) -> Targets:
    m = m if m is not None else robot.load()
    s = standing(m)
    p95, tmax, s95 = trot(m)
    return Targets(mass=float(m.body_mass.sum()), leg_mass_fraction=leg_mass_fraction(m), trot_torque_p95=p95,
                   trot_torque_max=tmax, trot_speed_p95=s95, stand_up_torque_max=stand_up(m), **s)
