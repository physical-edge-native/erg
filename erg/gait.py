"""A fixed trot gait (Issue #32): joint targets from the time and a forward speed, with no training.

A fallback to show the robot walking (December 2026 demonstration); it is not part of the comparison of split and
central control. Diagonal legs move together (front-left with hind-right, front-right with hind-left), half a
period apart. Each foot follows a path in its leg's vertical plane, relative to the hip flexion shaft:

- on the ground (the first half of its period), it moves straight back by one stride, at the walking speed;
- in the air (the second half), it moves forward again along a half-sine arc of height STEP_HEIGHT.

The joint angles come from the two-link inverse kinematics of thigh and shank (lengths from erg.design, which the
MuJoCo model follows). Abduction stays at zero. The trigonometric functions are erg.exactmath's (since #54), so that the
leg firmware computes the same targets bit for bit.
"""

from erg.exactmath import PI as pi
from erg.exactmath import acos, atan2, cos, sin

import numpy as np

from erg import design, robot

THIGH = design.THIGH  # m
SHANK = design.SHANK  # m
PERIOD = 0.45  # s, one full gait cycle (0.35 s for the first body, #32; 0.45 s for body L1, #59, within the servo's speed)
STEP_HEIGHT = 0.012  # m, foot lift in the air (0.015 m for the first body; lower for body L1, #59, within the servo's speed)
# Height of the foot centre below the hip flexion shaft: that of the standing posture (robot.STAND).
STAND_DEPTH = THIGH * cos(robot.STAND[1]) + SHANK * cos(robot.STAND[1] + robot.STAND[2])
# Phase of each leg in the order of robot.LEGS (FL, FR, HL, HR): diagonal pairs move together.
PHASE = {"FL": 0.0, "FR": 0.5, "HL": 0.5, "HR": 0.0}


def leg_angles(x: float, z: float) -> tuple[float, float]:
    """Hip flexion and knee angles that put the foot centre at (x forward, z up) from the hip flexion shaft.
    Knee bent backwards (negative angle), as in the standing posture."""
    r2 = x * x + z * z
    c = (r2 - THIGH * THIGH - SHANK * SHANK) / (2 * THIGH * SHANK)
    knee = -acos(max(-1.0, min(1.0, c)))
    # A foot at angle phi from straight down (positive backwards) with the knee bent by `knee`.
    phi = atan2(-x, -z)
    flex = phi - atan2(SHANK * sin(knee), THIGH + SHANK * cos(knee))
    return flex, knee


def foot_path(s: float, stride: float) -> tuple[float, float]:
    """Foot position at phase s in [0, 1) of its cycle, for the given stride (m)."""
    if s < 0.5:  # on the ground: from +stride/2 back to -stride/2
        u = s / 0.5
        return stride / 2 - stride * u, -STAND_DEPTH
    u = (s - 0.5) / 0.5  # in the air: forward again, lifted
    return -stride / 2 + stride * u, -STAND_DEPTH + STEP_HEIGHT * sin(pi * u)


def leg_targets(s: float, stride: float = 0.0) -> np.ndarray:
    """One leg's 3 joint targets (rad; abduction, hip flexion, knee) at phase s in [0, 1) of its cycle.
    With stride 0 the foot steps in place: the base gait of the learned controllers (Issue #51)."""
    flex, knee = leg_angles(*foot_path(s % 1.0, stride))
    return np.array([0.0, flex, knee])


def trot_targets(t: float, speed: float) -> np.ndarray:
    """The 12 joint targets (rad) at time t (s) for a forward speed (m/s), in the order of robot.JOINT_NAMES."""
    stride = speed * PERIOD / 2  # the foot is on the ground half of the period, moving back at `speed`
    return np.concatenate([leg_targets((t / PERIOD + PHASE[leg]) % 1.0, stride) for leg in robot.LEGS])
