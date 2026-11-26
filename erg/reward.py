"""Reward and fall condition for learning to walk (Issue #42), the same for the per-leg and the central form.

Written down before any training in docs/design/2026-11-26-training.md. Per control step:

    reward = W_SPEED * exp(-|v_xy - c_xy|^2 / SIGMA_SPEED^2)        tracking of forward and sideways speed
           + W_TURN  * exp(-(w_z - c_turn)^2 / SIGMA_TURN^2)          tracking of turning speed
           - W_TORQUE * mean((joint torque / stall torque)^2)          effort
           - W_ACTION_RATE * mean((joint targets - previous)^2)        smooth commands
           - W_FALL if the robot fell (and the episode ends)

v_xy is the torso's horizontal velocity in its own heading frame (forward, left), w_z its turning rate, and
c = (c_xy, c_turn) the command. The robot has fallen when the torso is lower than FALL_HEIGHT or tilts more than
FALL_TILT.
"""

import numpy as np

from erg import robot

W_SPEED = 1.0
SIGMA_SPEED = 0.1  # m/s
W_TURN = 0.5
SIGMA_TURN = 0.3  # rad/s
W_TORQUE = 0.05
W_ACTION_RATE = 0.01
W_FALL = 2.0
FALL_HEIGHT = 0.12  # m (standing: about 0.22 m)
FALL_TILT = np.radians(45.0)


def heading_velocity(data) -> tuple[np.ndarray, float]:
    """Torso velocity (forward, left) in its heading frame, and its turning rate about the vertical."""
    torso = 1  # body 0 is the world
    xmat = data.xmat[torso].reshape(3, 3)
    yaw = np.arctan2(xmat[1, 0], xmat[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = data.qvel[0], data.qvel[1]  # free joint: world-frame linear velocity
    forward = c * vx + s * vy
    left = -s * vx + c * vy
    turn = (xmat @ data.qvel[3:6])[2]  # free joint angular velocity is in the body frame
    return np.array([forward, left]), turn


def fallen(data) -> bool:
    up = data.xmat[1].reshape(3, 3)[2, 2]
    return data.xpos[1, 2] < FALL_HEIGHT or np.arccos(np.clip(up, -1.0, 1.0)) > FALL_TILT


def reward(data, command, targets, previous_targets) -> tuple[float, bool]:
    """Reward of one control step and whether the robot fell."""
    v, turn = heading_velocity(data)
    r = W_SPEED * np.exp(-np.sum((v - command[:2]) ** 2) / SIGMA_SPEED**2)
    r += W_TURN * np.exp(-((turn - command[2]) ** 2) / SIGMA_TURN**2)
    r -= W_TORQUE * np.mean((data.actuator_force / robot.SERVO_STALL_TORQUE) ** 2)
    r -= W_ACTION_RATE * np.mean((np.asarray(targets) - np.asarray(previous_targets)) ** 2)
    fell = fallen(data)
    if fell:
        r -= W_FALL
    return float(r), bool(fell)
