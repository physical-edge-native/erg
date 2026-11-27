"""Reward and fall condition for learning to walk (Issues #42, #49), the same for the per-leg and the central form.

Written down in docs/design/2026-11-26-training.md (first version before any training; widths and the air-time
term changed in #49 after the first pilot run stood still instead of walking). Per control step:

    reward = W_SPEED * exp(-|v_xy - c_xy|^2 / SIGMA_SPEED^2)        tracking of forward and sideways speed
           + W_TURN  * exp(-(w_z - c_turn)^2 / SIGMA_TURN^2)          tracking of turning speed
           - W_TORQUE * mean((joint torque / stall torque)^2)          effort
           - W_ACTION_RATE * mean((joint targets - previous)^2)        smooth commands
           + W_AIR * sum over feet touching down now of (air time - AIR_TARGET), only while moving is commanded
           - W_FALL if the robot fell (and the episode ends)
    then clipped below at zero (as in legged_gym), so that being alive is never worse than ending the episode

v_xy is the torso's horizontal velocity in its own heading frame (forward, left), w_z its turning rate, and
c = (c_xy, c_turn) the command. The robot has fallen when the torso is lower than FALL_HEIGHT or tilts more than
FALL_TILT.
"""

import numpy as np

from erg import robot

W_SPEED = 1.0
SIGMA_SPEED = 0.25  # m/s (0.1 before #49)
W_TURN = 0.5
SIGMA_TURN = 0.5  # rad/s (0.3 before #49)
W_AIR = 1.0  # 0.5 in the second pilot; the same weight as speed tracking, as in legged_gym
AIR_TARGET = 0.2  # s (0.15 in the second pilot); steps shorter than this are penalized, longer ones rewarded
MOVE_SPEED = 0.05  # m/s; below this (and below MOVE_TURN) the command asks to stand still
MOVE_TURN = 0.1  # rad/s
W_TORQUE = 0.05
W_ACTION_RATE = 0.01
W_FALL = 0.0  # 2.0 before the third pilot; a fall now only ends the episode, as in legged_gym
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


def clip_total(r: float) -> float:
    """The step's total reward clipped below at zero."""
    return max(0.0, r)


def moving_commanded(command) -> bool:
    return bool(np.hypot(command[0], command[1]) > MOVE_SPEED or abs(command[2]) > MOVE_TURN)


def air_time_reward(touchdown, air_time, command) -> float:
    """Reward for steps: for each foot that touches down now (touchdown[i]), its time in the air before
    (air_time[i], s) minus AIR_TARGET; zero when the command asks to stand still."""
    if not moving_commanded(command):
        return 0.0
    return float(W_AIR * np.sum((np.asarray(air_time) - AIR_TARGET) * np.asarray(touchdown, dtype=float)))
