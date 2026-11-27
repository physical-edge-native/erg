"""Standing up after a fall (Issue #59).

Pass criteria, in MuJoCo with the body L1 model:
- from lying on the back, on either side, and from standing (each with a random heading, a tilt of up to 0.3 rad and
  joint offsets of up to 0.1 rad, three seeds each, and the push side left and right), the robot ends standing: the
  torso's up axis within 18 degrees of vertical (its z component above 0.95) and the torso higher than 0.8 of the
  standing height;
- no joint torque goes above 0.9 of the servo's stall torque during the sequence (the redesign's target);
- the sequence starts and ends where it should (the current targets; robot.STAND).
"""

import mujoco
import numpy as np
import pytest

from erg import righting, robot

STEP = robot.CONTROL_PERIOD


def get_up(roll: float, push_side: int, seed: int):
    m = robot.load()
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(-np.pi, np.pi)
    r = roll + rng.uniform(-0.3, 0.3)
    q, qy, qr = np.zeros(4), np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), np.array([np.cos(r / 2), np.sin(r / 2), 0, 0])
    mujoco.mju_mulQuat(q, qy, qr)
    d.qpos[2], d.qpos[3:7] = 0.15, q
    d.qpos[7:] = robot.STAND + rng.uniform(-0.1, 0.1, 12)
    mujoco.mj_forward(m, d)
    for _ in range(500):  # falls and comes to rest
        d.ctrl[:] = robot.STAND
        mujoco.mj_step(m, d)
    moves = righting.sequence(robot.STAND.copy(), push_side)
    substeps = round(STEP / m.opt.timestep)
    worst = 0.0
    for k in range(round(righting.duration(moves) / STEP) + 1):
        d.ctrl[:] = righting.targets_at(moves, k * STEP)
        for _ in range(substeps):
            mujoco.mj_step(m, d)
            worst = max(worst, float(np.abs(d.actuator_force).max()))
    return d.xmat[1][8], d.xpos[1][2], worst / robot.SERVO_STALL_TORQUE


def standing_height():
    m = robot.load()
    d = mujoco.MjData(m)
    robot.reset_standing(m, d)
    for _ in range(500):
        mujoco.mj_step(m, d)
    return d.xpos[1][2]


@pytest.mark.parametrize("roll", [np.pi, np.pi / 2, -np.pi / 2, 0.0], ids=["back", "left-up", "right-up", "standing"])
@pytest.mark.parametrize("push_side", [1, -1])
def test_gets_up(roll, push_side):
    h0 = standing_height()
    for seed in range(3):
        up, height, torque = get_up(roll, push_side, seed)
        assert up > 0.95 and height > 0.8 * h0, (seed, up, height)
        assert torque <= 0.9, (seed, torque)


def test_sequence_ends():
    start = robot.STAND + 0.05
    moves = righting.sequence(start, 1)
    assert np.array_equal(righting.targets_at(moves, 0.0), start)
    assert np.array_equal(righting.targets_at(moves, righting.duration(moves) + 1.0), robot.STAND)
