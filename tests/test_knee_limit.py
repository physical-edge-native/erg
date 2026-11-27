"""The knee torque limit near the straight end of the knee range (Issue #81, erg/knee_limit.py).

Pass criteria:
- at the limited torque, and at the torque the register's value gives, the push rod's force stays under its
  allowable load (Euler buckling load of the carbon tube / safety factor 2) at every knee angle of the range
  (a grid of 0.5 mrad);
- the limit is the stall torque wherever the simulation skips computing it (FULL_BAND), and below it only near the
  straight end of the range (from about -0.23 rad);
- the servo model applies the limit to the knees: a knee held against a target beyond the straight limit is driven
  with at most the limit; with the limit off (ServoParams.position_actuator) nothing changes;
- the stand-up sequence (erg.righting) still succeeds in simulation with the servo model (erg.servo) and the limit,
  from the back, either side and standing, both push sides (the criteria of tests/test_righting.py, one seed each).
The firmware's C version is checked against this one bit for bit in tests/test_firmware_host.py.
"""

import mujoco
import numpy as np
import pytest

from erg import design, knee_limit, righting, robot, servo
from test_righting import standing_height


def test_rod_force_stays_under_the_allowable_load():
    for q in np.arange(design.KNEE_RANGE[0], design.KNEE_RANGE[1] + 1e-9, 0.0005):
        q = float(q)
        t = knee_limit.torque_limit(q)
        assert knee_limit.rod_force(q, t) <= knee_limit.ALLOWABLE * (1 + 1e-12), q
        reg = knee_limit.register_value(q) / 1000.0 * knee_limit.STALL
        assert reg <= t and knee_limit.rod_force(q, reg) <= knee_limit.ALLOWABLE * (1 + 1e-12), q


def test_limit_is_the_stall_torque_except_near_the_straight_end():
    lo, hi = knee_limit.FULL_BAND
    for q in np.linspace(lo, hi, 2001):
        assert knee_limit.torque_limit(float(q)) == knee_limit.STALL
    limited = [q for q in np.arange(design.KNEE_RANGE[0], design.KNEE_RANGE[1] + 1e-9, 0.0005)
               if knee_limit.torque_limit(float(q)) < knee_limit.STALL]
    assert min(limited) > -0.24
    assert knee_limit.register_value(design.KNEE_RANGE[1]) == 869


def _held_knee_torque(params):
    m, sv = servo.load(params)
    d = mujoco.MjData(m)
    robot.reset_standing(m, d)
    q = robot.STAND.copy()
    q[2] = -0.21  # front-left knee near the straight limit
    d.qpos[7:] = q
    mujoco.mj_forward(m, d)
    sv.reset(d, q)
    targets = q.copy()
    targets[2] = -0.2 + 0.5  # beyond the limit: the target range clips it to -0.2, the error drives the knee hard
    sv.command(targets)
    u = sv.apply(d)
    return abs(float(u[2])), float(d.qpos[7 + 2])


def test_servo_model_applies_the_limit():
    p = servo.ServoParams(kp=1000.0, delay=0.0)
    u, q = _held_knee_torque(p)
    assert u == pytest.approx(knee_limit.torque_limit(q)) and u < knee_limit.STALL
    from dataclasses import replace
    u_off, _ = _held_knee_torque(replace(p, knee_limit=False))
    assert u_off == knee_limit.STALL
    assert not servo.ServoParams.position_actuator().knee_limit


def get_up(roll, push_side, seed=0):
    m, sv = servo.load()
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(-np.pi, np.pi)
    r = roll + rng.uniform(-0.3, 0.3)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
                       np.array([np.cos(r / 2), np.sin(r / 2), 0, 0]))
    d.qpos[2], d.qpos[3:7] = 0.15, quat
    d.qpos[7:] = robot.STAND + rng.uniform(-0.1, 0.1, 12)
    mujoco.mj_forward(m, d)
    sv.reset(d, robot.STAND)
    for _ in range(500):  # falls and comes to rest
        sv.apply(d)
        mujoco.mj_step(m, d)
    moves = righting.sequence(robot.STAND.copy(), push_side)
    substeps = round(robot.CONTROL_PERIOD / m.opt.timestep)
    for k in range(round(righting.duration(moves) / robot.CONTROL_PERIOD) + 1):
        sv.command(righting.targets_at(moves, k * robot.CONTROL_PERIOD))
        for _ in range(substeps):
            sv.apply(d)
            mujoco.mj_step(m, d)
    return d.xmat[1][8], d.xpos[1][2]


@pytest.mark.parametrize("roll", [np.pi, np.pi / 2, -np.pi / 2, 0.0], ids=["back", "left-up", "right-up", "standing"])
@pytest.mark.parametrize("push_side", [1, -1])
def test_stand_up_with_the_servo_model_and_the_limit(roll, push_side):
    assert servo.ServoParams().knee_limit
    up, height = get_up(roll, push_side)
    assert up > 0.95 and height > 0.8 * standing_height(), (up, height)
