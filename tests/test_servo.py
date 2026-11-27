"""The servo model (Issue #70, erg/servo.py; docs/design/2026-11-27-servo-model.md).

Pass criteria:
- with no delay and the position actuator's gain, friction and limits (ServoParams.position_actuator), the servo model
  gives the joint positions of the model's position actuators within 1e-6 rad over 2 s of the hand-made trot. On
  Linux they agree bit for bit (the same floating-point operations in the same order); the tolerance allows for a
  compiler that fuses a multiply and an add (macOS on ARM), whose rounding differences of about 1e-16 grow through
  the foot contacts. 1e-6 rad is over a thousand times finer than the servo's position resolution (2 pi / 4096,
  about 1.5e-3 rad). The same holds with very large limits (1e6 N*m) against the position actuator without its force
  limit;
- a step of a joint target given with a delay of n simulation steps leaves the simulation bit for bit unchanged for
  those n steps, and changes it after them (the command reaches the joint no earlier than the delay);
- the torque on the joint (drive torque minus the back electromotive force) never leaves the speed-dependent window,
  and in the direction of motion never exceeds stall * (1 - |speed| / no-load speed), even with a gain high enough to
  saturate the drive;
- the randomised servo is deterministic for a seed: the same seed gives the same parameters and observations bit for
  bit; the delay stays in its range.
"""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from erg import gait, robot, servo
from erg.env import Env

TOL = 1e-6  # rad; see the module's docstring


def _trot_pair(params, model_changes=None, seconds=2.0, speed=0.2):
    """Largest joint-position difference between the model's position actuators and the servo model."""
    m0 = robot.load()
    if model_changes:
        model_changes(m0)
    d0 = mujoco.MjData(m0)
    robot.reset_standing(m0, d0)
    m1, sv = servo.load(params)
    d1 = mujoco.MjData(m1)
    robot.reset_standing(m1, d1)
    sv.reset(d1, robot.STAND)
    substeps = round(robot.CONTROL_PERIOD / m0.opt.timestep)
    worst = 0.0
    for k in range(round(seconds / robot.CONTROL_PERIOD)):
        targets = gait.trot_targets(k * robot.CONTROL_PERIOD, speed)
        d0.ctrl[:] = targets
        sv.command(targets)
        for _ in range(substeps):
            mujoco.mj_step(m0, d0)
            sv.apply(d1)
            mujoco.mj_step(m1, d1)
        worst = max(worst, float(np.abs(d0.qpos - d1.qpos).max()))
    return worst


def test_no_delay_reproduces_the_position_actuator():
    assert _trot_pair(servo.ServoParams.position_actuator()) <= TOL


def test_very_large_limits_reproduce_the_unlimited_position_actuator():
    p = replace(servo.ServoParams.position_actuator(), stall_torque=1e6, no_load_speed=1e6)

    def unlimited(m):
        m.actuator_forcelimited[:] = 0

    assert _trot_pair(p, unlimited) <= TOL


@pytest.mark.parametrize("delay", [0.010, 0.030])
def test_a_delayed_step_reaches_the_joint_no_earlier_than_the_delay(delay):
    p = replace(servo.ServoParams.position_actuator(), delay=delay)
    n = round(delay / 0.002)
    runs = []
    for step in (False, True):
        m, sv = servo.load(p)
        d = mujoco.MjData(m)
        robot.reset_standing(m, d)
        sv.reset(d, robot.STAND)
        for _ in range(100):  # hold the posture for 0.2 s
            sv.apply(d)
            mujoco.mj_step(m, d)
        targets = robot.STAND.copy()
        if step:
            targets[2] += 0.3  # front-left knee
        sv.command(targets)
        states = []
        for _ in range(n + 2):
            sv.apply(d)
            mujoco.mj_step(m, d)
            states.append(np.concatenate([d.qpos, d.qvel]))
        runs.append(states)
    for k in range(n):
        assert np.array_equal(runs[0][k], runs[1][k]), k  # not yet arrived
    assert not np.array_equal(runs[0][n], runs[1][n])  # the step acts in the (n + 1)-th step


def test_torque_stays_inside_the_speed_dependent_limit():
    p = replace(servo.ServoParams(), kp=200.0, delay=0.02)  # stiff enough to saturate the drive often
    m, sv = servo.load(p)
    d = mujoco.MjData(m)
    robot.reset_standing(m, d)
    sv.reset(d, robot.STAND)
    b = servo.back_emf(p)
    saturated = 0
    for k in range(round(2.0 / robot.CONTROL_PERIOD)):
        sv.command(gait.trot_targets(k * robot.CONTROL_PERIOD, 0.23))
        for _ in range(10):
            dq = d.qvel[sv.dadr].copy()
            u = sv.apply(d)
            mujoco.mj_step(m, d)
            assert np.array_equal(d.actuator_force, u)  # MuJoCo applied the model's drive torque
            tau = u - b * dq
            lo, hi = servo.drive_window(p, dq)
            assert np.all(tau >= lo - 1e-12) and np.all(tau <= hi + 1e-12)
            moving = tau * dq > 0  # torque in the direction of motion
            limit = p.stall_torque * np.maximum(0.0, 1.0 - np.abs(dq) / p.no_load_speed)
            assert np.all(np.abs(tau[moving]) <= limit[moving] + 1e-12)
            saturated += int(np.sum(np.abs(u) >= p.stall_torque))
    assert saturated > 0  # the limit was reached, so the check means something


def _run(seed, steps=25):
    env = Env(central=True, servo=servo.ServoParams(), randomize_servo=True)
    obs = [env.reset(seed, (0.2, 0.0, 0.0))]
    for k in range(steps):
        obs.append(env.step(gait.trot_targets(k * robot.CONTROL_PERIOD, 0.2)))
    return env.servo.p, np.array(obs)


def test_randomised_servo_is_deterministic_for_a_seed():
    pa, a = _run(5)
    pb, b = _run(5)
    pc, c = _run(6)
    assert pa == pb and np.array_equal(a, b)
    assert pa != pc and not np.array_equal(a, c)
    for p in (pa, pc):
        lo, hi = servo.ServoParams().delay_range
        assert lo <= p.delay <= hi


def test_servo_option_leaves_the_rest_of_the_reset_unchanged():
    """The servo's random draws use their own stream: the command and the joint offsets drawn from the seed stay."""
    plain, with_servo = Env(central=True), Env(central=True, servo=servo.ServoParams(), randomize_servo=True)
    plain.reset(11)
    with_servo.reset(11)
    assert np.array_equal(plain.command, with_servo.command)
    assert np.array_equal(plain.data.qpos, with_servo.data.qpos)
