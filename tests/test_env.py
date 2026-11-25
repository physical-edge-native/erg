"""The environment with one controller per leg (Issue #11).

Pass criteria:
- the same seed, command and actions give the same observations bit for bit on repeated runs;
  a different seed gives different observations;
- every observation value equals the MuJoCo state it stands for: joint positions and speeds equal qpos and qvel
  bit for bit, loads equal actuator force / stall torque bit for bit, the contact flag follows the touch sensor;
- the IMU reads like an IMU: the gyroscope equals the torso's angular velocity in its own frame (to 1e-12 rad/s;
  computed by MuJoCo along another path, so only rounding may differ) and, standing still, the accelerometer
  reads gravity (+9.81 m/s^2 on z, within 0.05 m/s^2, the small remaining sway);
- actions outside the joint ranges are clipped to the ranges;
- commands drawn at reset (Issue #27) stay inside COMMAND_MAX in both directions, cover both signs, and the same seed
  gives the same command bit for bit; a given command is used as given;
- the forward limit is within what the servos allow (design record 2026-11-25-walking-targets.md): with the leg
  length of the standing model, the hip turns at under half the no-load speed while the foot is on the ground
  (the servo is loaded), and the foot can be brought forward in the air (unloaded) at under the no-load speed when
  each foot is on the ground STANCE_FRACTION of the time.
"""

import mujoco
import numpy as np

from erg import robot
from erg.env import ACT_SIZE, COMMAND_MAX, N_LEGS, OBS_SIZE, STANCE_FRACTION, Env

COMMAND = (0.2, -0.1, 0.3)


def run(seed, steps=50):
    env = Env()
    obs = [env.reset(seed, COMMAND)]
    rng = np.random.Generator(np.random.PCG64(1))
    for _ in range(steps):
        a = robot.STAND.reshape(N_LEGS, ACT_SIZE) + rng.uniform(-0.1, 0.1, (N_LEGS, ACT_SIZE))
        obs.append(env.step(a))
    return np.array(obs)


def test_deterministic():
    a, b = run(7), run(7)
    assert a.shape == (51, N_LEGS, OBS_SIZE)
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))
    assert not np.array_equal(a, run(8))


def test_observation_matches_state():
    env = Env()
    env.reset(3, COMMAND)
    for _ in range(25):
        obs = env.step(robot.STAND.reshape(N_LEGS, ACT_SIZE))
    d = env.data
    for i, leg in enumerate(robot.LEGS):
        o = obs[i]
        joints = slice(3 * i, 3 * i + 3)
        assert np.array_equal(o[0:3], d.qpos[7:][joints])
        assert np.array_equal(o[3:6], d.qvel[6:][joints])
        assert np.array_equal(o[6:9], d.actuator_force[joints] / robot.SERVO_STALL_TORQUE)
        assert np.max(np.abs(o[12:15] - d.qvel[3:6])) < 1e-12  # free joint angular velocity is in the body frame
        assert abs(o[11] - 9.81) < 0.05 and np.max(np.abs(o[9:11])) < 0.05
        touch = d.sensordata[env.model.sensor(f"{leg}_touch").adr[0]]
        assert touch > 0 and o[15] == 1.0  # standing: every foot is pressed
        assert np.array_equal(o[16:19], COMMAND)
        assert np.array_equal(o[19:], np.eye(N_LEGS)[i])


def test_actions_are_clipped():
    env = Env()
    env.reset(0)
    env.step(np.full((N_LEGS, ACT_SIZE), 100.0))
    assert np.array_equal(env.data.ctrl, env.model.actuator_ctrlrange[:, 1])


def test_foot_in_air_reads_no_contact():
    env = Env()
    env.reset(0)
    env.data.qpos[2] += 0.2  # lift the robot
    mujoco.mj_forward(env.model, env.data)
    assert np.all(env.observe()[:, 15] == 0.0)


def test_commands_are_drawn_inside_the_ranges():
    env = Env()
    drawn = np.array([env.reset(seed)[0, 16:19] for seed in range(200)])
    assert np.all(np.abs(drawn) <= COMMAND_MAX)
    assert np.all(drawn.min(axis=0) < 0) and np.all(drawn.max(axis=0) > 0)  # both directions
    again = np.array([env.reset(seed)[0, 16:19] for seed in range(5)])
    assert np.array_equal(again.view(np.uint64), drawn[:5].view(np.uint64))
    assert np.array_equal(env.reset(0, COMMAND)[0, 16:19], COMMAND)


def test_forward_limit_is_within_the_servo_speed():
    m = robot.load()
    d = mujoco.MjData(m)
    robot.reset_standing(m, d)
    hip_flexion = d.xpos[m.body("FL_thigh").id][2]
    sole = d.geom("FL_foot").xpos[2] - m.geom("FL_foot").size[0]
    leg = hip_flexion - sole
    assert 0.14 < leg < 0.18, leg
    stance = COMMAND_MAX[0] / leg  # hip speed with the foot on the ground, rad/s
    swing = stance * STANCE_FRACTION / (1 - STANCE_FRACTION)  # bringing the foot forward in the air
    assert stance < robot.SERVO_NO_LOAD_SPEED / 2, stance
    assert swing < robot.SERVO_NO_LOAD_SPEED, swing
