"""The central mode of the environment (Issue #13).

Pass criteria:
- given the same sequence of joint commands, the split mode and the central mode give the same physical state
  (positions, speeds, actuator forces, contacts' sensor readings) bit for bit at every control step: only the
  controller differs, not the body;
- the central observation is each leg's own sensor values and gait phase (#51) in the order of robot.LEGS followed
  by the command once,
  equal bit for bit to the corresponding values of the split observations; its size is CENTRAL_OBS_SIZE;
- the central action has 12 values in the order of robot.JOINT_NAMES and is clipped to the joint ranges as in the
  split mode.
"""

import numpy as np

from erg import robot
from erg.env import ACT_SIZE, CENTRAL_ACT_SIZE, CENTRAL_OBS_SIZE, LEG_SIZE, N_LEGS, Env

COMMAND = (0.1, 0.05, -0.2)


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def test_same_commands_give_the_same_physical_state():
    split, central = Env(), Env(central=True)
    split.reset(11, COMMAND)
    central.reset(11, COMMAND)
    rng = np.random.Generator(np.random.PCG64(5))
    for _ in range(100):
        targets = robot.STAND + rng.uniform(-0.3, 0.3, CENTRAL_ACT_SIZE)
        obs_split = split.step(targets.reshape(N_LEGS, ACT_SIZE))
        obs_central = central.step(targets)
        for name in ("qpos", "qvel", "actuator_force", "sensordata"):
            assert np.array_equal(bits(getattr(split.data, name)), bits(getattr(central.data, name))), name
        assert obs_central.shape == (CENTRAL_OBS_SIZE,)
        expected = np.concatenate([obs_split[:, :LEG_SIZE].ravel(), COMMAND])
        assert np.array_equal(bits(obs_central), bits(expected))


def test_central_actions_are_clipped():
    env = Env(central=True)
    env.reset(0)
    env.step(np.full(CENTRAL_ACT_SIZE, -100.0))
    assert np.array_equal(env.data.ctrl, env.model.actuator_ctrlrange[:, 0])
    assert CENTRAL_ACT_SIZE == len(robot.JOINT_NAMES)
