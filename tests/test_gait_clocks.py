"""Per-leg gait clocks and the base gait (Issue #51).

Pass criteria:
- each leg's clock starts at the trot offsets of erg.gait.PHASE and, with no frequency correction, advances by
  CONTROL_PERIOD / erg.gait.PERIOD per control step; a frequency correction changes only that leg's clock, and
  corrections are clipped to [-1, 1] (0.5 to 1.5 times the base frequency);
- the central mode has the same clocks, advancing the same way;
- with zero corrections, the joint targets of the PPO controllers are the base gait (the trot with the
  commanded forward speed's stride) at each leg's phase, bit for bit, and with zero stride the base gait's foot stays above its standing point;
- a leg's phase reaches only the legs it is linked to: with "neighbours", never the diagonal leg;
- with zero corrections, the base gait alone keeps the robot up (no fall) for 10 s stepping in place, and walks
  forward at more than 0.3 of a command of 0.15 m/s (0.12 m/s with the first body; about 0.06 m/s with body L1, #59;
  the base gait alone is not expected to reach the command, which is left to the corrections).
"""

import numpy as np
import pytest

from erg import gait, ppo, reward, robot
from erg.env import (
    ACT_SIZE, AGE_CAP, FREQ_RANGE, LEARNED, LEG_SIZE, MSG_SLOT, MSG_VALUES, N_LEGS, OBS_SIZE, Env,
)

START = np.array([gait.PHASE[leg] for leg in robot.LEGS])
STEP = robot.CONTROL_PERIOD / gait.PERIOD
SLOTS = slice(LEG_SIZE + 3 + N_LEGS, OBS_SIZE)
DIAGONAL = {0: 3, 1: 2, 2: 1, 3: 0}  # FL-HR, FR-HL


def stand(env):
    return robot.STAND if env.central else robot.STAND.reshape(N_LEGS, ACT_SIZE)


@pytest.mark.parametrize("central", [False, True])
def test_clocks_advance_independently(central):
    env = Env(central=central)
    env.reset(0)
    assert np.array_equal(env.phase, START)
    phase = START.copy()
    for k in range(30):
        f = np.zeros(N_LEGS) if k < 10 else np.array([0.0, 1.0, -0.4, 3.0])  # 3.0 is clipped to 1.0
        env.step(stand(env), f)
        phase = (phase + STEP * (1.0 + FREQ_RANGE * np.clip(f, -1, 1))) % 1.0
        assert np.array_equal(env.phase, phase)
    env.reset(1)  # the clocks start again
    assert np.array_equal(env.phase, START)


def zero_action(form):
    return np.zeros(ppo.Agent(form).act_dim * (N_LEGS if form == "split" else 1))


def test_zero_corrections_follow_the_base_gait():
    for form in ("split", "central"):
        w = ppo.Worker(ppo.Config(form=form), 0)
        for _ in range(20):
            expected = ppo.base_targets(w.env.phase, w.env.command[0])
            w.step(zero_action(form))
            assert np.array_equal(w.env.data.ctrl, np.clip(expected.ravel(), w.env.model.actuator_ctrlrange[:, 0],
                                                           w.env.model.actuator_ctrlrange[:, 1]))
    x, z = gait.foot_path(0.75, 0.0)
    assert x == 0.0 and z > -gait.STAND_DEPTH


def test_phase_reaches_only_linked_legs():
    env = Env(links="neighbours")
    env.reset(0)
    for _ in range(5):
        obs = env.step(np.zeros((N_LEGS, ACT_SIZE + LEARNED)) + np.concatenate(
            [robot.STAND.reshape(N_LEGS, ACT_SIZE), np.zeros((N_LEGS, LEARNED))], axis=1))
    slots = obs[:, SLOTS].reshape(N_LEGS, N_LEGS - 1, MSG_SLOT)
    for r in range(N_LEGS):
        for j, s in enumerate(x for x in range(N_LEGS) if x != r):
            phase = slots[r, j, MSG_VALUES - 2:MSG_VALUES]
            if s == DIAGONAL[r]:
                assert np.all(phase == 0) and slots[r, j, MSG_VALUES] == AGE_CAP
            else:
                assert np.linalg.norm(phase) == pytest.approx(1.0)


def test_base_gait_alone_does_not_fall():
    w = ppo.Worker(ppo.Config(form="split"), 0)
    w.env.reset(0, np.zeros(3))
    for _ in range(ppo.EPISODE_STEPS):
        _, fell = w.step(zero_action("split"))
        assert not fell
    assert not reward.fallen(w.env.data)


def test_base_gait_walks_forward():
    w = ppo.Worker(ppo.Config(form="central"), 0)
    w.env.reset(0, np.array([0.15, 0.0, 0.0]))
    x0 = w.env.data.qpos[0]
    for _ in range(ppo.EPISODE_STEPS):
        _, fell = w.step(zero_action("central"))
        assert not fell
    assert (w.env.data.qpos[0] - x0) / (ppo.EPISODE_STEPS * robot.CONTROL_PERIOD) > 0.3 * 0.15
