"""Messages between legs in the per-leg environment (Issue #35).

Pass criteria:
- with no loss and no delay, each leg reads, one control step later, exactly (bit for bit) the values its linked
  legs sent: gravity direction, angular velocity, foot contact and the learned values of their action, with age 1;
- a leg never reads anything from a leg it is not linked to: with links "none", every slot stays zeros and AGE_CAP;
  with "neighbours", the slot of the diagonal leg does (front-left and hind-right, front-right and hind-left);
- with 100 % message loss nothing arrives (every slot stays zeros and AGE_CAP), and when messages stop, the age
  counts up and stops at AGE_CAP;
- with message loss from 0 to 100 %, random actions (including large learned values) keep every observation finite
  and every joint command inside the joint limits; the learned values sent stay in [-1, 1];
- the same seed gives the same observations bit for bit, network included; passing the values directly (wire=False,
  for training) gives the same observations bit for bit as encoding them in Dyne's wire format;
- the time of one control step with messages is measured and printed (pytest -s).
"""

import time

import numpy as np
import pytest

from erg import robot
from erg.env import (
    ACT_SIZE, AGE_CAP, LEARNED, LEG_SIZE, MSG_SLOT, MSG_VALUES, N_LEGS, OBS_SIZE, SENSOR_SIZE, Env,
)

SLOTS = slice(LEG_SIZE + 3 + N_LEGS, OBS_SIZE)
OTHERS = [[s for s in range(N_LEGS) if s != r] for r in range(N_LEGS)]


def slots(obs):
    return obs[:, SLOTS].reshape(N_LEGS, N_LEGS - 1, MSG_SLOT)


def random_actions(rng, scale=1.0):
    a = np.zeros((N_LEGS, ACT_SIZE + LEARNED))
    a[:, :ACT_SIZE] = robot.STAND.reshape(N_LEGS, ACT_SIZE) + rng.uniform(-0.2, 0.2, (N_LEGS, ACT_SIZE))
    a[:, ACT_SIZE:] = scale * rng.normal(size=(N_LEGS, LEARNED))
    return a


def test_values_arrive_one_step_later_bit_for_bit():
    env = Env(links="all")
    env.reset(0, (0.1, 0.0, 0.0))
    rng = np.random.Generator(np.random.PCG64(1))
    for _ in range(20):
        a = random_actions(rng, scale=0.5)
        obs = env.step(a)
        sent = env.message_values()  # what each leg sent after this step
        phase = env.phase_values()
        obs = env.step(random_actions(rng, scale=0.5))
        for r in range(N_LEGS):
            for j, s in enumerate(OTHERS[r]):
                got = slots(obs)[r, j]
                assert got[MSG_VALUES] == 1
                assert np.array_equal(got[:MSG_VALUES].view(np.uint64), sent[s].view(np.uint64))
        assert np.array_equal(sent[:, -LEARNED - 2:-2], np.clip(a[:, ACT_SIZE:], -1, 1))
        assert np.array_equal(sent[:, -2:], phase)  # gait phase (#51)


@pytest.mark.parametrize("links,silent", [
    ("none", [(r, s) for r in range(N_LEGS) for s in range(N_LEGS) if s != r]),
    ("neighbours", [(0, 3), (3, 0), (1, 2), (2, 1)]),
])
def test_unlinked_legs_read_nothing(links, silent):
    env = Env(links=links)
    env.reset(2)
    rng = np.random.Generator(np.random.PCG64(2))
    for _ in range(30):
        obs = env.step(random_actions(rng))
        for r, s in silent:
            got = slots(obs)[r, OTHERS[r].index(s)]
            assert np.all(got[:MSG_VALUES] == 0) and got[MSG_VALUES] == AGE_CAP


def test_total_loss_delivers_nothing_and_age_saturates():
    env = Env(links="all", loss=1.0)
    env.reset(3)
    rng = np.random.Generator(np.random.PCG64(3))
    for _ in range(20):
        obs = env.step(random_actions(rng))
        assert np.all(slots(obs)[:, :, :MSG_VALUES] == 0) and np.all(slots(obs)[:, :, MSG_VALUES] == AGE_CAP)
    # messages that stop: age counts up from 1 and stops at AGE_CAP
    env = Env(links="all")
    env.reset(4)
    env.step(random_actions(rng))
    env.network.setting = (2, 1.0, 0, 0, 20_000)  # from now on every message is lost
    env.network.reset(4)
    ages = [slots(env.step(random_actions(rng)))[0, 0, MSG_VALUES] for _ in range(AGE_CAP + 5)]
    # the message sent at step 1 arrives at step 2 (age 1); nothing after it
    assert ages == [min(AGE_CAP, k) for k in range(1, AGE_CAP + 6)]


@pytest.mark.parametrize("loss", [0.0, 0.3, 0.7, 1.0])
def test_contract_holds_under_loss(loss):
    env = Env(links="neighbours", loss=loss, lat_us=10_000, jitter_us=10_000)
    env.reset(5)
    rng = np.random.Generator(np.random.PCG64(5))
    lo, hi = env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1]
    for _ in range(50):
        obs = env.step(random_actions(rng, scale=100.0))
        assert np.all(np.isfinite(obs))
        assert np.all(env.data.ctrl >= lo) and np.all(env.data.ctrl <= hi)
        assert np.all(np.abs(env.message_values()[:, -LEARNED:]) <= 1.0)


def run(seed, wire=True):
    env = Env(links="neighbours", loss=0.2, lat_us=10_000, jitter_us=10_000, wire=wire)
    obs = [env.reset(seed, (0.1, 0.0, 0.0))]
    rng = np.random.Generator(np.random.PCG64(9))
    for _ in range(40):
        obs.append(env.step(random_actions(rng)))
    return np.array(obs)


def test_deterministic_with_network():
    a, b = run(6), run(6)
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))
    assert not np.array_equal(a, run(7))
    c = run(6, wire=False)
    assert np.array_equal(a.view(np.uint64), c.view(np.uint64))


@pytest.mark.parametrize("wire", [True, False])
def test_time_per_control_step(wire):
    env = Env(links="all", loss=0.1, lat_us=10_000, jitter_us=5_000, wire=wire)
    env.reset(0)
    a = random_actions(np.random.Generator(np.random.PCG64(0)))
    n = 300
    t = time.perf_counter()
    for _ in range(n):
        env.step(a)
    per = (time.perf_counter() - t) / n
    print(f"\nper-leg environment with messages ('all' links, wire={wire}): {per * 1e6:.0f} us per control step")
    assert per < 0.02  # well within real time (one control period)
