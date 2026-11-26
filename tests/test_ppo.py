"""PPO training (Issue #42).

Pass criteria:
- the actor sizes are under the 8,192-parameter limit of Dyne's full mlp-actor (5,575 per leg, 6,204 central);
- for both forms, a very short training (1,024 environment steps, 2 environments) run twice with the same seed
  gives the same weights bit for bit, and a different seed gives different weights; every logged loss is finite;
- the reward rewards tracking: standing still scores higher with a zero command than with a forward command;
  a fall is detected when the torso is low;
- the time per environment step and per training update is measured and printed (pytest -s).
"""

import mujoco
import numpy as np
import pytest
import torch

from erg import ppo, reward, robot
from erg.env import Env

STEPS = 1024


def run(form, seed):
    cfg = ppo.Config(form=form, n_envs=2, rollout=64, seed=seed)
    agent, timing = ppo.train(cfg, STEPS)
    weights = torch.cat([p.detach().flatten() for p in agent.parameters()]).numpy().copy()
    return weights, cfg.log, timing


def test_actor_sizes_fit_dyne():
    assert ppo.n_params(ppo.Agent("split").actor) == 5575
    assert ppo.n_params(ppo.Agent("central").actor) == 6204
    assert all(ppo.n_params(ppo.Agent(f).actor) <= 8192 for f in ("split", "central"))


@pytest.mark.parametrize("form", ["split", "central"])
def test_training_is_deterministic(form):
    a, log_a, timing = run(form, 3)
    b, _, _ = run(form, 3)
    c, _, _ = run(form, 4)
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))
    assert not np.array_equal(a, c)
    assert all(np.isfinite(e["policy_loss"]) and np.isfinite(e["value_loss"]) for e in log_a)
    print(f"\n{form}: {timing['env_seconds_per_step'] * 1e6:.0f} us per environment step, "
          f"{timing['update_seconds_per_step'] * 1e6:.0f} us of training update per step")


def test_reward_tracks_the_command():
    env = Env(central=True)
    env.reset(0)
    for _ in range(25):
        env.step(robot.STAND)
    still, _ = reward.reward(env.data, np.zeros(3), robot.STAND, robot.STAND)
    asked, _ = reward.reward(env.data, np.array([0.2, 0.0, 0.0]), robot.STAND, robot.STAND)
    assert still > asked
    assert not reward.fallen(env.data)
    env.data.qpos[2] = 0.05
    mujoco.mj_forward(env.model, env.data)
    assert reward.fallen(env.data)
