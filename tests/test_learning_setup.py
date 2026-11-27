"""The learning setup on the base controller (Issue #82, docs/design/2026-11-27-learning-setup.md).

Pass criteria:
- a zero action is the base controller exactly: in training (erg.ppo.Worker, both forms, servo model randomised)
  and in evaluation (an untrained agent, whose last layer starts at zero, through erg.base_controller.run), the joint
  targets equal those of the base controller alone bit for bit;
- the action mapping clips each output to [-1, 1] and scales it by ADJUST_SCALE into erg.base_controller.Adjust;
- the reward terms on hand-made cases: tracking terms at their maxima when the torso moves at the command; the
  height term zero at the standing height and -100 * error^2 off it; the posture term sin^2 of a roll; the first and
  second differences of the actions; the CAPS terms are zero for an actor whose output is constant;
- the adjustments are held over the policy period, or ramped linearly from the previous policy step's and reach
  the new action exactly in the last control period (#86); a trained policy gives the same joint targets bit for
  bit in training (erg.ppo.Worker) and in evaluation (erg.ppo.PolicyAdjust), held or ramped;
- a tiny training run of both forms through scripts/train.py writes its weights, log and timing, and the trained
  agent runs in erg.base_controller.run (checks the pipeline only, not the result).
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from erg import base_controller as bc
from erg import ppo, reward
from erg.env import Env, N_LEGS, WALK_COMMAND_HIGH, WALK_COMMAND_LOW
from erg.servo import ServoParams

ROOT = Path(__file__).resolve().parent.parent
COMMAND = (0.1, 0.02, 0.1)


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


@pytest.mark.parametrize("form", ["split", "central"])
def test_zero_action_is_the_base_controller(form):
    steps = 60  # policy steps: 2.4 s
    base = bc.run(COMMAND, seed=7, seconds=steps * ppo.POLICY_PERIODS * bc.DT, servo=ServoParams(),
                  randomize_servo=True, window=0.4)
    w = ppo.Worker(ppo.Config(form=form), 0)
    w.reset(seed=7, command=COMMAND)
    zero = np.zeros(ppo.Agent(form).act_dim * (N_LEGS if form == "split" else 1))
    for j in range(steps):
        _, fell = w.step(zero)
        assert not fell
        assert np.array_equal(bits(w.targets), bits(base.targets[ppo.POLICY_PERIODS * j + 1]))


@pytest.mark.parametrize("form", ["split", "central"])
def test_untrained_agent_is_the_base_controller(form):
    agent = ppo.Agent(form)
    base = bc.run(COMMAND, seed=3, seconds=1.0, window=0.4, servo=ServoParams(), randomize_servo=True)
    pol = bc.run(COMMAND, seed=3, seconds=1.0, window=0.4, servo=ServoParams(), randomize_servo=True,
                 adjust=ppo.PolicyAdjust(agent))
    assert np.array_equal(bits(base.targets), bits(pol.targets))


def test_action_mapping():
    assert bc.adjust_from_action(np.zeros(bc.N_ADJUST)) == bc.Adjust()
    adj = bc.adjust_from_action([2.0, -3.0, 0.5, 1.0, -0.5, 0.25])
    assert adj == bc.Adjust(freq=0.3, foot_dx=-0.02, foot_dy=0.5 * 0.02, step_height=0.006, place_gain=-0.25,
                            att_gain=0.125)
    with pytest.raises(ValueError):
        bc.adjust_from_action(np.zeros(bc.N_ADJUST + 1))


def test_period_actions():
    prev, a = np.full((N_LEGS, bc.N_ADJUST), -0.2), np.full((N_LEGS, bc.N_ADJUST), 0.6)
    held = ppo.period_actions(prev, a, False)
    assert len(held) == ppo.POLICY_PERIODS and all(h is a for h in held)
    ramp = ppo.period_actions(prev, a, True)
    assert len(ramp) == ppo.POLICY_PERIODS and ramp[-1] is a
    assert np.allclose(ramp[0], -0.2 + 0.8 / ppo.POLICY_PERIODS)
    zero = np.zeros((N_LEGS, bc.N_ADJUST))
    assert all(np.array_equal(bits(p), bits(zero)) for p in ppo.period_actions(zero, zero, True))
    assert ppo.Config().interpolate is True and ppo.Config(action="gait").interpolate is False


@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("form", ["split", "central"])
def test_training_and_evaluation_apply_the_same_adjustments(form, interpolate):
    agent = ppo.Agent(form)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        agent.actor[-1].weight.normal_(generator=g)
        agent.actor[-1].weight.mul_(0.05)
    steps = 25
    ev = bc.run(COMMAND, seed=5, seconds=steps * ppo.POLICY_PERIODS * bc.DT, window=0.4, servo=ServoParams(),
                randomize_servo=True, adjust=ppo.PolicyAdjust(agent, interpolate))
    w = ppo.Worker(ppo.Config(form=form, interpolate=interpolate), 0)
    w.reset(seed=5, command=COMMAND)
    for j in range(steps):
        w.step(ppo.policy_action(agent, w.actor_obs()))
        assert np.array_equal(bits(w.targets), bits(ev.targets[ppo.POLICY_PERIODS * j + 1]))


def test_walk_commands_in_range():
    w = ppo.Worker(ppo.Config(form="split", servo="none"), 0)
    for _ in range(20):
        w.reset()
        assert np.all(w.env.command >= WALK_COMMAND_LOW) and np.all(w.env.command <= WALK_COMMAND_HIGH)


def standing(env, height=reward.WALK_HEIGHT, roll=0.0, vx=0.0):
    env.reset(0, (0.0, 0.0, 0.0))
    d = env.data
    d.qpos[2] = height
    d.qpos[3:7] = (math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0)
    d.qvel[:] = 0.0
    d.qvel[0] = vx
    mujoco.mj_forward(env.model, d)
    return d


def test_reward_terms():
    env = Env()
    d = standing(env, vx=0.1)
    t = reward.walk_terms(d, np.array([0.1, 0.0, 0.0]))
    assert t["speed"] == reward.WALK_W_SPEED and t["turn"] == reward.WALK_W_TURN
    assert t["height"] == 0.0 and t["posture"] == 0.0
    assert t["torque"] <= 0.0
    t = reward.walk_terms(standing(env, vx=0.15), np.array([0.1, 0.0, 0.0]))
    assert t["speed"] == pytest.approx(math.exp(-1.0))  # one width off
    d = standing(env, height=reward.WALK_HEIGHT + 0.01)
    assert reward.walk_terms(d, np.zeros(3))["height"] == pytest.approx(-100.0 * 0.01**2)
    d = standing(env, roll=0.2)
    assert reward.walk_terms(d, np.zeros(3))["posture"] == pytest.approx(-math.sin(0.2) ** 2)

    a = np.ones((N_LEGS, bc.N_ADJUST))
    s = reward.smoothness_terms(a, a, a)
    assert s["rate"] == 0.0 and s["rate2"] == 0.0
    s = reward.smoothness_terms(a, 0 * a, 0 * a)
    assert s["rate"] == -reward.WALK_W_RATE and s["rate2"] == -reward.WALK_W_RATE2
    s = reward.smoothness_terms(2 * a, a, 0 * a)  # constant rate: no second difference
    assert s["rate"] == -reward.WALK_W_RATE and s["rate2"] == 0.0


@pytest.mark.parametrize("form", ["split", "central"])
def test_caps_zero_for_a_constant_actor(form):
    agent = ppo.Agent(form)  # last layer zero: constant output
    shape = (5, N_LEGS, 67) if form == "split" else (5, 75)
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(shape, generator=g, dtype=torch.float64)
    nxt = torch.randn(shape, generator=g, dtype=torch.float64)
    t, s = ppo.caps_losses(agent, obs, nxt, torch.ones(5, dtype=torch.float64), g)
    assert t.item() == 0.0 and s.item() == 0.0
    with torch.no_grad():
        agent.actor[-1].bias.fill_(0.3)
        agent.actor[-1].weight.normal_(generator=g)
    t, s = ppo.caps_losses(agent, obs, nxt, torch.ones(5, dtype=torch.float64), g)
    assert t.item() > 0.0 and s.item() > 0.0
    t0, _ = ppo.caps_losses(agent, obs, nxt, torch.zeros(5, dtype=torch.float64), g)
    assert t0.item() == 0.0  # pairs across an episode's end do not count


@pytest.mark.parametrize("form", ["split", "central"])
def test_training_pipeline(form, tmp_path):
    out = tmp_path / form
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--form", form, "--steps", "512",
                    "--n-envs", "2", "--seed", "1", "--out", str(out)], check=True, cwd=ROOT,
                   capture_output=True)
    info = json.loads((out / "run.json").read_text())
    assert info["config"]["action"] == "base" and info["config"]["servo"] == "random"
    assert info["timing"]["steps"] >= 512
    assert (out / "log.csv").read_text().startswith("steps,")
    agent = ppo.Agent(form)
    agent.load_state_dict(torch.load(out / "agent.pt"))
    r = bc.run((0.1, 0.0, 0.0), seed=0, seconds=0.5, window=0.2, servo=ServoParams(), adjust=ppo.PolicyAdjust(agent))
    assert np.all(np.isfinite(r.targets))
