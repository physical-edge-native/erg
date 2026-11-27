"""PPO (proximal policy optimization) on the CPU, for both forms of control (Issue #42).

- **Per-leg form** ("split"): one actor network shared by the four legs; each leg's actor sees only its own
  observation (its sensors, its gait phase, the command, its leg identifier and its neighbours' messages).
- **Central form**: one actor sees the central observation (every leg's sensors and gait phase, the command).

Both forms are trained with the same algorithm, the same reward (erg.reward), the same budget of environment
steps, and the same critic: a value network on the central observation (every leg's sensors and the command),
used only during training and never on the robot.

Actors are ReLU networks with ACTOR_HIDDEN units per layer (bit-exact in Dyne's WebAssembly runtime; under the
8,192-parameter limit of Dyne's full mlp-actor). Actions are Gaussian with a learned, state-independent standard
deviation; the mean is used on the robot.

Two action mappings:

- **"base"** (default since #82, docs/design/2026-11-27-learning-setup.md): the policy adjusts the base controller of
  erg.base_controller. Each leg runs the base controller every control period, in both forms, from its own sensors
  and its neighbours' messages (the environment is the per-leg one for both forms). Every POLICY_PERIODS control
  periods (25 Hz) the policy gives erg.base_controller.N_ADJUST adjustments per leg (frequency, foot-placement
  residuals, step height, gains; erg.base_controller.adjust_from_action), held in between; the central actor gives
  the same outputs for each leg. A zero action is the base controller exactly, and the actor's last layer starts at
  zero. The environment has the servo model of #70 with randomised delay; the reward is erg.reward's walk terms
  with penalties on the first and second differences of the actions; the actor's loss adds CAPS terms (temporal and
  spatial smoothness of the mean action; Mysore et al., 2021).
- **"gait"** (#51): a joint target is the base gait's target at the leg's phase (the trot of erg.gait with the
  stride of the commanded forward speed, erg.gait.leg_targets) + ACTION_SCALE * correction; per leg, 3 joint
  corrections, the learned message values and the frequency correction of its gait clock; central, 12 joint
  corrections and 4 frequency corrections. The leg firmware runs this mapping (scripts/gen_firmware_layout.py).

Runs are deterministic: one thread for PyTorch, seeded generators, environments stepped in a fixed order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from erg import base_controller as bc
from erg import gait, robot
from erg import reward as rw
from erg.env import (
    ACT_SIZE, CENTRAL_ACT_SIZE, CENTRAL_OBS_SIZE, CONTACT_THRESHOLD, LEARNED, LEG_SIZE, N_LEGS, OBS_SIZE, Env,
    draw_command, draw_walk_command,
)
from erg.servo import ServoParams

ACTOR_HIDDEN = (48, 48)
CRITIC_HIDDEN = (128, 128)
ACTION_SCALE = 0.25  # "gait": rad per unit of correction, around the base gait (0.5 around standing before #51)
EPISODE_STEPS = 500  # control periods: 10 s at 20 ms
# Command curriculum (#49), "gait" only: commands are drawn in the full ranges scaled by a factor that grows linearly
# from CURRICULUM_START to 1 over the first CURRICULUM_FRACTION of the training steps.
CURRICULUM_START = 0.3
CURRICULUM_FRACTION = 0.3
POLICY_PERIODS = 2  # "base": control periods per policy step (40 ms, 25 Hz)
CAPS_NOISE = 0.01  # standard deviation of the observation noise of the spatial CAPS term

# Settings that differ between the action mappings; Config fields left at None take these.
DEFAULTS = {
    "base": {"servo": "random", "rollout": 64, "entropy_coef": 0.0, "init_log_std": -1.6, "caps_temporal": 0.1,
             "caps_spatial": 1.0, "separate_clip": True},
    "gait": {"servo": "none", "rollout": 128, "entropy_coef": 0.01, "init_log_std": -2.3, "caps_temporal": 0.0,
             "caps_spatial": 0.0, "separate_clip": False},
}


@dataclass
class Config:
    form: str = "split"  # "split" or "central"
    action: str = "base"  # "base": adjust the base controller (#82); "gait": correct the base gait's joints (#51)
    servo: str | None = None  # "none" (the model's position actuators), "nominal" or "random" (erg.servo, #70)
    links: str = "neighbours"
    loss: float = 0.0
    lat_us: int = 0
    jitter_us: int = 0
    n_envs: int = 4
    rollout: int | None = None  # policy steps per environment between updates
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99  # per policy step
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float | None = None  # "gait": 0.01 (0 before the third pilot, #49; as in legged_gym)
    max_grad_norm: float = 1.0
    init_log_std: float | None = None  # "gait": -2.3, about 0.1 (-0.5 in the third and fourth pilots, #49, #51)
    caps_temporal: float | None = None  # weight of the temporal CAPS term (#82)
    caps_spatial: float | None = None  # weight of the spatial CAPS term (#82)
    # Clip the gradient norms of the actor (with its log std) and of the critic separately (#82), so that the
    # critic's large gradients early in training do not shrink the actor's step; "gait": one norm for both.
    separate_clip: bool | None = None
    seed: int = 0
    log: list = field(default_factory=list)

    def __post_init__(self):
        if self.action not in DEFAULTS:
            raise ValueError(self.action)
        if self.servo not in (None, "none", "nominal", "random"):
            raise ValueError(self.servo)
        for k, v in DEFAULTS[self.action].items():
            if getattr(self, k) is None:
                setattr(self, k, v)


def mlp(n_in: int, hidden: tuple, n_out: int) -> nn.Sequential:
    layers, last = [], n_in
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, n_out))
    return nn.Sequential(*layers).double()


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


class Agent(nn.Module):
    def __init__(self, form: str, action: str = "base"):
        super().__init__()
        self.form = form
        self.action = action
        if action == "base":  # the base controller's adjustments, per leg (#82)
            per_leg = bc.N_ADJUST
        elif action == "gait":  # joint corrections, learned values, clock frequency correction (#51)
            per_leg = ACT_SIZE + LEARNED + 1
        else:
            raise ValueError(action)
        if form == "split":
            self.act_dim = per_leg
            self.actor = mlp(OBS_SIZE, ACTOR_HIDDEN, self.act_dim)
        elif form == "central":
            # "base": each leg's adjustments in the order of robot.LEGS; "gait": 12 joint corrections, then 4 clock
            # frequency corrections (no learned values: no messages)
            self.act_dim = N_LEGS * per_leg if action == "base" else CENTRAL_ACT_SIZE + N_LEGS
            self.actor = mlp(CENTRAL_OBS_SIZE, ACTOR_HIDDEN, self.act_dim)
        else:
            raise ValueError(form)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), -1.0, dtype=torch.float64))
        if action == "base":  # start at the zero action: the base controller alone
            with torch.no_grad():
                self.actor[-1].weight.zero_()
                self.actor[-1].bias.zero_()
        self.critic = mlp(CENTRAL_OBS_SIZE, CRITIC_HIDDEN, 1)

    def dist(self, actor_obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(actor_obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


def central_obs(obs_split: np.ndarray) -> np.ndarray:
    """The central observation (every leg's sensors and gait phase, then the command) from the per-leg ones."""
    return np.concatenate([obs_split[:, :LEG_SIZE].ravel(), obs_split[0, LEG_SIZE:LEG_SIZE + 3]])


def base_targets(phase: np.ndarray, forward: float = 0.0) -> np.ndarray:
    """Joint targets of the base gait at each leg's phase, shape (4, 3): the trot of erg.gait with the stride that
    walks at the commanded forward speed (stepping in place at zero). Each leg's depend only on its own phase and the
    command, so a leg's controller can compute them alone."""
    return np.stack([gait.leg_targets(p, forward * gait.PERIOD / 2) for p in phase])


def servo_params(setting: str) -> ServoParams | None:
    return None if setting == "none" else ServoParams()


class Worker:
    """One environment with its episode bookkeeping."""

    def __init__(self, cfg: Config, index: int):
        self.cfg = cfg
        self.index = index
        self.episode = 0
        self.command_scale = 1.0
        base = cfg.action == "base"
        # "base": the per-leg environment for both forms, since the base controller runs on each leg with its
        # neighbours' messages in both forms; the central actor sees the central observation cut out of it.
        self.env = Env(central=cfg.form == "central" and not base, links=cfg.links, loss=cfg.loss,
                       lat_us=cfg.lat_us, jitter_us=cfg.jitter_us, wire=False, servo=servo_params(cfg.servo),
                       randomize_servo=cfg.servo == "random")
        self.reset()

    def reset(self, seed: int | None = None, command=None):
        """A new episode; its seed and command come from the worker's sequence unless given."""
        if seed is None:
            seed = (self.cfg.seed * 1_000_003 + self.index) * 100_003 + self.episode
            self.episode += 1
        if command is None:
            rng = np.random.Generator(np.random.PCG64([seed, 1]))
            command = draw_walk_command(rng) if self.cfg.action == "base" else draw_command(rng) * self.command_scale
        self.obs = self.env.reset(seed, command)
        self.steps = 0  # control periods
        self.prev_targets = base_targets(self.env.phase, self.env.command[0]).ravel()
        self.in_contact = self.contacts()
        self.air_time = np.zeros(N_LEGS)
        self.ctrls = [bc.LegController(leg) for leg in robot.LEGS]
        zero = np.zeros((N_LEGS, bc.N_ADJUST))
        self.prev_actions = (zero, zero)  # the clipped actions of the two policy steps before
        self.targets = None  # "base": joint targets of the last control period, (4, 3)

    def contacts(self) -> np.ndarray:
        d = self.env.data
        return np.array([d.sensor(f"{leg}_touch").data[0] > CONTACT_THRESHOLD for leg in robot.LEGS])

    def actor_obs(self) -> np.ndarray:
        """(4, OBS_SIZE) per-leg, or (CENTRAL_OBS_SIZE,) central."""
        if self.cfg.form == "central" and self.obs.ndim == 2:
            return central_obs(self.obs)
        return self.obs

    def critic_obs(self) -> np.ndarray:
        return self.obs if self.obs.ndim == 1 else central_obs(self.obs)

    def step(self, action: np.ndarray) -> tuple[float, bool]:
        """Apply one action (raw policy output); returns the reward and whether the episode ended by a fall
        (time limits end episodes too, but are not falls). self.steps counts the control periods."""
        if self.cfg.action == "base":
            return self._step_base(action)
        return self._step_gait(action)

    def control_period(self, adjs) -> None:
        """One control period of the legs' base controllers with their Adjust (None: the base controller alone)."""
        outs = [c.step(bc.leg_input(self.obs[i], leg), adjs[i])
                for i, (c, leg) in enumerate(zip(self.ctrls, robot.LEGS))]
        actions = np.array([list(o.targets) + list(o.message) for o in outs])
        self.targets = actions[:, :ACT_SIZE]
        self.obs = self.env.step(actions, freq=[o.freq for o in outs])
        self.steps += 1

    def _step_base(self, action: np.ndarray) -> tuple[float, bool]:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(N_LEGS, bc.N_ADJUST), -1.0, 1.0)
        adjs = [bc.adjust_from_action(a[i]) for i in range(N_LEGS)]
        total, n, fell = 0.0, 0, False
        for _ in range(POLICY_PERIODS):
            self.control_period(adjs)
            total += sum(rw.walk_terms(self.env.data, self.env.command).values())
            n += 1
            fell = rw.fallen(self.env.data)
            if fell or self.steps >= EPISODE_STEPS:
                break
        a1, a2 = self.prev_actions
        r = total / n + sum(rw.smoothness_terms(a, a1, a2).values())
        self.prev_actions = (a, a1)
        return rw.clip_total(r), bool(fell)

    def _step_gait(self, action: np.ndarray) -> tuple[float, bool]:
        base = base_targets(self.env.phase, self.env.command[0])
        if self.cfg.form == "split":
            a = action.reshape(N_LEGS, ACT_SIZE + LEARNED + 1)
            env_action = np.concatenate([base + ACTION_SCALE * a[:, :ACT_SIZE], a[:, ACT_SIZE:ACT_SIZE + LEARNED]],
                                        axis=1)
            targets = env_action[:, :ACT_SIZE].ravel()
            freq = a[:, ACT_SIZE + LEARNED]
        else:
            env_action = base.ravel() + ACTION_SCALE * action[:CENTRAL_ACT_SIZE]
            targets = env_action
            freq = action[CENTRAL_ACT_SIZE:]
        self.obs = self.env.step(env_action, freq)
        targets = np.clip(targets, self.env.model.actuator_ctrlrange[:, 0], self.env.model.actuator_ctrlrange[:, 1])
        r, fell = rw.reward(self.env.data, self.env.command, targets, self.prev_targets)
        contact = self.contacts()
        touchdown = contact & ~self.in_contact
        self.air_time += robot.CONTROL_PERIOD
        r = rw.clip_total(r + rw.air_time_reward(touchdown, self.air_time, self.env.command))
        self.air_time[contact] = 0.0
        self.in_contact = contact
        self.prev_targets = targets
        self.steps += 1
        return r, fell


def caps_losses(agent: Agent, obs: torch.Tensor, next_obs: torch.Tensor, valid: torch.Tensor,
                gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """CAPS terms on the actor's mean (Mysore et al., 2021): temporal, the mean squared change of the mean action
    between consecutive policy steps of the same episode (valid marks them); spatial, the mean squared change under
    observation noise of standard deviation CAPS_NOISE (a local Lipschitz penalty)."""
    mu = agent.actor(obs)
    d_t = ((mu - agent.actor(next_obs)) ** 2).reshape(len(obs), -1).mean(-1)
    temporal = (d_t * valid).sum() / valid.sum().clamp(min=1.0)
    noise = torch.randn(obs.shape, generator=gen, dtype=torch.float64) * CAPS_NOISE
    spatial = ((mu - agent.actor(obs + noise)) ** 2).mean()
    return temporal, spatial


def train(cfg: Config, total_steps: int, progress=None) -> tuple[Agent, dict]:
    """Train for about `total_steps` environment steps (control periods); returns the agent and timing figures."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    agent = Agent(cfg.form, cfg.action)
    with torch.no_grad():
        agent.log_std.fill_(cfg.init_log_std)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    workers = [Worker(cfg, i) for i in range(cfg.n_envs)]
    use_caps = cfg.caps_temporal > 0 or cfg.caps_spatial > 0
    t_env = t_update = 0.0
    steps_done = 0
    while steps_done < total_steps:
        if cfg.action == "gait":
            scale = min(1.0, CURRICULUM_START + (1 - CURRICULUM_START) * steps_done
                        / (CURRICULUM_FRACTION * total_steps))
        else:
            scale = 1.0
        for w in workers:
            w.command_scale = scale
        a_obs, c_obs, acts, logps, rews, dones, vals, n_obs, valid = [], [], [], [], [], [], [], [], []
        falls = 0
        t0 = time.perf_counter()
        for _ in range(cfg.rollout):
            ao = np.stack([w.actor_obs() for w in workers])
            co = np.stack([w.critic_obs() for w in workers])
            with torch.no_grad():
                d = agent.dist(torch.from_numpy(ao))
                noise = torch.randn(d.mean.shape, generator=gen, dtype=torch.float64)
                a = d.mean + d.stddev * noise
                lp = d.log_prob(a).reshape(cfg.n_envs, -1).sum(-1)
                v = agent.critic(torch.from_numpy(co)).squeeze(-1)
            r_step, d_step, nxt, ok = [], [], [], []
            for i, w in enumerate(workers):
                before = w.steps
                r, fell = w.step(a[i].numpy())
                steps_done += w.steps - before
                end = fell or w.steps >= EPISODE_STEPS
                r_step.append(r)
                d_step.append(1.0 if fell else 0.0)
                falls += int(fell)
                nxt.append(w.actor_obs())
                ok.append(0.0 if end else 1.0)
                if end:
                    if not fell:  # time limit: bootstrap from the value of the last state
                        with torch.no_grad():
                            r_step[-1] += cfg.gamma * agent.critic(torch.from_numpy(w.critic_obs())).item()
                        d_step[-1] = 1.0
                    w.reset()
            a_obs.append(ao); c_obs.append(co); acts.append(a.numpy()); logps.append(lp.numpy())
            rews.append(r_step); dones.append(d_step); vals.append(v.numpy())
            n_obs.append(np.stack(nxt)); valid.append(ok)
        with torch.no_grad():
            last_v = agent.critic(torch.from_numpy(np.stack([w.critic_obs() for w in workers]))).squeeze(-1).numpy()
        t1 = time.perf_counter()
        t_env += t1 - t0

        rews, dones, vals = np.array(rews), np.array(dones), np.array(vals)
        adv = np.zeros_like(rews)
        gae = np.zeros(cfg.n_envs)
        for t in reversed(range(cfg.rollout)):
            nxt_v = last_v if t == cfg.rollout - 1 else vals[t + 1]
            delta = rews[t] + cfg.gamma * nxt_v * (1 - dones[t]) - vals[t]
            gae = delta + cfg.gamma * cfg.lam * (1 - dones[t]) * gae
            adv[t] = gae
        ret = adv + vals

        n = cfg.rollout * cfg.n_envs
        a_obs = np.array(a_obs)
        B_ao = torch.from_numpy(a_obs.reshape(n, *a_obs.shape[2:]))
        B_no = torch.from_numpy(np.array(n_obs).reshape(n, *a_obs.shape[2:]))
        B_ok = torch.from_numpy(np.array(valid).reshape(n))
        B_co = torch.from_numpy(np.array(c_obs).reshape(n, -1))
        B_a = torch.from_numpy(np.array(acts).reshape(n, *np.array(acts).shape[2:]))
        B_lp = torch.from_numpy(np.array(logps).reshape(n))
        B_adv = torch.from_numpy(adv.reshape(n))
        B_ret = torch.from_numpy(ret.reshape(n))
        B_adv = (B_adv - B_adv.mean()) / (B_adv.std() + 1e-8)
        mb = n // cfg.minibatches
        losses = []
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, generator=gen)
            for k in range(cfg.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                d = agent.dist(B_ao[idx])
                lp = d.log_prob(B_a[idx]).reshape(len(idx), -1).sum(-1)
                ratio = (lp - B_lp[idx]).exp()
                pg = -torch.min(ratio * B_adv[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * B_adv[idx]).mean()
                vl = ((agent.critic(B_co[idx]).squeeze(-1) - B_ret[idx]) ** 2).mean()
                ent = d.entropy().reshape(len(idx), -1).sum(-1).mean()
                loss = pg + cfg.value_coef * vl - cfg.entropy_coef * ent
                caps_t = caps_s = torch.zeros((), dtype=torch.float64)
                if use_caps:
                    caps_t, caps_s = caps_losses(agent, B_ao[idx], B_no[idx], B_ok[idx], gen)
                    loss = loss + cfg.caps_temporal * caps_t + cfg.caps_spatial * caps_s
                opt.zero_grad()
                loss.backward()
                if cfg.separate_clip:
                    nn.utils.clip_grad_norm_([*agent.actor.parameters(), agent.log_std], cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(agent.critic.parameters(), cfg.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()
                losses.append((pg.item(), vl.item(), caps_t.item(), caps_s.item()))
        t_update += time.perf_counter() - t1
        entry = {"steps": steps_done, "command_scale": scale, "mean_reward": float(rews.mean()),
                 "falls": falls,
                 "policy_loss": float(np.mean([l[0] for l in losses])),
                 "value_loss": float(np.mean([l[1] for l in losses])),
                 "caps_temporal": float(np.mean([l[2] for l in losses])),
                 "caps_spatial": float(np.mean([l[3] for l in losses]))}
        cfg.log.append(entry)
        if progress:
            progress(entry)
    timing = {"env_seconds_per_step": t_env / steps_done, "update_seconds_per_step": t_update / steps_done,
              "steps": steps_done}
    return agent, timing


def policy_action(agent: Agent, obs: np.ndarray) -> np.ndarray:
    """The deterministic action (mean) for one observation, as run on the robot."""
    with torch.no_grad():
        return agent.actor(torch.from_numpy(np.asarray(obs, dtype=np.float64))).numpy()


class PolicyAdjust:
    """A trained "base" agent as the `adjust` of erg.base_controller.run: every POLICY_PERIODS control periods its
    mean action on the current observation (per-leg rows or the central observation), held in between."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.adjs = None

    def __call__(self, k: int, obs: np.ndarray) -> list:
        if k % POLICY_PERIODS == 0:
            x = obs if self.agent.form == "split" else central_obs(obs)
            a = policy_action(self.agent, x).reshape(N_LEGS, bc.N_ADJUST)
            self.adjs = [bc.adjust_from_action(a[i]) for i in range(N_LEGS)]
        return self.adjs
