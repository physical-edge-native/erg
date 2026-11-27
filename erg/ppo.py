"""PPO (proximal policy optimization) on the CPU, for both forms of control (Issue #42).

- **Per-leg form** ("split"): one actor network shared by the four legs; each leg's actor sees only its own
  observation (its sensors, the command, its leg identifier and its neighbours' messages) and outputs its 3 joint
  targets and its learned message values. The joint action of the robot is the four legs' actions together.
- **Central form**: one actor sees the central observation and outputs the 12 joint targets.

Both forms are trained with the same algorithm, the same reward (erg.reward), the same budget of environment
steps, and the same critic: a value network on the central observation (every leg's sensors and the command),
used only during training and never on the robot.

Actors are ReLU networks with ACTOR_HIDDEN units per layer (bit-exact in Dyne's WebAssembly runtime; under the
8,192-parameter limit of Dyne's full mlp-actor). Actions are Gaussian with a learned, state-independent standard
deviation; the mean is used on the robot. A joint target is robot.STAND + ACTION_SCALE * action.

Runs are deterministic: one thread for PyTorch, seeded generators, environments stepped in a fixed order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from erg import reward as rw
from erg import robot
from erg.env import (
    ACT_SIZE, CENTRAL_ACT_SIZE, CENTRAL_OBS_SIZE, CONTACT_THRESHOLD, LEARNED, N_LEGS, OBS_SIZE, SENSOR_SIZE, Env,
    draw_command,
)

ACTOR_HIDDEN = (48, 48)
CRITIC_HIDDEN = (128, 128)
ACTION_SCALE = 0.5  # rad per unit of action, around the standing posture
EPISODE_STEPS = 500  # 10 s at 20 ms
# Command curriculum (#49): commands are drawn in the full ranges scaled by a factor that grows linearly from
# CURRICULUM_START to 1 over the first CURRICULUM_FRACTION of the training steps.
CURRICULUM_START = 0.3
CURRICULUM_FRACTION = 0.3
STAND = robot.STAND.reshape(N_LEGS, ACT_SIZE)


@dataclass
class Config:
    form: str = "split"  # "split" or "central"
    links: str = "neighbours"
    loss: float = 0.0
    lat_us: int = 0
    jitter_us: int = 0
    n_envs: int = 4
    rollout: int = 128  # steps per environment between updates
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01  # 0 before the third pilot (#49); as in legged_gym
    max_grad_norm: float = 1.0
    init_log_std: float = -0.5  # -1.0 before the third pilot (#49): more exploration at the start
    seed: int = 0
    log: list = field(default_factory=list)


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
    def __init__(self, form: str):
        super().__init__()
        self.form = form
        if form == "split":
            self.act_dim = ACT_SIZE + LEARNED
            self.actor = mlp(OBS_SIZE, ACTOR_HIDDEN, self.act_dim)
            self.log_std = nn.Parameter(torch.full((self.act_dim,), -1.0, dtype=torch.float64))
        elif form == "central":
            self.act_dim = CENTRAL_ACT_SIZE
            self.actor = mlp(CENTRAL_OBS_SIZE, ACTOR_HIDDEN, self.act_dim)
            self.log_std = nn.Parameter(torch.full((self.act_dim,), -1.0, dtype=torch.float64))
        else:
            raise ValueError(form)
        self.critic = mlp(CENTRAL_OBS_SIZE, CRITIC_HIDDEN, 1)

    def dist(self, actor_obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(actor_obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


def central_obs(obs_split: np.ndarray) -> np.ndarray:
    """The central observation (every leg's sensors, then the command) from the per-leg observations."""
    return np.concatenate([obs_split[:, :SENSOR_SIZE].ravel(), obs_split[0, SENSOR_SIZE:SENSOR_SIZE + 3]])


class Worker:
    """One environment with its episode bookkeeping."""

    def __init__(self, cfg: Config, index: int):
        self.cfg = cfg
        self.index = index
        self.episode = 0
        self.command_scale = 1.0
        self.env = Env(central=cfg.form == "central", links=cfg.links, loss=cfg.loss, lat_us=cfg.lat_us,
                       jitter_us=cfg.jitter_us, wire=False)
        self.reset()

    def reset(self):
        seed = (self.cfg.seed * 1_000_003 + self.index) * 100_003 + self.episode
        self.episode += 1
        command = draw_command(np.random.Generator(np.random.PCG64([seed, 1]))) * self.command_scale
        self.obs = self.env.reset(seed, command)
        self.steps = 0
        self.prev_targets = robot.STAND.copy()
        self.in_contact = self.contacts()
        self.air_time = np.zeros(N_LEGS)

    def contacts(self) -> np.ndarray:
        d = self.env.data
        return np.array([d.sensor(f"{leg}_touch").data[0] > CONTACT_THRESHOLD for leg in robot.LEGS])

    def actor_obs(self) -> np.ndarray:
        return self.obs  # (4, OBS_SIZE) per-leg, or (CENTRAL_OBS_SIZE,) central

    def critic_obs(self) -> np.ndarray:
        return self.obs if self.cfg.form == "central" else central_obs(self.obs)

    def step(self, action: np.ndarray) -> tuple[float, bool]:
        """Apply one action (raw policy output); returns the reward and whether the episode ended by a fall
        (time limits end episodes too, but are not falls)."""
        if self.cfg.form == "split":
            a = action.reshape(N_LEGS, ACT_SIZE + LEARNED)
            env_action = np.concatenate([STAND + ACTION_SCALE * a[:, :ACT_SIZE], a[:, ACT_SIZE:]], axis=1)
            targets = env_action[:, :ACT_SIZE].ravel()
        else:
            env_action = robot.STAND + ACTION_SCALE * action
            targets = env_action
        self.obs = self.env.step(env_action)
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


def train(cfg: Config, total_steps: int, progress=None) -> tuple[Agent, dict]:
    """Train for about `total_steps` environment steps; returns the agent and timing figures."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    agent = Agent(cfg.form)
    with torch.no_grad():
        agent.log_std.fill_(cfg.init_log_std)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    workers = [Worker(cfg, i) for i in range(cfg.n_envs)]
    t_env = t_update = 0.0
    steps_done = 0
    while steps_done < total_steps:
        scale = min(1.0, CURRICULUM_START + (1 - CURRICULUM_START) * steps_done / (CURRICULUM_FRACTION * total_steps))
        for w in workers:
            w.command_scale = scale
        a_obs, c_obs, acts, logps, rews, dones, vals = [], [], [], [], [], [], []
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
            r_step, d_step = [], []
            for i, w in enumerate(workers):
                r, fell = w.step(a[i].numpy())
                end = fell or w.steps >= EPISODE_STEPS
                r_step.append(r)
                d_step.append(1.0 if fell else 0.0)
                if end:
                    if not fell:  # time limit: bootstrap from the value of the last state
                        with torch.no_grad():
                            r_step[-1] += cfg.gamma * agent.critic(torch.from_numpy(w.critic_obs())).item()
                        d_step[-1] = 1.0
                    w.reset()
            a_obs.append(ao); c_obs.append(co); acts.append(a.numpy()); logps.append(lp.numpy())
            rews.append(r_step); dones.append(d_step); vals.append(v.numpy())
        with torch.no_grad():
            last_v = agent.critic(torch.from_numpy(np.stack([w.critic_obs() for w in workers]))).squeeze(-1).numpy()
        t1 = time.perf_counter()
        t_env += t1 - t0
        steps_done += cfg.rollout * cfg.n_envs

        rews, dones, vals = np.array(rews), np.array(dones), np.array(vals)
        adv = np.zeros_like(rews)
        gae = np.zeros(cfg.n_envs)
        for t in reversed(range(cfg.rollout)):
            nxt = last_v if t == cfg.rollout - 1 else vals[t + 1]
            delta = rews[t] + cfg.gamma * nxt * (1 - dones[t]) - vals[t]
            gae = delta + cfg.gamma * cfg.lam * (1 - dones[t]) * gae
            adv[t] = gae
        ret = adv + vals

        n = cfg.rollout * cfg.n_envs
        B_ao = torch.from_numpy(np.array(a_obs).reshape(n, *np.array(a_obs).shape[2:]))
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
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()
                losses.append((pg.item(), vl.item()))
        t_update += time.perf_counter() - t1
        entry = {"steps": steps_done, "command_scale": scale, "mean_reward": float(rews.mean()),
                 "falls": int(dones.sum()),
                 "policy_loss": float(np.mean([l[0] for l in losses])),
                 "value_loss": float(np.mean([l[1] for l in losses]))}
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
