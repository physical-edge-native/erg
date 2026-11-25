"""The robot as an environment with one controller per leg (Issue #11).

Each leg's observation holds only what that leg's microcontroller can read on the robot, plus the command and the
leg's identifier (the four legs share one policy). Messages between legs are added in Issue #12.

Observation of one leg (OBS_SIZE values, in this order):

    joint positions (3)   rad, hip_abd, hip_flex, knee
    joint speeds (3)      rad/s
    joint loads (3)       servo torque / stall torque, in [-1, 1]
    accelerometer (3)     m/s^2, in the torso frame, gravity included (reads about +9.81 on z at rest)
    gyroscope (3)         rad/s, in the torso frame
    foot contact (1)      1.0 if the touch sensor reads more than CONTACT_THRESHOLD, else 0.0
    command (3)           forward speed m/s, sideways speed m/s, turning speed rad/s
    leg identifier (4)    one-hot, in the order of robot.LEGS

Action of one leg: target angles (rad) of its three joints, clipped to the joint ranges.
"""

import mujoco
import numpy as np

from erg import robot

N_LEGS = len(robot.LEGS)
OBS_SIZE = 3 + 3 + 3 + 3 + 3 + 1 + 3 + N_LEGS
ACT_SIZE = len(robot.JOINTS_PER_LEG)

# Foot force above which the microswitch is taken as pressed. An estimate, to be set from the chosen switch.
CONTACT_THRESHOLD = 0.5  # N

# Half-width of the uniform random offset added to the standing joint angles at reset.
RESET_JOINT_NOISE = 0.05  # rad

# Walking targets (design record docs/design/2026-11-25-walking-targets.md, Issue #27).
# Commands are drawn uniformly in [-max, +max] for each of forward speed, sideways speed and turning speed.
# The values are provisional until the maintainer chooses among the options of Issue #27.
COMMAND_MAX = np.array([0.23, 0.12, 0.8])  # m/s, m/s, rad/s
# A commanded speed counts as reached after REACH_SECONDS of walking without a fall with the mean speed within
# REACH_TOLERANCE (relative) of the command.
REACH_SECONDS = 10.0
REACH_TOLERANCE = 0.2
# Uneven test ground, not seen in training: blocks from 0 to this height, laid out from a seed.
TERRAIN_MAX_HEIGHT = 0.02  # m


def draw_command(rng: np.random.Generator) -> np.ndarray:
    """A command drawn uniformly inside COMMAND_MAX, in both directions."""
    return rng.uniform(-COMMAND_MAX, COMMAND_MAX)


class Env:
    def __init__(self):
        self.model = robot.load()
        self.data = mujoco.MjData(self.model)
        self.substeps = round(robot.CONTROL_PERIOD / self.model.opt.timestep)
        self.command = np.zeros(3)
        self._sensor = {}
        for leg in robot.LEGS:
            for kind in ("pos", "vel", "load"):
                self._sensor[leg, kind] = np.concatenate(
                    [self._slice(f"{leg}_{j}_{kind}") for j in robot.JOINTS_PER_LEG]
                )
            for kind in ("acc", "gyro", "touch"):
                self._sensor[leg, kind] = self._slice(f"{leg}_{kind}")
        self._lo = self.model.actuator_ctrlrange[:, 0].reshape(N_LEGS, ACT_SIZE)
        self._hi = self.model.actuator_ctrlrange[:, 1].reshape(N_LEGS, ACT_SIZE)

    def _slice(self, name):
        s = self.model.sensor(name)
        return np.arange(s.adr[0], s.adr[0] + s.dim[0])

    def reset(self, seed: int, command=None) -> np.ndarray:
        """Standing posture with a small random joint offset drawn from `seed`, and the given command or, if none
        is given, a command drawn from the same seed (draw_command); returns the observations."""
        rng = np.random.Generator(np.random.PCG64(seed))
        robot.reset_standing(self.model, self.data)
        self.data.qpos[7:] += rng.uniform(-RESET_JOINT_NOISE, RESET_JOINT_NOISE, size=robot.STAND.size)
        mujoco.mj_forward(self.model, self.data)
        drawn = draw_command(rng)
        self.command = drawn if command is None else np.asarray(command, dtype=np.float64).copy()
        return self.observe()

    def step(self, actions) -> np.ndarray:
        """Apply each leg's joint targets (shape (4, 3)) for one control period; returns the observations."""
        targets = np.clip(np.asarray(actions, dtype=np.float64).reshape(N_LEGS, ACT_SIZE), self._lo, self._hi)
        self.data.ctrl[:] = targets.ravel()
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        # mj_step computes the sensors before it integrates; recompute them for the state reached.
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self) -> np.ndarray:
        """Observations of all legs, shape (4, OBS_SIZE)."""
        sd = self.data.sensordata
        obs = np.empty((N_LEGS, OBS_SIZE))
        for i, leg in enumerate(robot.LEGS):
            touch = sd[self._sensor[leg, "touch"]][0]
            obs[i] = np.concatenate([
                sd[self._sensor[leg, "pos"]],
                sd[self._sensor[leg, "vel"]],
                sd[self._sensor[leg, "load"]] / robot.SERVO_STALL_TORQUE,
                sd[self._sensor[leg, "acc"]],
                sd[self._sensor[leg, "gyro"]],
                [1.0 if touch > CONTACT_THRESHOLD else 0.0],
                self.command,
                np.eye(N_LEGS)[i],
            ])
        return obs
