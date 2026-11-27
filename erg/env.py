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
    gait phase (2)        cosine and sine of 2 pi times the phase of the leg's own gait clock (Issue #51)
    command (3)           forward speed m/s, sideways speed m/s, turning speed rad/s
    leg identifier (4)    one-hot, in the order of robot.LEGS
    messages (3 x 14)     one slot per other leg, in the order of robot.LEGS (Issue #35): the 13 values of the
                          last message received from that leg (zeros if none) and its age in control steps since
                          it was sent, capped at AGE_CAP (AGE_CAP if none). A leg that is not linked to this one
                          (link setting "none", or the diagonal leg with "neighbours") always reads zeros and
                          AGE_CAP, so the input has the same size in every link setting.

Action of one leg: target angles (rad) of its three joints, clipped to the joint ranges, optionally followed by
the LEARNED values it sends to the legs it is linked to (clipped to [-1, 1]; zeros if not given).

Gait clocks (Issue #51): each leg has its own clock, a phase in [0, 1) that starts at the trot offsets of
erg.gait.PHASE and advances each control step by CONTROL_PERIOD / erg.gait.PERIOD times (1 + FREQ_RANGE * f), where
f in [-1, 1] is the frequency correction given for that leg (zero if none). No clock reads another; the legs can
keep step only through their messages. The controllers use the phase to follow the base gait (erg.gait.leg_targets).

Messages (Issue #35): after each control step, each leg sends its gravity direction (the accelerometer reading
divided by its length, 3), angular velocity (gyroscope, 3), foot contact (1), its LEARNED values (4) and its gait
phase (cosine and sine, 2; Issue #51), encoded in
Dyne's wire format (erg.links) and carried by Dyne's virtual network (loss, delay, jitter; erg.links.Network). A
message sent after step k reaches its receiver at step k + 1 at the earliest.

Central mode (Issue #13), for comparison on the same body: one policy sees all sensors and commands all 12 joints,
with no messages between legs. Its observation (CENTRAL_OBS_SIZE values) is each leg's sensor values above (joints,
IMU, foot contact) and gait phase (18 per leg, in the order of robot.LEGS) followed by the command once; there is no
leg identifier. It has the same four independent gait clocks.
Its action is the 12 target angles in the order of robot.JOINT_NAMES. Only the controller differs between the modes:
the same joint commands give the same physical state bit for bit.

Servo model (Issue #70), optional: with `servo` set, the joint targets go through erg.servo (command delay,
proportional-derivative loop, torque limit falling with speed, friction) at every simulation step instead of the
model's position actuators; with `randomize_servo`, its delay, gains and friction are drawn at each reset from the
reset seed. Without it the environment is unchanged.
"""

import mujoco
import numpy as np

from erg import links as links_mod
from erg import exactmath, gait, robot
from erg import servo as servo_mod

N_LEGS = len(robot.LEGS)
SENSOR_SIZE = 3 + 3 + 3 + 3 + 3 + 1  # one leg's own sensor values
LEARNED = 4  # learned values in each message
PHASE_SIZE = 2  # cosine and sine of the gait phase
LEG_SIZE = SENSOR_SIZE + PHASE_SIZE  # one leg's own values: sensors and gait phase
MSG_VALUES = 3 + 3 + 1 + LEARNED + PHASE_SIZE  # values in one leg's message (erg.links.N_VALUES)
FREQ_RANGE = 0.5  # a gait clock runs between 0.5 and 1.5 times the base frequency 1 / erg.gait.PERIOD
AGE_CAP = 10  # control steps (0.2 s); the age of a message, and the age read when no message has arrived
MSG_SLOT = MSG_VALUES + 1
OBS_SIZE = LEG_SIZE + 3 + N_LEGS + (N_LEGS - 1) * MSG_SLOT
ACT_SIZE = len(robot.JOINTS_PER_LEG)
CENTRAL_OBS_SIZE = N_LEGS * LEG_SIZE + 3
CENTRAL_ACT_SIZE = N_LEGS * ACT_SIZE

# Foot force above which the microswitch is taken as pressed. An estimate, to be set from the chosen switch.
CONTACT_THRESHOLD = 0.5  # N

# Half-width of the uniform random offset added to the standing joint angles at reset.
RESET_JOINT_NOISE = 0.05  # rad

# Walking targets (design record docs/design/2026-11-25-walking-targets.md, Issue #27).
# Commands are drawn uniformly in [-max, +max] for each of forward speed, sideways speed and turning speed.
# Forward: near what the servos allow. Standing, the leg (hip flexion shaft to the sole) is about 0.16 m long, so
# walking at 0.23 m/s turns the hip at about 1.4 rad/s while the foot is on the ground (the servo gives about
# 2.8 rad/s at half its stall torque) and at up to about 4.3 rad/s to bring the foot forward in the air when three
# feet are down three quarters of the time (no-load speed 5.5 rad/s). tests/test_env.py checks these margins.
COMMAND_MAX = np.array([0.23, 0.12, 0.8])  # m/s, m/s, rad/s
STANCE_FRACTION = 0.75  # slowest walk assumed for the swing-speed margin: each foot on the ground 3/4 of the time
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
    def __init__(self, central: bool = False, links: str = "neighbours", loss: float = 0.0, lat_us: int = 0,
                 jitter_us: int = 0, wire: bool = True, servo: "servo_mod.ServoParams | None" = None,
                 randomize_servo: bool = False):
        """central=False: one controller per leg (observations (4, OBS_SIZE), actions (4, 3) or (4, 3 + LEARNED)),
        with messages between legs as set by `links` ("none", "neighbours", "all"), `loss`, `lat_us` and
        `jitter_us` (Dyne's virtual network; its seed is the reset seed). wire=True encodes and decodes every
        message in Dyne's wire format; wire=False passes the values directly, which is faster for training and
        gives the same observations bit for bit (tests/test_env_messages.py checks it).
        central=True: one policy for the whole body (observation (CENTRAL_OBS_SIZE,), action (12,)), no messages.
        servo: parameters of erg.servo's servo model (None: the model's position actuators); randomize_servo: draw
        the servo's delay, gains and friction at each reset from the reset seed."""
        self.central = central
        self.net_setting = (links, loss, lat_us, jitter_us)
        self.network = None if central else links_mod.Network(links, loss, lat_us, jitter_us)
        self.wire = links_mod.Links() if (wire and not central) else None
        self.step_count = 0
        self.learned = np.zeros((N_LEGS, LEARNED))
        self.phase = np.array([gait.PHASE[leg] for leg in robot.LEGS])
        self._inbox = []  # (delivery step, source, destination, message bytes)
        self._vals = np.zeros((N_LEGS, N_LEGS, MSG_VALUES))  # newest values received, [receiver, sender]
        self._sent = np.full((N_LEGS, N_LEGS), -1, dtype=np.int64)  # step at which they were sent; -1: none
        self.model = robot.load()
        self._lo = self.model.actuator_ctrlrange[:, 0].reshape(N_LEGS, ACT_SIZE)
        self._hi = self.model.actuator_ctrlrange[:, 1].reshape(N_LEGS, ACT_SIZE)
        self.servo = servo_mod.Servo(self.model, servo) if servo is not None else None
        self.randomize_servo = randomize_servo
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
        if self.servo is not None:  # its own stream from the same seed, so the draws above do not change
            srng = np.random.Generator(np.random.PCG64([seed, 70])) if self.randomize_servo else None
            self.servo.reset(self.data, robot.STAND, srng)
            mujoco.mj_forward(self.model, self.data)
        self.command = drawn if command is None else np.asarray(command, dtype=np.float64).copy()
        self.step_count = 0
        self.learned[:] = 0.0
        self.phase = np.array([gait.PHASE[leg] for leg in robot.LEGS])
        self._inbox = []
        self._vals[:] = 0.0
        self._sent[:] = -1
        if self.network is not None:
            self.network.reset(seed)
        return self.observe()

    def step(self, actions, freq=None) -> np.ndarray:
        """Apply the joint targets (shape (4, 3), or (12,) in central mode) for one control period; returns the
        observations. In the per-leg mode the actions may be (4, 3 + LEARNED): joint targets, then the learned
        values each leg sends. `freq` (4 values, clipped to [-1, 1]; zeros if None) corrects each leg's gait clock
        frequency for this step."""
        f = np.zeros(N_LEGS) if freq is None else np.clip(np.asarray(freq, dtype=np.float64), -1.0, 1.0)
        a = np.asarray(actions, dtype=np.float64)
        if not self.central and a.shape == (N_LEGS, ACT_SIZE + LEARNED):
            self.learned = np.clip(a[:, ACT_SIZE:], -1.0, 1.0)
            a = a[:, :ACT_SIZE]
        elif not self.central:
            self.learned = np.zeros((N_LEGS, LEARNED))
        targets = np.clip(a.reshape(N_LEGS, ACT_SIZE), self._lo, self._hi)
        if self.servo is None:
            self.data.ctrl[:] = targets.ravel()
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.data)
        else:
            self.servo.command(targets.ravel())
            for _ in range(self.substeps):
                self.servo.apply(self.data)
                mujoco.mj_step(self.model, self.data)
            self.servo.refresh(self.data)  # the load sensors read the torque for the state reached
        # mj_step computes the sensors before it integrates; recompute them for the state reached.
        mujoco.mj_forward(self.model, self.data)
        self.step_count += 1
        self.phase = (self.phase + robot.CONTROL_PERIOD / gait.PERIOD * (1.0 + FREQ_RANGE * f)) % 1.0
        if self.network is not None:
            self._exchange(self.step_count)
        return self.observe()

    def message_values(self) -> np.ndarray:
        """What each leg sends now, shape (4, MSG_VALUES): gravity direction, angular velocity, foot contact,
        learned values, gait phase."""
        sd = self.data.sensordata
        out = np.empty((N_LEGS, MSG_VALUES))
        for i, leg in enumerate(robot.LEGS):
            acc = sd[self._sensor[leg, "acc"]]
            norm = np.linalg.norm(acc)
            gravity = acc / norm if norm > 0 else np.zeros(3)
            touch = 1.0 if sd[self._sensor[leg, "touch"]][0] > CONTACT_THRESHOLD else 0.0
            out[i] = np.concatenate([gravity, sd[self._sensor[leg, "gyro"]], [touch], self.learned[i], self.phase_values()[i]])
        return out

    def _exchange(self, t: int):
        """At control step t (counted from 1 after reset): first deliver the messages due by t (sent at t - 1 or
        earlier), then send this step's messages through the wire format and the virtual network; they arrive at
        t + 1 at the earliest."""
        due = [m for m in self._inbox if m[0] <= t]
        self._inbox = [m for m in self._inbox if m[0] > t]
        for _, _, dst, msg in sorted(due, key=lambda m: (m[0], m[1], m[2])):
            sender, sent, v = msg
            if sent > self._sent[dst, sender]:  # keep the newest message from each sender
                self._vals[dst, sender] = v
                self._sent[dst, sender] = sent
        period_us = round(robot.CONTROL_PERIOD * 1e6)
        values = self.message_values()
        if self.wire is not None:  # through Dyne's wire format: what a receiver gets is the decoded message
            sent = []
            for i in range(N_LEGS):
                sender, step, _, v = self.wire.decode(self.wire.encode(i, t, t * period_us, values[i]))
                sent.append((sender, step, v))
        else:
            sent = [(i, t, values[i].copy()) for i in range(N_LEGS)]
        for src, dst, at in self.network.step(t):
            if at is not None:
                self._inbox.append((at, src, dst, sent[src]))

    def message_slots(self) -> np.ndarray:
        """Each leg's message slots, shape (4, 3, MSG_SLOT): for each other leg in the order of robot.LEGS, the
        newest values received and their age in control steps (capped at AGE_CAP; zeros and AGE_CAP if none)."""
        out = np.zeros((N_LEGS, N_LEGS - 1, MSG_SLOT))
        for r in range(N_LEGS):
            for j, s_ in enumerate(x for x in range(N_LEGS) if x != r):
                if self._sent[r, s_] < 0:
                    out[r, j, MSG_VALUES] = AGE_CAP
                else:
                    out[r, j, :MSG_VALUES] = self._vals[r, s_]
                    out[r, j, MSG_VALUES] = min(AGE_CAP, self.step_count - self._sent[r, s_])
        return out

    def phase_values(self) -> np.ndarray:
        """Cosine and sine of each leg's gait phase, shape (4, 2) (erg.exactmath: the same bits in the firmware)."""
        out = np.empty((N_LEGS, PHASE_SIZE))
        for i, p in enumerate(self.phase):
            s, c = exactmath.sincos(2.0 * exactmath.PI * float(p))
            out[i] = (c, s)
        return out

    def observe(self) -> np.ndarray:
        """Observations of all legs, shape (4, OBS_SIZE); in central mode, shape (CENTRAL_OBS_SIZE,)."""
        legs = self._observe_legs()
        if self.central:
            return np.concatenate([legs[:, :LEG_SIZE].ravel(), self.command])
        return legs

    def _observe_legs(self) -> np.ndarray:
        sd = self.data.sensordata
        slots = self.message_slots()
        obs = np.empty((N_LEGS, OBS_SIZE))
        phase = self.phase_values()
        for i, leg in enumerate(robot.LEGS):
            touch = sd[self._sensor[leg, "touch"]][0]
            obs[i] = np.concatenate([
                sd[self._sensor[leg, "pos"]],
                sd[self._sensor[leg, "vel"]],
                sd[self._sensor[leg, "load"]] / robot.SERVO_STALL_TORQUE,
                sd[self._sensor[leg, "acc"]],
                sd[self._sensor[leg, "gyro"]],
                [1.0 if touch > CONTACT_THRESHOLD else 0.0],
                phase[i],
                self.command,
                np.eye(N_LEGS)[i],
                slots[i].ravel(),
            ])
        return obs
