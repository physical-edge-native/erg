"""Base controller (Issue #71): a walking controller on each leg, with no learning, under the later learned policy.

Design record: docs/design/2026-11-27-base-controller.md (with the pass conditions, written before the run). Each
leg's controller (LegController) gets, each control period, only a LegInput: its own sensor values and gait phase,
the command, and the message slots of its two neighbours on the ring front-left, front-right, hind-right, hind-left
(NEIGHBOURS; the ring of erg.links). It returns the leg's three joint targets, the frequency correction of its own
gait clock, and the values it sends to its neighbours.

Per leg:

- phase oscillator: the leg's gait clock of erg.env is the oscillator; the frequency correction pulls each neighbour
  towards half a period away (anti-phase coupling), so the trot is the stable solution on the ring;
- body velocity estimate: each period, the leg averages its estimate with its neighbours' (from their messages) and,
  while its foot is on the ground, corrects it by its own measurement: the foot's motion relative to the hip (forward
  kinematics of the leg's own joint angles) with the body's rotation (gyroscope) removed, through a low-pass filter;
- speed integral: the stance speed is the command plus the integral of the speed error, averaged with the
  neighbours' integrals in the same way. Without the averaging, the legs' integrals drift apart in opposite
  directions (left against right) and the legs push against each other (design record);
- foot placement by the capture-point rule (Raibert's rule), forward and sideways: touch-down at half the stance's
  travel ahead of the neutral point plus PLACE_GAIN (v_estimated - v_commanded), PLACE_GAIN = 1 / omega,
  omega = sqrt(g / h); turning adds the stance feet's motion for the commanded turning speed plus YAW_GAIN times its
  error read on the gyroscope;
- body attitude by leg-length differences: roll and pitch from the leg's IMU (gyroscope integrated, pulled towards
  the accelerometer's gravity direction with time constant ATT_TAU while the reading is near 1 g), each leg
  lengthens by ATT_GAIN times the height error of its own hip;
- swing-leg retraction: the foot moves back by RETRACT over the end of the swing;
- inverse kinematics: abduction for the sideways foot position, then the thigh and shank as in erg.gait.

The arithmetic is plain double-precision operations, square roots and erg.exactmath (no library trigonometry), on
Python floats, so that it can be ported to the leg firmware bit for bit as the base gait was (#54).

The policy may later adjust each leg through an Adjust (frequency, foot-placement residuals, step height, gains);
all zero by default. The controller outputs joint targets only: a servo model (#70) can replace the model's position
actuators without changing it.
"""

from dataclasses import dataclass, field
from math import sqrt  # correctly rounded, as sqrt in C

import numpy as np

from erg import design, env as env_mod, gait, robot
from erg.exactmath import PI, atan2, sin, sincos

LEGS = robot.LEGS
DT = robot.CONTROL_PERIOD
G = 9.81
# Neighbours of each leg on the ring FL - FR - HR - HL - FL (erg.links.Network "neighbours"; tests check it).
NEIGHBOURS = {"FL": ("FR", "HL"), "FR": ("FL", "HR"), "HL": ("FL", "HR"), "HR": ("FR", "HL")}

COM_HEIGHT = 0.142  # m, centre of mass above the ground, standing (erg.body_targets on body L1)
OMEGA = sqrt(G / COM_HEIGHT)  # rad/s, of the linear inverted pendulum
PLACE_GAIN = 1.0 / OMEGA  # s, capture-point (Raibert) gain, about 0.12 s

PERIOD = gait.PERIOD  # s, at zero frequency correction
STANCE = 0.5  # fraction of the period on the ground (trot)
STEP_HEIGHT = gait.STEP_HEIGHT  # m, as the hand-made trot
RETRACT = 0.003  # m, swing-leg retraction at the end of the swing
RETRACT_START = 0.8  # fraction of the swing at which the retraction starts
SWING_RAMP = 0.25  # fraction of the swing's forward move over which its speed rises (and falls)
COUPLING = 0.5  # frequency correction per unit of phase error (sine of the error) summed over the two neighbours
SPEED_INTEGRAL = 1.0  # 1/s, integral of the speed error into the stance speed
INTEGRAL_LIMIT = 0.15  # m/s, bound on the integral term
VEL_TAU = 0.15  # s, low-pass filter of the velocity estimate
ATT_TAU = 1.0  # s, complementary filter of roll and pitch (gyroscope below, accelerometer above this time)
ACC_TRUST = 0.2  # the accelerometer corrects the attitude only when its reading is within this fraction of g squared
ATT_GAIN = 0.5  # leg-length change per metre of hip height error
YAW_GAIN = 0.5  # extra turning speed of the stance feet per unit of turning-speed error (gyroscope)
MAX_REACH = 0.05  # m, bound on the foot's forward and sideways offset from its neutral point
MAX_ATT = 0.03  # m, bound on the attitude's leg-length change

# Indices in one leg's observation of erg.env (OBS_SIZE values).
_POS, _ACC, _GYRO, _TOUCH = slice(0, 3), slice(9, 12), slice(12, 15), 15
_PHASE, _CMD = slice(16, 18), slice(18, 21)
_SLOTS = env_mod.LEG_SIZE + 3 + env_mod.N_LEGS
# Indices in a message slot (erg.env.message_values, then the age).
_M_LEARNED, _M_PHASE, _M_AGE = slice(7, 11), slice(11, 13), env_mod.MSG_VALUES


@dataclass(frozen=True)
class LegInput:
    """All that one leg's controller may read in one control period."""

    leg: str
    joint_pos: tuple  # rad, abduction, hip flexion, knee
    acc: tuple  # m/s^2, the leg's IMU, torso frame
    gyro: tuple  # rad/s
    contact: bool
    phase_cs: tuple  # cosine and sine of the leg's own gait phase
    command: tuple  # forward m/s, sideways m/s, turning rad/s
    neighbours: dict  # neighbour leg name -> its message slot (MSG_VALUES values and the age), as a tuple


def leg_input(obs_row, leg: str) -> LegInput:
    """Cut a leg's LegInput out of its row of erg.env's per-leg observation, keeping only its neighbours' slots
    (the slot of the diagonal leg, if any, is dropped here)."""
    o = [float(v) for v in obs_row]
    others = [x for x in LEGS if x != leg]
    slots = {}
    for j, other in enumerate(others):
        if other in NEIGHBOURS[leg]:
            a = _SLOTS + j * env_mod.MSG_SLOT
            slots[other] = tuple(o[a:a + env_mod.MSG_SLOT])
    return LegInput(leg=leg, joint_pos=tuple(o[_POS]), acc=tuple(o[_ACC]), gyro=tuple(o[_GYRO]),
                    contact=o[_TOUCH] > 0.5, phase_cs=tuple(o[_PHASE]), command=tuple(o[_CMD]), neighbours=slots)


@dataclass
class Adjust:
    """What a policy may change in one leg's controller; all zero by default (the base controller alone)."""

    freq: float = 0.0  # added to the gait clock's frequency correction (erg.env: 1 + 0.5 f times the base frequency)
    foot_dx: float = 0.0  # m, foot-placement residual, forward
    foot_dy: float = 0.0  # m, sideways
    step_height: float = 0.0  # m, added to STEP_HEIGHT
    place_gain: float = 0.0  # relative change of PLACE_GAIN
    att_gain: float = 0.0  # relative change of ATT_GAIN


# The policy's adjustment of one leg (Issue #82, docs/design/2026-11-27-learning-setup.md): N_ADJUST values, each
# clipped to [-1, 1] and multiplied by ADJUST_SCALE, in the order of Adjust's fields. A zero action is Adjust().
N_ADJUST = 6
ADJUST_SCALE = (0.3, 0.02, 0.02, 0.006, 0.5, 0.5)  # freq, m, m, m, relative, relative


def adjust_from_action(a) -> Adjust:
    """One leg's Adjust from the policy's N_ADJUST outputs (clipped to [-1, 1], then scaled by ADJUST_SCALE)."""
    v = [_clip(float(x), -1.0, 1.0) * k for x, k in zip(a, ADJUST_SCALE, strict=True)]
    return Adjust(freq=v[0], foot_dx=v[1], foot_dy=v[2], step_height=v[3], place_gain=v[4], att_gain=v[5])


@dataclass
class LegOutput:
    targets: tuple  # rad, abduction, hip flexion, knee
    freq: float  # frequency correction of the leg's own gait clock, in [-1, 1]
    message: tuple  # erg.env.LEARNED values sent to the neighbours: velocity estimate vx, vy, speed integral x, y


# --- Leg kinematics (exactmath) ---------------------------------------------------------------------------------

def leg_frame(leg: str) -> tuple:
    """(sx, sy, hip x, hip y): signs and the abduction shaft's place in the torso frame."""
    sx, sy = design.leg_sign(leg)
    return sx, sy, sx * design.HIP_X, sy * design.HIP_Y


def foot_position(leg: str, q) -> tuple:
    """Foot centre relative to the abduction shaft (torso frame axes, standing), from the joint angles."""
    _, sy, _, _ = leg_frame(leg)
    a, q1, q2 = q
    s1, c1 = sincos(q1)
    s12, c12 = sincos(q1 + q2)
    x = -design.THIGH * s1 - design.SHANK * s12  # in the leg's plane, from the hip flexion shaft
    zp = -design.THIGH * c1 - design.SHANK * c12
    lat = sy * design.LEG_OFFSET
    sa, ca = sincos(a)
    return x, lat * ca - zp * sa, lat * sa + zp * ca


def leg_ik(leg: str, x: float, y: float, z: float) -> tuple:
    """Joint angles that put the foot centre at (x, y, z) from the abduction shaft (x along the hip flexion
    plane's forward axis)."""
    _, sy, _, _ = leg_frame(leg)
    lat = sy * design.LEG_OFFSET
    d2 = y * y + z * z - lat * lat
    d = sqrt(d2) if d2 > 1e-6 else 1e-3
    a = atan2(z, y) - atan2(-d, lat)
    flex, knee = gait.leg_angles(x, -d)
    return a, flex, knee


def _blend(u: float) -> float:
    """Smooth step from 0 to 1 over u in [0, 1] (half a cosine)."""
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return 0.5 - 0.5 * sincos(PI * u)[1]


def _glide(u: float, r: float) -> float:
    """Smooth step from 0 to 1 over u in [0, 1] whose speed rises over the first fraction r (half a cosine), stays
    constant, and falls over the last fraction r: a peak speed of 1 / (1 - r) times the mean, against pi / 2 for
    _blend, so the swing asks less speed of the servos."""
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    if r <= 0.0:
        return u
    if u > 0.5:
        return 1.0 - _glide(1.0 - u, r)
    if u < r:
        p = u / 2.0 - r / (2.0 * PI) * sin(PI * u / r)
    else:
        p = r / 2.0 + (u - r)
    return p / (1.0 - r)


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class LegController:
    """One leg's base controller. Its state is its own; it sees the rest of the robot only through LegInput."""

    leg: str
    vel: list = field(default_factory=lambda: [0.0, 0.0])  # filtered body velocity estimate, m/s
    integral: list = field(default_factory=lambda: [0.0, 0.0])  # m/s
    roll: float = 0.0
    pitch: float = 0.0
    prev_foot: tuple | None = None  # foot position at the previous period, if the foot was on the ground
    prev_s: float | None = None
    td: tuple = (0.0, 0.0)  # touch-down offset from the neutral point (forward, sideways)
    lo: tuple = (0.0, 0.0)  # lift-off offset
    last: tuple = (0.0, 0.0)  # offset planned at the previous period

    def step(self, inp: LegInput, adj: Adjust | None = None) -> LegOutput:
        if inp.leg != self.leg:
            raise ValueError(f"input of leg {inp.leg} given to the controller of leg {self.leg}")
        adj = adj or Adjust()
        sx, sy, hx, hy = leg_frame(self.leg)
        lat = sy * design.LEG_OFFSET
        vx_c, vy_c, wz_c = inp.command

        # Phase of the own clock, and anti-phase coupling to the neighbours (their phases advanced by the age).
        c_i, s_i = inp.phase_cs
        s = atan2(s_i, c_i) / (2.0 * PI)
        s = s + 1.0 if s < 0.0 else s
        couple = 0.0
        shared = []  # the neighbours' velocity estimates and speed integrals, from their messages
        for name in NEIGHBOURS[self.leg]:
            m = inp.neighbours.get(name)
            if m is None or m[_M_AGE] >= env_mod.AGE_CAP:
                continue
            sd, cd = sincos(2.0 * PI * DT / PERIOD * m[_M_AGE])
            c_j0, s_j0 = m[_M_PHASE]
            c_j, s_j = c_j0 * cd - s_j0 * sd, s_j0 * cd + c_j0 * sd
            couple += -(s_j * c_i - c_j * s_i)  # sin(2 pi (phase_j - phase_i - 1/2))
            shared.append(m[_M_LEARNED])
        freq = _clip(adj.freq + COUPLING * couple, -1.0, 1.0)

        # Velocity estimate and speed integral, averaged with the neighbours' (consensus on the ring, so that the
        # legs do not drift apart and push against each other), then corrected by the leg's own measurement.
        n = 1.0 + len(shared)
        vel = [(self.vel[0] + sum(v[0] for v in shared)) / n, (self.vel[1] + sum(v[1] for v in shared)) / n]
        integral = [(self.integral[0] + sum(v[2] for v in shared)) / n,
                    (self.integral[1] + sum(v[3] for v in shared)) / n]
        # Own stance velocity: the foot is fixed on the ground, the body moves against it.
        foot = foot_position(self.leg, inp.joint_pos)
        fx, fy, fz = foot[0] + hx, foot[1] + hy, foot[2]  # from the torso's origin
        wx, wy, wz = inp.gyro
        if inp.contact and self.prev_foot is not None:
            vx = -(fx - self.prev_foot[0]) / DT - (wy * fz - wz * fy)
            vy = -(fy - self.prev_foot[1]) / DT - (wz * fx - wx * fz)
            a = DT / (VEL_TAU + DT)
            vel = [vel[0] + a * (vx - vel[0]), vel[1] + a * (vy - vel[1])]
        self.prev_foot = (fx, fy) if inp.contact else None
        self.vel = vel
        ex, ey = vx_c - vel[0], vy_c - vel[1]
        self.integral = [_clip(integral[0] + SPEED_INTEGRAL * ex * DT, -INTEGRAL_LIMIT, INTEGRAL_LIMIT),
                         _clip(integral[1] + SPEED_INTEGRAL * ey * DT, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)]

        # Attitude from the IMU: complementary filter of roll and pitch.
        ax, ay, az = inp.acc
        self.roll = self.roll + wx * DT
        self.pitch = self.pitch + wy * DT
        a2 = ax * ax + ay * ay + az * az
        if (1.0 - ACC_TRUST) * G * G < a2 < (1.0 + ACC_TRUST) * G * G:  # trust gravity only near 1 g
            b = DT / (ATT_TAU + DT)
            self.roll = self.roll + b * (atan2(ay, az) - self.roll)
            self.pitch = self.pitch + b * (atan2(-ax, az) - self.pitch)
        hip_x, hip_y = hx, hy + lat  # the hip flexion shaft, from the torso's origin
        drop = hip_x * sin(self.pitch) - hip_y * sin(self.roll)  # how far this hip is below level
        att = _clip(ATT_GAIN * (1.0 + adj.att_gain) * drop, -MAX_ATT, MAX_ATT)

        # Stance speed of the foot (body frame) for the commanded motion, with speed feedback.
        turn = wz_c + YAW_GAIN * (wz_c - wz)
        hip_vx = vx_c + self.integral[0] - turn * hip_y
        hip_vy = vy_c + self.integral[1] + turn * hip_x
        t_stance = STANCE * PERIOD / (1.0 + env_mod.FREQ_RANGE * freq)
        k = PLACE_GAIN * (1.0 + adj.place_gain)
        # Capture-point placement: half the stance travel, plus the gain times the speed error.
        tdx = _clip(hip_vx * t_stance / 2.0 + k * (self.vel[0] - vx_c) + adj.foot_dx, -MAX_REACH, MAX_REACH)
        tdy = _clip(hip_vy * t_stance / 2.0 + k * (self.vel[1] - vy_c) + adj.foot_dy, -MAX_REACH, MAX_REACH)

        # Phase changes: record touch-down and lift-off offsets.
        if self.prev_s is not None:
            if s < STANCE <= self.prev_s:  # wrapped: swing ended
                self.td = self.last
            elif self.prev_s < STANCE <= s:  # stance ended
                self.lo = self.last
        self.prev_s = s

        lift = 0.0
        if s < STANCE:  # on the ground: back from the touch-down offset at the stance speed
            u = s / STANCE
            ox = self.td[0] - hip_vx * t_stance * u
            oy = self.td[1] - hip_vy * t_stance * u
        else:  # in the air: forward to the placement, lifted, with retraction at the end
            u = (s - STANCE) / (1.0 - STANCE)
            g1 = _glide(u / RETRACT_START, SWING_RAMP)
            g2 = _blend((u - RETRACT_START) / (1.0 - RETRACT_START))
            ox = self.lo[0] + (tdx + RETRACT - self.lo[0]) * g1 - RETRACT * g2
            oy = self.lo[1] + (tdy - self.lo[1]) * g1
            lift = (STEP_HEIGHT + adj.step_height) * sin(PI * u)
        self.last = (ox, oy)

        depth = design.STAND_DEPTH + att
        targets = leg_ik(self.leg, gait.FOOT_X + ox, lat + oy, -depth + lift)
        message = (self.vel[0], self.vel[1], self.integral[0], self.integral[1])
        return LegOutput(targets=targets, freq=freq, message=message)


# --- Running it in the environment and measuring it -------------------------------------------------------------

STANDING_HEIGHT = 0.15  # m, torso centre standing (design record)
FALL_TILT_DEG = 45.0
FALL_HEIGHT = 0.6 * STANDING_HEIGHT
PUSH_FORCE = 3.0  # N, sideways, on the torso's centre of mass
PUSH_SECONDS = 0.1  # 0.3 N s


@dataclass
class Result:
    forward: float  # m/s, mean over the window, in the body's heading frame
    sideways: float
    yaw_rate: float  # rad/s
    max_tilt_deg: float
    min_height: float  # m
    fell: bool
    torque_p95: float  # worst joint's 95th percentile of torque / stall torque over the window
    torque_max: float
    targets: np.ndarray  # joint targets of every period, (steps, 4, 3)


def _yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run(command, seed: int = 0, seconds: float = 20.0, window: float = 5.0, push: float = 0.0,
        push_time: float = 10.0, push_force: float = PUSH_FORCE, links: str = "neighbours",
        wire: bool = True, servo=None, randomize_servo: bool = False, adjust=None) -> Result:
    """Walk with the base controller for `seconds` from the standing posture (reset noise from `seed`), and measure
    over [window, seconds]. `push`: +1 or -1 for a sideways push to the left or right at `push_time`, 0 for none.
    `servo`, `randomize_servo`: erg.env's servo model (Issue #70). `adjust`: None for the base controller alone, or a
    callable (control step, observations) -> four Adjust, e.g. a learned policy (Issue #82)."""
    e = env_mod.Env(links=links, wire=wire, servo=servo, randomize_servo=randomize_servo)
    obs = e.reset(seed, command)
    ctrls = [LegController(leg) for leg in LEGS]
    torso = e.model.body("torso").id
    steps = round(seconds / DT)
    start = round(window / DT)
    push_steps = range(round(push_time / DT), round((push_time + PUSH_SECONDS) / DT))
    yaws, vels, torques, all_targets = [], [], [], []
    max_tilt, min_height = 0.0, np.inf
    for k in range(steps):
        adjs = adjust(k, obs) if adjust is not None else [None] * len(LEGS)
        outs = [c.step(leg_input(obs[i], leg), adjs[i]) for i, (c, leg) in enumerate(zip(ctrls, LEGS))]
        actions = np.array([list(o.targets) + list(o.message) for o in outs])
        all_targets.append(actions[:, :3].copy())
        e.data.xfrc_applied[torso] = 0.0
        if push and k in push_steps:
            yaw = _yaw(e.data.qpos[3:7])
            e.data.xfrc_applied[torso, :2] = push * push_force * np.array([-np.sin(yaw), np.cos(yaw)])
        obs = e.step(actions, freq=[o.freq for o in outs])
        yaw = _yaw(e.data.qpos[3:7])
        v = e.data.qvel[:2]
        c, s_ = np.cos(yaw), np.sin(yaw)
        yaws.append(yaw)
        vels.append((c * v[0] + s_ * v[1], -s_ * v[0] + c * v[1]))
        torques.append(np.abs(e.data.actuator_force) / robot.SERVO_STALL_TORQUE)
        up = e.data.xmat[torso].reshape(3, 3)[2, 2]
        max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(up, -1.0, 1.0)))))
        min_height = min(min_height, float(e.data.xpos[torso, 2]))
    e.data.xfrc_applied[:] = 0.0
    yaws = np.unwrap(np.array(yaws))
    vels = np.array(vels[start:])
    tq = np.array(torques[start:])
    return Result(
        forward=float(vels[:, 0].mean()), sideways=float(vels[:, 1].mean()),
        yaw_rate=float((yaws[-1] - yaws[start - 1]) / (seconds - window)),
        max_tilt_deg=max_tilt, min_height=min_height,
        fell=max_tilt > FALL_TILT_DEG or min_height < FALL_HEIGHT,
        torque_p95=float(np.percentile(tq, 95, axis=0).max()), torque_max=float(tq.max()),
        targets=np.array(all_targets))


# --- Pass conditions of the design record ---------------------------------------------------------------------

FORWARD_COMMANDS = [(0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (0.3, 0.0, 0.0)]
TURN_COMMANDS = [(0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.15, 0.0, 0.5)]
SIDEWAYS_COMMANDS = [(0.0, 0.1, 0.0), (0.0, -0.1, 0.0)]
FORWARD_TOL = 0.25
TURN_TOL = 0.5
SIDEWAYS_TOL = 0.5
DRIFT_SPEED = 0.05  # m/s
DRIFT_YAW = 0.1  # rad/s
TORQUE_P95 = 0.5
# Condition 6 as changed by the maintainer after the first run (design record): the limit applies at 0.1 m/s;
# 0.15 and 0.2 m/s are reported only. The first version applied it at 0.2 m/s and failed (0.72-0.73).
TORQUE_COMMAND = (0.1, 0.0, 0.0)
TORQUE_REPORTED = [(0.15, 0.0, 0.0), (0.2, 0.0, 0.0)]


def check(command, r: Result) -> list[str]:
    """The failed conditions (1-4) of the design record for one run of a command; empty if it passes."""
    vx, vy, wz = command
    bad = []
    if r.fell:
        bad.append("fell")
    if (vx, vy, wz) in FORWARD_COMMANDS:
        if abs(r.forward - vx) > FORWARD_TOL * vx:
            bad.append(f"forward {r.forward:.3f} not within {FORWARD_TOL:.0%} of {vx}")
        if abs(r.sideways) > DRIFT_SPEED:
            bad.append(f"sideways drift {r.sideways:.3f}")
        if abs(r.yaw_rate) > DRIFT_YAW:
            bad.append(f"yaw drift {r.yaw_rate:.3f}")
    if (vx, vy, wz) in TURN_COMMANDS and abs(r.yaw_rate - wz) > TURN_TOL * abs(wz):
        bad.append(f"yaw rate {r.yaw_rate:.3f} not within {TURN_TOL:.0%} of {wz}")
    if (vx, vy, wz) in SIDEWAYS_COMMANDS:
        if abs(r.sideways - vy) > SIDEWAYS_TOL * abs(vy):
            bad.append(f"sideways {r.sideways:.3f} not within {SIDEWAYS_TOL:.0%} of {vy}")
        if abs(r.yaw_rate) > DRIFT_YAW:
            bad.append(f"yaw drift {r.yaw_rate:.3f}")
    return bad

