"""A position servo model applied at every simulation step (Issue #70).

docs/design/2026-11-27-redesign.md (control architecture, item 4) asks for a servo model with delay before training;
docs/design/2026-11-27-servo-model.md describes this one, its parameters and the bench plan that will replace them.

The model of one joint, evaluated before each MuJoCo step of `timestep` seconds:

1. Delay: the commanded position reaches the servo's control loop `delay` seconds later, rounded to whole simulation
   steps (a ring buffer). A delay of 50 ms is two and a half control periods of 20 ms.
2. Position loop (proportional-derivative): u = kp * (target - q) - kd * dq.
3. Drive limit: u is clipped to +-stall_torque (the motor voltage saturates).
4. Back electromotive force: the motor's torque falls by (stall_torque / no_load_speed) * dq. MuJoCo applies it as
   the joint's damping, integrated implicitly (stable at 2 ms steps).
5. Friction: Coulomb friction (MuJoCo's frictionloss) and the rotor's reflected inertia (armature) on the joint.

The torque on the joint is tau = clip(u, +-stall) - (stall / no_load_speed) * dq. In the direction of motion it is at
most stall * (1 - |dq| / no_load_speed): the limit falls linearly from the stall torque at rest to zero at the
no-load speed, the same for both directions (`drive_window`). Against the motion (braking) the motor can give up to
stall * (1 + |dq| / no_load_speed), as a voltage-limited DC motor does; clipping the braking torque to the same line
would let a leg pushed by the ground run away beyond the no-load speed with no torque at all.

This is the same structure as the position actuator of models/quadruped.xml (kp 10 N*m/rad, forcerange +-stall,
joint damping stall / no_load_speed), plus the delay, the derivative gain, and parameters that can be drawn at random
for training. The model's file is not changed: `Servo` turns the loaded model's position actuators into torque
actuators in memory, and writes each step's drive torque u into data.ctrl. The joint load sensors (actuatorfrc) then
read u, as they read the position actuator's force before.
"""

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from erg import design


@dataclass(frozen=True)
class ServoParams:
    """Parameters of one servo model; every joint uses the same servo. Sources in the comments; all values are
    unverified for erg's 12 V servo (Waveshare ST3215) until the bench identification of
    docs/design/2026-11-27-servo-model.md."""

    # Seller's listing of the 12 V servo (30 kg*cm, 0.222 s per 60 degrees), as in erg/design.py; unverified.
    stall_torque: float = design.SERVO_STALL_TORQUE  # N*m
    no_load_speed: float = design.SERVO_NO_LOAD_SPEED  # rad/s
    # Command delay: bus write, the servo's control loop and the position reading. Assumption, unverified: one bus
    # transfer and one servo loop, about 10 ms. Open Duck Mini v2 trains with 0 to 3 control steps of 20 ms (0-60 ms)
    # of action delay (Open_Duck_Playground, open_duck_mini_v2/joystick.py, noise_config.action_max_delay).
    delay: float = 0.010  # s
    # Proportional gain of the position loop. Open Duck Mini v2's MuJoCo model of the STS3215, exported from the BAM
    # identification (Open_Duck_Playground, open_duck_mini_v2/xmls/open_duck_mini_v2.xml: kp 13.37, kv 0). That is
    # the 7.4 V STS3215 at the firmware's default gains; unverified for the 12 V servo.
    kp: float = 13.37  # N*m/rad
    kd: float = 0.0  # N*m*s/rad, the same source (kv 0): the back electromotive force gives the damping
    # Back electromotive force as joint damping (N*m*s/rad); None: stall_torque / no_load_speed (0.623 for the values
    # above; Open Duck Mini v2's identified damping is 0.56).
    damping: float | None = None
    # Coulomb friction and reflected rotor inertia, the same source (frictionloss 0.068, armature 0.027); unverified.
    friction: float = 0.068  # N*m
    armature: float = 0.027  # kg*m^2
    # Training randomisation (assumptions, to be narrowed by the bench data): delay drawn uniformly in this range,
    # kp, kd and friction each multiplied by a factor drawn uniformly in [1 - gain_spread, 1 + gain_spread].
    delay_range: tuple[float, float] = (0.0, 0.040)  # s
    gain_spread: float = 0.2

    @classmethod
    def position_actuator(cls) -> "ServoParams":
        """The position actuator of models/quadruped.xml (erg/design.py) with no delay: reproduces the model."""
        # the model's joint damping is stall / no-load speed written with 4 significant digits (erg/design.py)
        damping = float(f"{design.SERVO_STALL_TORQUE / design.SERVO_NO_LOAD_SPEED:.4g}")
        return cls(delay=0.0, kp=10.0, kd=0.0, damping=damping, friction=0.05, armature=0.01, delay_range=(0.0, 0.0),
                   gain_spread=0.0)


def back_emf(p: ServoParams) -> float:
    """The back electromotive force's damping (N*m*s/rad)."""
    return p.damping if p.damping is not None else p.stall_torque / p.no_load_speed


def drive_window(p: ServoParams, dq):
    """Lowest and highest joint torque the servo can give at joint speed dq (N*m): the drive's +-stall torque minus
    the back electromotive force. In the direction of motion the magnitude is at most
    stall * (1 - |dq| / no_load_speed)."""
    b = back_emf(p)
    dq = np.asarray(dq, dtype=np.float64)
    return -p.stall_torque - b * dq, p.stall_torque - b * dq


class Servo:
    """The servo model on all actuated joints of a loaded MuJoCo model. Converts the model's position actuators into
    torque actuators (in memory) when created; then, each simulation step: `command` (when the controller sends new
    targets), `apply` (just before mujoco.mj_step)."""

    def __init__(self, model: mujoco.MjModel, params: ServoParams | None = None):
        self.base = params if params is not None else ServoParams()
        self.p = self.base
        self.model = model
        self.dt = float(model.opt.timestep)
        self.target_range = model.actuator_ctrlrange.copy()  # the joint target ranges of the position actuators
        joints = model.actuator_trnid[:, 0]
        self.qadr = model.jnt_qposadr[joints].copy()
        self.dadr = model.jnt_dofadr[joints].copy()
        nu = model.nu
        model.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_NONE
        model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_gainprm[:] = 0.0
        model.actuator_gainprm[:, 0] = 1.0
        model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_biasprm[:] = 0.0
        model.actuator_ctrllimited[:] = 0
        model.actuator_forcelimited[:] = 1
        self.max_delay_steps = max(self._steps(self.base.delay), self._steps(self.base.delay_range[1]))
        self.buf = np.zeros((self.max_delay_steps + 1, nu))
        self.head = 0
        self.cmd = np.zeros(nu)
        self.delayed = np.zeros(nu)
        self._set(self.base)

    def _steps(self, seconds: float) -> int:
        return int(round(seconds / self.dt))

    def _set(self, p: ServoParams) -> None:
        self.p = p
        self.delay_steps = self._steps(p.delay)
        if self.delay_steps > self.max_delay_steps:
            raise ValueError(f"delay {p.delay} s is longer than the buffer ({self.max_delay_steps} steps)")
        m = self.model
        m.actuator_forcerange[:] = (-p.stall_torque, p.stall_torque)
        m.dof_damping[self.dadr] = back_emf(p)
        m.dof_frictionloss[self.dadr] = p.friction
        m.dof_armature[self.dadr] = p.armature

    def reset(self, data: mujoco.MjData, targets, rng: np.random.Generator | None = None) -> None:
        """Start holding `targets` (no command in flight), with the nominal parameters, or with parameters drawn from
        `rng` (delay, kp, kd, friction; see ServoParams). Call after the joint positions are set."""
        p = self.base
        if rng is not None:
            s = p.gain_spread
            f = rng.uniform(1.0 - s, 1.0 + s, size=3)
            p = replace(p, delay=float(rng.uniform(*p.delay_range)), kp=p.kp * f[0], kd=p.kd * f[1],
                        friction=p.friction * f[2])
        self._set(p)
        t = np.clip(np.asarray(targets, dtype=np.float64), self.target_range[:, 0], self.target_range[:, 1])
        self.cmd[:] = t
        self.buf[:] = t
        self.head = 0
        self.delayed[:] = t
        self.refresh(data)

    def command(self, targets) -> None:
        """New joint targets (rad), held until the next command; clipped to the joint target ranges."""
        self.cmd[:] = np.clip(np.asarray(targets, dtype=np.float64), self.target_range[:, 0], self.target_range[:, 1])

    def apply(self, data: mujoco.MjData) -> np.ndarray:
        """Advance the delay line by one simulation step and write the drive torque into data.ctrl; call just before
        each mujoco.mj_step. Returns the drive torque u (N*m)."""
        n = len(self.buf)
        self.head = (self.head + 1) % n
        self.buf[self.head] = self.cmd
        self.delayed[:] = self.buf[(self.head - self.delay_steps) % n]
        return self.refresh(data)

    def refresh(self, data: mujoco.MjData) -> np.ndarray:
        """Write the drive torque for the current state and the target now in the loop, without advancing the delay
        (for mujoco.mj_forward after a control period, so that the load sensors read the current torque)."""
        q = data.qpos[self.qadr]
        dq = data.qvel[self.dadr]
        # The same order of operations as MuJoCo's position actuator (gain * ctrl + bias1 * q + bias2 * dq), so that
        # ServoParams.position_actuator() gives the same bits.
        u = self.p.kp * self.delayed + ((-self.p.kp) * q + (-self.p.kd) * dq)
        u = np.clip(u, -self.p.stall_torque, self.p.stall_torque)
        data.ctrl[:] = u
        return u


def load(params: ServoParams | None = None) -> tuple[mujoco.MjModel, Servo]:
    """The robot's model (models/quadruped.xml) with the servo model on its 12 joints."""
    from erg import robot

    model = robot.load()
    return model, Servo(model, params)
