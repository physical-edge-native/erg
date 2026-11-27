"""Servo bench: drive one servo through step and chirp tests, log its responses, and fit the servo model (Issue #88).

The plan is in docs/design/2026-11-27-servo-model.md ("Bench identification plan"); scripts/servo_bench.py is the
command line. The same code runs against the real servo (a serial port opened with pyserial) and against
`SimServoPort`, a simulated servo on a simulated bus, so that the whole path runs in CI without hardware.

The rig: one servo on a fixed stand turns an arm about a horizontal shaft; the arm hangs down at the zero angle, and
carries a known load at its end (`Rig`). Its MuJoCo model (`rig_mjcf`) has one hinge joint driven by erg.servo's
model, the same as each joint of the robot.

The fit (`fit`) simulates the rig with the logged commands at their logged times and adjusts the servo model's
parameters (delay, kp, kd, friction, armature, stall torque, no-load speed) by least squares on the logged position
and speed. The delay is searched over whole simulation steps; the rest by Levenberg-Marquardt with finite
differences.
"""

import csv
import math
import time
from dataclasses import asdict, dataclass, fields, replace

import mujoco
import numpy as np

from erg import sts
from erg.servo import Servo, ServoParams

RAD_PER_STEP = 2.0 * math.pi / sts.STEPS_PER_TURN
TIMESTEP = 0.002  # s, as the robot's model
GRAVITY = 9.81


@dataclass(frozen=True)
class Rig:
    """The bench rig: an arm of uniform mass and a load at its end, hanging down at the zero angle."""

    arm_mass: float = 0.030  # kg
    arm_length: float = 0.150  # m, shaft to the load's centre
    load_mass: float = 0.300  # kg; its inertia is comparable with the rotor's, so the fit can tell the two apart

    def inertia(self) -> float:
        """About the shaft (kg m^2), without the servo's own rotor (the armature, which is fitted)."""
        return self.arm_mass * self.arm_length**2 / 3 + self.load_mass * self.arm_length**2


def rig_mjcf(rig: Rig, limit: float = math.pi) -> str:
    m = rig.arm_mass + rig.load_mass
    c = (rig.arm_mass * rig.arm_length / 2 + rig.load_mass * rig.arm_length) / m  # centre of mass below the shaft
    iyc = rig.inertia() - m * c * c
    return f"""<mujoco model="servo_bench">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast" gravity="0 0 {-GRAVITY}"/>
  <worldbody>
    <body name="arm">
      <joint name="shaft" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {-c!r}" mass="{m!r}" diaginertia="{iyc!r} {iyc!r} 1e-9"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shaft" joint="shaft" kp="10" ctrlrange="{-limit!r} {limit!r}"/>
  </actuator>
</mujoco>
"""


class SimServoPort:
    """A simulated servo on a simulated serial bus, with the methods of a pyserial port that the bench uses (write,
    read, reset_input_buffer, close) and a virtual clock (`clock`, `sleep`). Each byte takes 10 bit times at `baud`;
    the servo model (erg.servo) runs on the rig in steps of TIMESTEP up to the clock.

    Register values: position in steps with 2048 at the zero angle; speed in steps per second; load as the drive
    torque in tenths of a percent of the stall torque; current from the load with STALL_CURRENT (an assumption of
    the simulation only); voltage 12.0 V; temperature 30 C. With `noise`, a random number of steps (normal, that
    standard deviation) is added to the position read."""

    STALL_CURRENT = 2.7  # A, assumed for the simulated current register only

    def __init__(self, params: ServoParams, rig: Rig = Rig(), servo_id: int = 1, baud: int = 1_000_000,
                 noise: float = 0.0, seed: int = 0, angle0: float = 0.0):
        self.params, self.rig, self.id, self.baud, self.noise = params, rig, servo_id, baud, noise
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.model = mujoco.MjModel.from_xml_string(rig_mjcf(rig))
        self.servo = Servo(self.model, params)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0] = angle0
        mujoco.mj_forward(self.model, self.data)
        self.servo.reset(self.data, [angle0])
        self.regs = bytearray(256)
        self.regs[sts.ADDR_P_GAIN:sts.ADDR_P_GAIN + 3] = bytes([32, 32, 0])  # Feetech's defaults
        self.regs[sts.ADDR_PRESENT_VOLTAGE] = 120
        self.regs[sts.ADDR_PRESENT_TEMPERATURE] = 30
        self.torque = False
        self.t = 0.0  # the bus's clock
        self.t_sim = 0.0
        self.out = bytearray()
        self.u = 0.0
        self.pending = []  # goal positions received (time, angle), taken by the first step that starts after them

    # --- clock ---
    def clock(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(0.0, dt)
        self._advance()

    def _advance(self) -> None:
        while self.t_sim + TIMESTEP <= self.t + 1e-12:
            while self.pending and self.pending[0][0] <= self.t_sim + 1e-12:
                self.servo.command([self.pending.pop(0)[1]])
            if self.torque:
                self.u = float(self.servo.apply(self.data)[0])
            else:
                self.servo.apply(self.data)
                self.u = 0.0
                self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)
            self.t_sim += TIMESTEP

    # --- the port ---
    def reset_input_buffer(self) -> None:
        self.out.clear()

    def close(self) -> None:
        pass

    def write(self, data: bytes) -> int:
        self.sleep(len(data) * 10 / self.baud)
        p = bytes(data)
        if len(p) < 6 or p[:2] != b"\xff\xff" or p[3] + 4 != len(p) or sts.checksum(p) != p[-1]:
            return len(data)  # a servo ignores a bad packet
        id_, inst, params = p[2], p[4], p[5:-1]
        if inst == sts.INST_SYNC_WRITE and id_ == sts.BROADCAST_ID:
            addr, n = params[0], params[1]
            for k in range(2, len(params), n + 1):
                if params[k] == self.id:
                    self._write_regs(addr, params[k + 1:k + 1 + n])
            return len(data)
        if id_ != self.id:
            return len(data)
        if inst == sts.INST_PING:
            self.out += sts.reply(self.id, 0, [])
        elif inst == sts.INST_READ:
            self._present()
            addr, n = params[0], params[1]
            self.out += sts.reply(self.id, 0, self.regs[addr:addr + n])
        elif inst == sts.INST_WRITE:
            self._write_regs(params[0], params[1:])
            self.out += sts.reply(self.id, 0, [])
        return len(data)

    def read(self, n: int) -> bytes:
        got = bytes(self.out[:n])
        del self.out[:n]
        self.sleep(len(got) * 10 / self.baud)
        return got

    def _write_regs(self, addr: int, data: bytes) -> None:
        self.regs[addr:addr + len(data)] = data
        if addr <= sts.ADDR_TORQUE_ENABLE < addr + len(data):
            self.torque = self.regs[sts.ADDR_TORQUE_ENABLE] != 0
        if addr <= sts.ADDR_GOAL_POSITION < addr + len(data):
            steps = self.regs[sts.ADDR_GOAL_POSITION] | self.regs[sts.ADDR_GOAL_POSITION + 1] << 8
            self.pending.append((self.t, (steps - 2048) * RAD_PER_STEP))

    def _present(self) -> None:
        dq = float(self.data.qvel[0])
        q = float(self.data.qpos[0]) + dq * (self.t - self.t_sim)  # the shaft keeps turning within the step
        pos = round(2048 + q / RAD_PER_STEP + (self.rng.normal(0.0, self.noise) if self.noise else 0.0))
        pos = min(max(pos, 0), 4095)
        load = round(self.u / self.params.stall_torque * 1000)
        cur = round(self.u / self.params.stall_torque * self.STALL_CURRENT / sts.CURRENT_PER_UNIT)
        a = sts.ADDR_PRESENT_POSITION
        r = self.regs
        r[a:a + 2] = pos.to_bytes(2, "little")
        r[a + 2:a + 4] = sts.to_sign_magnitude(round(dq / RAD_PER_STEP), 15).to_bytes(2, "little")
        r[a + 4:a + 6] = sts.to_sign_magnitude(load, 10).to_bytes(2, "little")
        r[sts.ADDR_PRESENT_CURRENT:sts.ADDR_PRESENT_CURRENT + 2] = sts.to_sign_magnitude(cur, 15).to_bytes(2, "little")


# --- test schedules (the plan's experiments) ---

@dataclass(frozen=True)
class Segment:
    name: str
    seconds: float
    kind: str  # "hold", "step", "chirp", "ramp"
    a: float = 0.0  # step size or chirp amplitude or ramp end (rad)
    f0: float = 0.0  # chirp start frequency (Hz) or ramp speed (rad/s)
    f1: float = 0.0  # chirp end frequency (Hz)

    def target(self, t: float) -> float:
        if self.kind == "step":
            return self.a
        if self.kind == "chirp":
            phase = self.f0 * t + (self.f1 - self.f0) * t * t / (2 * self.seconds)
            return self.a * math.sin(2 * math.pi * phase)
        if self.kind == "ramp":
            return math.copysign(min(abs(self.a), self.f0 * t), self.a)
        return 0.0


def schedule(kind: str = "full") -> list[Segment]:
    """The plan's tests around the zero angle. "full": delay (50 small steps), steps of 0.1-1.0 rad both ways,
    chirps 0.5-10 Hz at 0.05 and 0.2 rad, slow ramps both ways (about 2.5 minutes). "short": a few seconds of the
    same kinds, for the dry run and the tests."""
    s = [Segment("settle", 1.0, "hold")]
    if kind == "short":
        s += [Segment("delay", 0.25, "step", 0.05), Segment("delay", 0.25, "hold")]
        for a in (0.1, 0.6):
            s += [Segment("step", 0.4, "step", a), Segment("step", 0.4, "step", -a), Segment("step", 0.3, "hold")]
        s += [Segment("chirp", 2.0, "chirp", 0.1, 0.5, 5.0)]
        return s
    if kind != "full":
        raise ValueError(kind)
    for _ in range(50):
        s += [Segment("delay", 0.3, "step", 0.05), Segment("delay", 0.3, "hold")]
    for _ in range(2):
        for a in (0.1, 0.3, 0.6, 1.0):
            s += [Segment("step", 1.0, "step", a), Segment("step", 1.0, "step", -a), Segment("step", 1.0, "hold")]
    for a in (0.05, 0.2):
        s += [Segment("chirp", 20.0, "chirp", a, 0.5, 10.0), Segment("chirp", 1.0, "hold")]
    for end in (0.8, -0.8):
        s += [Segment("ramp", 9.0, "ramp", end, 0.1), Segment("ramp", 9.0, "ramp", end, 0.0),
              Segment("ramp", 1.0, "hold")]
    return s


def target_at(segments: list[Segment], t: float) -> tuple[float, str] | None:
    """Target (rad, relative to the zero angle) and segment name at time t; None after the end."""
    for s in segments:
        if t < s.seconds:
            if s.kind == "ramp" and s.f0 == 0.0:  # hold at the ramp's end
                return s.a, s.name
            return s.target(t), s.name
        t -= s.seconds
    return None


# --- running the bench ---

COLUMNS = ("t_write", "target", "t_read", "position", "speed", "load", "voltage", "temperature", "current", "error",
           "position_steps", "speed_steps", "load_raw", "segment")


@dataclass
class Limits:
    """Safety limits of a run: the run stops (torque off) when one is passed."""

    angle: float = 1.2  # rad from the zero angle: targets are clipped to it
    max_temperature: int = 65  # C
    min_voltage: float = 10.0  # V
    max_current: float = 3.0  # A


class BenchError(Exception):
    pass


class Bench:
    """One servo on a port (pyserial or SimServoPort). `clock` and `sleep` are the port's for the simulated servo
    and the PC's otherwise."""

    def __init__(self, port, servo_id: int = 1, zero: int = 2048, echo: bool = False, clock=None, sleep=None):
        self.port, self.id, self.zero, self.echo = port, servo_id, zero, echo
        self.clock = clock or getattr(port, "clock", None) or time.perf_counter
        self.sleep = sleep or getattr(port, "sleep", None) or time.sleep

    def _send(self, packet: bytes) -> None:
        self.port.reset_input_buffer()
        self.port.write(packet)
        if self.echo:  # adapters that hear their own transmission
            self.port.read(len(packet))

    def _ask(self, packet: bytes, n_reply: int) -> bytes:
        self._send(packet)
        return self.port.read(n_reply)

    def ping(self) -> None:
        err, _ = sts.parse_reply(self._ask(sts.ping(self.id), 6), self.id, 0)
        if err:
            raise BenchError(f"servo error bits {err:#04x}")

    def read(self, addr: int, n: int) -> bytes:
        return sts.parse_reply(self._ask(sts.read(self.id, addr, n), n + 6), self.id, n)[1]

    def write(self, addr: int, data) -> None:
        sts.parse_reply(self._ask(sts.write(self.id, addr, data), 6), self.id, 0)

    def state(self) -> tuple[sts.State, float]:
        """The present position to current, and the time the request was sent (when the servo takes its reading;
        the reply then takes its transfer time)."""
        self._send(sts.read(self.id, sts.ADDR_PRESENT_POSITION, sts.BENCH_BLOCK))
        t = self.clock()
        return sts.parse_bench_block(self.port.read(sts.BENCH_BLOCK + 6), self.id), t

    def goal(self, angle: float) -> float:
        """The robot's path: a sync write (no reply), as the leg firmware sends. Returns the angle sent (rounded to
        whole steps, as the firmware rounds)."""
        steps = min(max(math.floor(self.zero + angle / RAD_PER_STEP + 0.5), 0), 4095)
        self._send(sts.sync_write_goal([self.id], [steps]))
        return (steps - self.zero) * RAD_PER_STEP

    def run(self, segments: list[Segment], period: float = 0.005, limits: Limits = Limits(), log=print) -> list[dict]:
        """Enable the torque holding the present position, move to the zero angle, run the segments, and switch the
        torque off at the end or on any error. Returns the rows (COLUMNS)."""
        self.ping()
        gains = self.read(sts.ADDR_P_GAIN, 3)
        log(f"servo {self.id}: P {gains[0]} D {gains[1]} I {gains[2]}")
        rows = []
        try:
            here = (self.state()[0].position - self.zero) * RAD_PER_STEP
            self.goal(here)
            self.write(sts.ADDR_TORQUE_ENABLE, [1])
            t0 = self.clock()
            first = True
            while True:
                t = self.clock() - t0
                tg = target_at(segments, t)
                if tg is None:
                    break
                target = max(-limits.angle, min(limits.angle, tg[0]))
                if first:  # come from the present position to the zero angle slowly, during "settle"
                    target = here * max(0.0, 1.0 - t / 0.8)
                    first = tg[1] == "settle"
                target = self.goal(target)
                t_write = self.clock() - t0
                s, t_read = self.state()
                t_read -= t0
                row = {
                    "t_write": t_write, "target": target, "t_read": t_read,
                    "position": (s.position - self.zero) * RAD_PER_STEP, "speed": s.speed * RAD_PER_STEP,
                    "load": s.load / 1000.0, "voltage": s.voltage / 10.0, "temperature": s.temperature,
                    "current": s.current * sts.CURRENT_PER_UNIT, "error": s.error, "position_steps": s.position,
                    "speed_steps": s.speed, "load_raw": s.load, "segment": tg[1],
                }
                rows.append(row)
                if row["temperature"] > limits.max_temperature:
                    raise BenchError(f"temperature {row['temperature']} C")
                if row["voltage"] < limits.min_voltage:
                    raise BenchError(f"voltage {row['voltage']} V")
                if abs(row["current"]) > limits.max_current:
                    raise BenchError(f"current {row['current']:.2f} A")
                wait = (t_write - (self.clock() - t0)) + period
                self.sleep(max(0.0, wait))
        finally:
            try:
                self.write(sts.ADDR_TORQUE_ENABLE, [0])
            except Exception as e:  # noqa: BLE001 -- the torque must be tried off; the first error matters more
                log(f"could not switch the torque off: {e}")
        return rows


def write_csv(path, rows: list[dict], rig: Rig, meta: dict | None = None) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        for k, v in {**asdict(rig), **(meta or {})}.items():
            f.write(f"# {k}={v}\n")
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path) -> tuple[dict[str, np.ndarray], Rig, dict]:
    meta, lines = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition("=")
                meta[k] = v
            else:
                lines.append(line)
    rows = list(csv.DictReader(lines))
    cols = {c: np.array([float(r[c]) for r in rows]) for c in COLUMNS if c != "segment"}
    rig = Rig(**{f.name: float(meta[f.name]) for f in fields(Rig) if f.name in meta})
    return cols, rig, meta


# --- the fit ---

FITTED = ("kp", "kd", "friction", "armature", "stall_torque", "no_load_speed")
SIGMA_POSITION = 2e-3  # rad: about one step of the position reading
SIGMA_SPEED = 0.2  # rad/s


def simulate(params: ServoParams, log: dict[str, np.ndarray], rig: Rig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position, speed and drive torque / stall torque of the simulated rig at the logged reading times, with the
    logged commands applied from their logged write times. Starts at rest at the first logged position."""
    model = mujoco.MjModel.from_xml_string(rig_mjcf(rig))
    sv = Servo(model, params)
    d = mujoco.MjData(model)
    d.qpos[0] = log["position"][0]
    mujoco.mj_forward(model, d)
    sv.reset(d, [log["target"][0]])
    t_end = float(log["t_read"][-1])
    n = int(t_end / TIMESTEP) + 2
    t_w, tgt = log["t_write"], log["target"]
    q, dq, u = np.empty(n + 1), np.empty(n + 1), np.empty(n + 1)
    q[0], dq[0], u[0] = d.qpos[0], d.qvel[0], 0.0
    i = 0
    for k in range(n):
        t = k * TIMESTEP
        while i < len(t_w) and t_w[i] <= t:
            sv.command([tgt[i]])
            i += 1
        u[k + 1] = sv.apply(d)[0]
        mujoco.mj_step(model, d)
        q[k + 1], dq[k + 1] = d.qpos[0], d.qvel[0]
    ts = np.arange(n + 1) * TIMESTEP
    tr = log["t_read"]
    return np.interp(tr, ts, q), np.interp(tr, ts, dq), np.interp(tr, ts, u) / params.stall_torque


def _residual(params, log, rig):
    q, dq, _ = simulate(params, log, rig)
    return np.concatenate([(q - log["position"]) / SIGMA_POSITION, (dq - log["speed"]) / SIGMA_SPEED])


def _params(base: ServoParams, x: np.ndarray, delay: float) -> ServoParams:
    v = {}
    for name, xi in zip(FITTED, x):
        v[name] = max(0.0, float(xi) * 0.05) if name == "kd" else getattr(base, name) * math.exp(float(xi))
    return replace(base, delay=delay, delay_range=(0.0, delay), gain_spread=0.0, **v)


def _cost(base, x, delay, logs):
    r = np.concatenate([_residual(_params(base, x, delay), lg, rig) for lg, rig in logs])
    return float(r @ r), r


def _lm(base, x, delay, logs, iterations, log):
    cost, r = _cost(base, x, delay, logs)
    lam = 1e-2
    h = 1e-3
    for it in range(iterations):
        jac = np.empty((len(r), len(x)))
        for j in range(len(x)):
            xp = x.copy()
            xp[j] += h
            jac[:, j] = (_cost(base, xp, delay, logs)[1] - r) / h
        g = jac.T @ r
        a = jac.T @ jac
        improved = False
        for _ in range(8):
            step = np.linalg.solve(a + lam * np.diag(np.diag(a) + 1e-9), -g)
            xn = x + step
            cn, rn = _cost(base, xn, delay, logs)
            if cn < cost:
                x, r, rel = xn, rn, (cost - cn) / cost
                cost = cn
                lam = max(lam / 3, 1e-6)
                improved = True
                break
            lam *= 4
        log(f"  delay {delay * 1000:.0f} ms, iteration {it}: cost {cost:.4g}")
        if not improved or rel < 1e-4:
            break
    return x, cost


@dataclass
class FitResult:
    params: ServoParams
    cost: float
    rms_position: float  # rad
    rms_speed: float  # rad/s
    load_scale: float  # logged load / simulated drive torque fraction (1.0 if the load reading is calibrated)


def fit(logs: list[tuple[dict, Rig]], start: ServoParams = ServoParams(), max_delay: float = 0.030,
        iterations: int = 30, log=print) -> FitResult:
    """Fit the servo model to one or more logs (each with its rig): the delay by a search over whole simulation
    steps, the other parameters (FITTED) by Levenberg-Marquardt from `start`."""
    x = np.zeros(len(FITTED))
    x[FITTED.index("kd")] = start.kd / 0.05
    delays = [k * TIMESTEP for k in range(int(round(max_delay / TIMESTEP)) + 1)]
    costs = [_cost(start, x, dl, logs)[0] for dl in delays]
    delay = delays[int(np.argmin(costs))]
    log(f"delay search with the starting values: {delay * 1000:.0f} ms")
    x, cost = _lm(start, x, delay, logs, iterations, log)
    tried = {delay}
    while True:  # the delay again, with the fitted values
        near = [dl for dl in (delay - TIMESTEP, delay + TIMESTEP) if 0 <= dl <= max_delay + 1e-12 and dl not in tried]
        if not near:
            break
        tried.update(near)
        best = min(near, key=lambda dl: _cost(start, x, dl, logs)[0])
        x2, c2 = _lm(start, x, best, logs, iterations, log)
        if c2 >= cost:
            break
        x, cost, delay = x2, c2, best
    p = _params(start, x, delay)
    p = replace(p, delay_range=start.delay_range, gain_spread=start.gain_spread)
    eq, ev, num, den = [], [], 0.0, 0.0
    for lg, rig in logs:
        q, dq, u = simulate(p, lg, rig)
        eq.append(q - lg["position"])
        ev.append(dq - lg["speed"])
        num += float(u @ lg["load"])
        den += float(u @ u)
    return FitResult(p, cost, float(np.sqrt(np.mean(np.concatenate(eq) ** 2))),
                     float(np.sqrt(np.mean(np.concatenate(ev) ** 2))), num / den if den > 0 else float("nan"))


# --- the dry run: a simulated servo with known parameters (scripts/servo_bench.py dry-run, tests/test_servo_bench.py)

# Known parameters of the simulated servo: each differs from the fit's starting values (ServoParams()).
DRY_RUN_TRUTH = replace(ServoParams(), delay=0.014, kp=12.0, kd=0.03, friction=0.08, armature=0.024,
                        stall_torque=2.8, no_load_speed=5.0)

# How close the fit of the short schedule's simulated log must come to the known parameters (relative, except the
# delay in seconds and kd in N*m*s/rad). Grounds (docs/design/2026-11-27-servo-model.md): with the position read in
# whole steps (1.5e-3 rad), trials with random parameters recovered the delay exactly and kp, armature, stall torque
# and no-load speed within 4 %, friction within 3 %; kd is barely identifiable from a short log (in motion that does
# not saturate the drive it adds to the back electromotive force's damping) and came out between 0 and the truth.
FIT_TOLERANCE = {"delay": 1e-9, "kp": 0.08, "armature": 0.08, "stall_torque": 0.08, "no_load_speed": 0.08,
                 "friction": 0.10, "kd": 0.05}


def within_tolerance(fitted: ServoParams, truth: ServoParams) -> dict[str, tuple[float, float, bool]]:
    """For each parameter of FIT_TOLERANCE: the fitted value, the truth, and whether they agree."""
    out = {}
    for k, tol in FIT_TOLERANCE.items():
        a, b = float(getattr(fitted, k)), float(getattr(truth, k))
        err = abs(a - b) if k in ("delay", "kd") else abs(a / b - 1)
        out[k] = (a, b, err <= tol)
    return out
