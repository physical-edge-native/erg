"""The leg firmware's parts that run without the hardware (Issue #54), checked on the PC.

Pass criteria:
- firmware/leg-esp32/main/obs_layout.h is what scripts/gen_firmware_layout.py writes from erg/env.py now, so the
  firmware builds each leg's observation in the training environment's order and size;
- the servo packets made by the firmware's C code (sts.c) equal, byte for byte, the packets built here in Python
  from the protocol's definition (header, length, checksum, little-endian values), including the ping packet of
  Feetech's documentation (FF FF 01 02 01 FB); replies are parsed to the same values, and replies with a wrong
  header, id, length or checksum are rejected;
- the firmware's conversions (servo steps to joint angle, speed and load; joint angle to servo goal with clipping
  and rounding; IMU steps to m/s^2 and rad/s with the board's mounting) and its observation equal the same
  definitions in Python bit for bit;
- its arithmetic-only sine, cosine, arctangent and arccosine (xmath.c) equal erg.exactmath bit for bit, and so do
  its gait clock, gait phase values and base gait (erg.env, erg.gait) and its joint targets (erg.ppo: base gait at
  the leg's phase with the commanded forward speed's stride, plus the scaled corrections).

The C code is compiled with the C compiler on the PATH (cc, gcc or clang), with floating-point contraction off so
that no multiply-add is fused (Python, and the ESP32-S3's software double precision, round every operation). The
test is skipped, saying so, if no C compiler is found.
"""

import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from erg import env, exactmath, gait, ppo, robot

ROOT = Path(__file__).resolve().parent.parent
FW = ROOT / "firmware" / "leg-esp32"
RAD_PER_STEP = 2.0 * 3.141592653589793 / 4096
ACC_PER_STEP = 0.122e-3 * 9.80665
GYRO_PER_STEP = 17.5e-3 * 3.141592653589793 / 180.0


def test_layout_header_is_current():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_firmware_layout.py"), "--check"], cwd=ROOT)
    assert r.returncode == 0, "run scripts/gen_firmware_layout.py"


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler on the PATH")
    exe = tmp_path_factory.mktemp("fw") / ("host_check.exe" if sys.platform == "win32" else "host_check")
    src = [FW / "host_test" / "host_check.c", FW / "main" / "sts.c", FW / "main" / "leg.c", FW / "main" / "xmath.c",
           FW / "main" / "base_ctrl.c"]
    subprocess.run([cc, "-std=c11", "-O2", "-ffp-contract=off", "-Wall", "-Werror", "-o", str(exe),
                    *map(str, src), "-lm"], check=True)

    def run(lines):
        out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
        return out.stdout.splitlines()

    return run


def h(x: float) -> str:
    return struct.pack(">d", x).hex()


def doubles(line: str) -> list[float]:
    return [struct.unpack(">d", bytes.fromhex(t))[0] for t in line.split()]


def bits(xs) -> list[str]:
    return [h(float(x)) for x in xs]


def packet(id_, inst, params) -> bytes:
    body = [id_, len(params) + 2, inst, *params]
    return bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF])


def goal_params(pos):
    return [pos & 0xFF, pos >> 8, 0, 0, 0, 0]


def test_packets(host):
    cases = [("ping 1", bytes.fromhex("ffff010201fb"))]
    cases += [(f"ping {i}", packet(i, 0x01, [])) for i in (0, 7, 253)]
    cases += [(f"read {i} 56 6", packet(i, 0x02, [0x38, 6])) for i in (1, 2, 3)]
    cases += [(f"goal {i} {p}", packet(i, 0x03, [0x2A, *goal_params(p)])) for i, p in ((1, 0), (2, 2048), (3, 4095))]
    ids, pos = [1, 2, 3], [100, 2048, 4000]
    sync = [0x2A, 6] + [v for i, p in zip(ids, pos) for v in (i, *goal_params(p))]
    cases.append(("sync 3 1 100 2 2048 3 4000", packet(0xFE, 0x83, sync)))
    out = host([c for c, _ in cases])
    for (c, want), got in zip(cases, out):
        assert bytes.fromhex(got) == want, c


def sign_magnitude(v, bit):
    return -(v & ((1 << bit) - 1)) if v & (1 << bit) else v & ((1 << bit) - 1)


def test_replies(host):
    def reply(id_, err, pos, spd, load):
        data = [pos & 0xFF, pos >> 8, spd & 0xFF, spd >> 8, load & 0xFF, load >> 8]
        return packet(id_, err, data)  # same layout: the error byte takes the instruction's place

    raws = [(1, 0, 2048, 0, 0), (2, 0, 4095, 0x8000 | 300, 0x400 | 512), (3, 0x20, 17, 1200, 999)]
    lines = [f"parse {r[0]} {reply(*r).hex()}" for r in raws]
    good = reply(1, 0, 10, 20, 30)
    bad = [good[:1] + b"\x00" + good[2:], good, good[:-1] + bytes([good[-1] ^ 1]), good[:-2]]
    lines += [f"parse 1 {bad[0].hex()}", f"parse 2 {bad[1].hex()}", f"parse 1 {bad[2].hex()}", f"parse 1 {bad[3].hex()}"]
    out = host(lines)
    for r, got in zip(raws, out):
        want = [0, r[2], sign_magnitude(r[3], 15), sign_magnitude(r[4], 10), r[1]]
        assert list(map(int, got.split())) == want
    assert [int(x.split()[0]) for x in out[3:]] == [-1, -2, -4, -3]


def test_conversions(host):
    rng = np.random.Generator(np.random.PCG64(54))
    lines, want = [], []
    for _ in range(50):
        zero, sign = int(rng.integers(0, 4096)), int(rng.choice([-1, 1]))
        p, s, l = int(rng.integers(0, 4096)), int(rng.integers(-3000, 3000)), int(rng.integers(-1000, 1001))
        lines += [f"angle {zero} {sign} {p}", f"speed {zero} {sign} {s}", f"load {zero} {sign} {l}"]
        want += [[sign * ((p - zero) * RAD_PER_STEP)], [sign * (s * RAD_PER_STEP)], [sign * (l / 1000.0)]]
        lo, hi = -float(rng.uniform(0.1, 2.5)), float(rng.uniform(0.1, 2.5))
        a = float(rng.uniform(-3, 3))
        lines.append(f"goalangle {zero} {sign} {h(lo)} {h(hi)} {h(a)}")
        c = min(max(a, lo), hi)
        steps = min(max(math.floor(zero + sign * (c / RAD_PER_STEP) + 0.5), 0), 4095)
        want.append(steps)
    out = host(lines)
    for line, got, w in zip(lines, out, want):
        if isinstance(w, int):
            assert int(got) == w, line
        else:
            assert got.split() == bits(w), line


def test_imu(host):
    rng = np.random.Generator(np.random.PCG64(4))
    lines, want = [], []
    for axis, sign in [((0, 1, 2), (1, 1, 1)), ((1, 0, 2), (-1, 1, 1)), ((2, 1, 0), (1, -1, -1))]:
        a, g = rng.integers(-32768, 32768, 3), rng.integers(-32768, 32768, 3)
        lines.append("imu " + " ".join(map(str, [*axis, *sign, *a, *g])))
        acc = [sign[i] * (int(a[axis[i]]) * ACC_PER_STEP) for i in range(3)]
        gyro = [sign[i] * (int(g[axis[i]]) * GYRO_PER_STEP) for i in range(3)]
        want.append(bits(acc + gyro))
    for got, w in zip(host(lines), want):
        assert got.split() == w


def test_observation_and_targets(host):
    rng = np.random.Generator(np.random.PCG64(11))
    joints = [(2048, 1, 2100, -40, 100), (1000, -1, 1500, 300, -250), (3000, 1, 2900, 0, 999)]
    acc, gyro, command = rng.normal(size=3), rng.normal(size=3), rng.uniform(-0.2, 0.2, 3)
    phase = float(rng.uniform(0, 1))
    e = env.Env(wire=False)
    e.phase = np.full(env.N_LEGS, phase)
    phase_values = list(e.phase_values()[0])
    for leg in range(env.N_LEGS):
        for contact in (0, 1):
            line = f"obs {leg} {contact} " + " ".join(" ".join(map(str, j)) for j in joints)
            line += " " + " ".join(bits([*acc, *gyro, phase, *command]))
            got = host([line])[0].split()
            want = [sign * ((p - zero) * RAD_PER_STEP) for zero, sign, p, _, _ in joints]
            want += [sign * (s * RAD_PER_STEP) for _, sign, _, s, _ in joints]
            want += [sign * (l / 1000.0) for _, sign, _, _, l in joints]
            want += [*acc, *gyro, float(contact), *phase_values, *command, *np.eye(env.N_LEGS)[leg]]
            slots = np.zeros((env.N_LEGS - 1, env.MSG_SLOT))
            slots[:, env.MSG_VALUES] = env.AGE_CAP
            want += list(slots.ravel())
            assert len(got) == env.OBS_SIZE
            assert got == bits(want)
    for _ in range(20):
        a = rng.normal(size=env.ACT_SIZE + env.LEARNED + 1)
        p, forward = float(rng.uniform(0, 1)), float(rng.uniform(-0.23, 0.23))
        got = host(["targets " + " ".join(bits([*a, p, forward]))])[0].split()
        base = ppo.base_targets(np.array([p]), forward)[0]
        assert got == bits(base + ppo.ACTION_SCALE * a[: env.ACT_SIZE])


def test_exact_math_clock_and_base_gait(host):
    rng = np.random.Generator(np.random.PCG64(7))
    lines, want = [], []
    for x in [*rng.uniform(-20, 20, 50), 0.0, -0.0, 1e-300, *(k * exactmath.HALF_PI for k in range(-8, 9))]:
        lines.append(f"sincos {h(float(x))}")
        want.append(bits(exactmath.sincos(float(x))))
    for y, x in [*rng.normal(size=(50, 2)), (0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (-0.0, -1.0)]:
        lines.append(f"atan2 {h(float(y))} {h(float(x))}")
        want.append(bits([exactmath.atan2(float(y), float(x))]))
    for c in [*rng.uniform(-1, 1, 50), -1.0, 1.0, 0.0]:
        lines.append(f"acos {h(float(c))}")
        want.append(bits([exactmath.acos(float(c))]))
    e = env.Env(wire=False)
    for _ in range(50):
        p, f = float(rng.uniform(0, 1)), float(rng.uniform(-1.5, 1.5))
        lines.append(f"clock {h(p)} {h(f)}")
        nxt = (np.array([p]) + robot.CONTROL_PERIOD / gait.PERIOD * (1.0 + env.FREQ_RANGE * np.clip(f, -1.0, 1.0))) % 1.0
        want.append(bits([nxt[0]]))
        lines.append(f"phasevals {h(p)}")
        e.phase = np.full(env.N_LEGS, p)
        want.append(bits(e.phase_values()[0]))
        stride = float(rng.uniform(-0.04, 0.04))
        lines.append(f"base {h(p)} {h(stride)}")
        want.append(bits(gait.leg_targets(p, stride)))
    for line, got, w in zip(lines, host(lines), want):
        assert got.split() == w, line


def test_knee_torque_limit(host):
    """The knee torque limit (erg.knee_limit, Issue #81): limit and register value bit for bit over a grid of knee
    angles across the whole range and beyond; the write rule; the torque limit register's packet."""
    from erg import knee_limit

    rng = np.random.Generator(np.random.PCG64(81))
    qs = [*np.linspace(-3.2, 0.1, 331), *rng.uniform(-0.3, -0.19, 100), -0.2, -0.23, -2.6, 0.0, -0.0]
    lines = [f"kneelimit {h(float(q))}" for q in qs]
    for q, got in zip(qs, host(lines)):
        t, v = got.split()
        assert t == h(knee_limit.torque_limit(float(q))) and int(v) == knee_limit.register_value(float(q)), q
    cases = [(-1, 1000), (1000, 1000), (1000, 999), (869, 870), (869, 878), (869, 879), (995, 1000), (997, 990)]
    got = host([f"kneewrite {a} {b}" for a, b in cases])
    assert [int(g) for g in got] == [int(knee_limit.should_write(a, b)) for a, b in cases]
    assert host(["syncu16 48 1 3 869"])[0] == packet(0xFE, 0x83, [0x30, 2, 3, 869 & 0xFF, 869 >> 8]).hex()


def _base_controller_records(command, seed, steps, adjust, loss=0.0):
    """Run the per-leg base controllers (erg.base_controller) in the environment with messages between neighbours,
    with the policy's adjustments drawn at random and held POLICY_PERIODS control periods (or none); returns, per leg,
    the inputs (observation row, action, whether adjusted) and the Python outputs."""
    from erg import base_controller as bc

    e = env.Env(links="neighbours", loss=loss, wire=False)
    obs = e.reset(seed, command)
    ctrls = [bc.LegController(leg) for leg in robot.LEGS]
    rng = np.random.Generator(np.random.PCG64([seed, 85]))
    rec = [[] for _ in robot.LEGS]
    a = np.zeros((env.N_LEGS, bc.N_ADJUST))
    for k in range(steps):
        if k % ppo.POLICY_PERIODS == 0:
            a = rng.normal(0.0, 0.8, (env.N_LEGS, bc.N_ADJUST))  # some beyond [-1, 1], to check the clipping
        outs = []
        for i, (c, leg) in enumerate(zip(ctrls, robot.LEGS)):
            adj = bc.adjust_from_action(a[i]) if adjust else None
            o = c.step(bc.leg_input(obs[i], leg), adj)
            rec[i].append((obs[i].copy(), a[i].copy(), adjust, o))
            outs.append(o)
        actions = np.array([list(o.targets) + list(o.message) for o in outs])
        obs = e.step(actions, freq=[o.freq for o in outs])
    return rec


@pytest.mark.parametrize("command, seed, adjust, loss", [
    ((0.1, 0.0, 0.0), 0, False, 0.0), ((0.15, -0.05, -0.3), 1, True, 0.0), ((0.0, 0.05, 0.3), 2, True, 0.0),
    ((0.0, 0.0, 0.0), 3, True, 0.3)])
def test_base_controller_and_its_adjustments(host, command, seed, adjust, loss):
    """The firmware's base controller (base_ctrl.c) equals erg.base_controller bit for bit over 1.6 s of walking per
    leg, with the neighbours' messages (and, with loss, missing and aged messages) and the policy's adjustments."""
    from erg import base_controller as bc

    recs = _base_controller_records(command, seed, 80, adjust, loss)
    for i, rec in enumerate(recs):
        lines = [f"bcreset {i}"]
        lines += [f"bcstep {int(adj)} " + " ".join(bits([*a, *o])) for o, a, adj, _ in rec]
        out = host(lines)[1:]
        for k, ((_, _, _, py), got) in enumerate(zip(rec, out)):
            assert got.split() == bits([*py.targets, py.freq, *py.message]), (robot.LEGS[i], k)
    first_slot_age = env.LEG_SIZE + 3 + env.N_LEGS + env.MSG_VALUES
    assert sum(o[first_slot_age] < env.AGE_CAP for rec in recs for o, *_ in rec) > 0  # messages were heard
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(20):
        a = rng.normal(0.0, 1.0, bc.N_ADJUST)
        adj = bc.adjust_from_action(a)
        got = host(["bcadjust " + " ".join(bits(a))])[0].split()
        assert got == bits([adj.freq, adj.foot_dx, adj.foot_dy, adj.step_height, adj.place_gain, adj.att_gain])


@pytest.mark.parametrize("interpolate", [True, False], ids=["ramp", "hold"])
@pytest.mark.parametrize("command, seed", [((0.1, 0.0, 0.0), 4), ((0.12, 0.03, -0.2), 5)])
def test_policy_actions_ramped_over_the_policy_step(host, interpolate, command, seed):
    """The firmware's policy step (bc_policy_new, bc_policy_period) equals erg.ppo.period_actions and PolicyAdjust bit
    for bit, ramped (the generated default, POLICY_INTERPOLATE) and held, and so do the base controller's joint
    targets, frequency corrections and messages that follow, over 1.6 s of walking with the neighbours' messages
    (Issue #95)."""
    from erg import base_controller as bc

    e = env.Env(links="neighbours", wire=False)
    obs = e.reset(seed, command)
    ctrls = [bc.LegController(leg) for leg in robot.LEGS]
    rng = np.random.Generator(np.random.PCG64([seed, 95]))
    prev = np.zeros((env.N_LEGS, bc.N_ADJUST))
    raw, plan = prev, []
    rec = [[] for _ in robot.LEGS]
    for k in range(80):
        j = k % ppo.POLICY_PERIODS
        if j == 0:
            raw = rng.normal(0.0, 0.8, (env.N_LEGS, bc.N_ADJUST))  # some beyond [-1, 1]
            a = np.clip(raw, -1.0, 1.0)
            plan = ppo.period_actions(prev, a, interpolate)
            prev = a
        outs = []
        for i, (c, leg) in enumerate(zip(ctrls, robot.LEGS)):
            o = c.step(bc.leg_input(obs[i], leg), bc.adjust_from_action(plan[j][i]))
            rec[i].append((j, raw[i].copy(), obs[i].copy(), plan[j][i].copy(), o))
            outs.append(o)
        actions = np.array([list(o.targets) + list(o.message) for o in outs])
        obs = e.step(actions, freq=[o.freq for o in outs])
    for i, r in enumerate(rec):
        lines = [f"bcreset {i}"]
        lines += [f"bcpstep {int(interpolate)} {j} " + " ".join(bits([*rw, *o])) for j, rw, o, _, _ in r]
        for k, ((_, _, _, p, py), got) in enumerate(zip(r, host(lines)[1:])):
            assert got.split() == bits([*p, *py.targets, py.freq, *py.message]), (robot.LEGS[i], k)
    if interpolate:  # the ramp is used: the first period of a step is neither the old nor the new action
        assert any(not np.array_equal(p, rw.clip(-1, 1)) for j, rw, _, p, _ in rec[0] if j == 0)


def test_base_controller_rolling_foot_term_is_checked(host, monkeypatch):
    """The recorded walks exercise the rolling foot's term of the velocity estimate (Issue #98): the firmware, built
    with ROLL_RADIUS from erg.base_controller, differs from the Python controller with the term turned off."""
    from erg import base_controller as bc

    assert bc.ROLL_RADIUS > 0.0
    monkeypatch.setattr(bc, "ROLL_RADIUS", 0.0)
    rec = _base_controller_records((0.1, 0.0, 0.0), 0, 40, False)[0]
    lines = ["bcreset 0"] + ["bcstep 0 " + " ".join(bits([*a, *o])) for o, a, _, _ in rec]
    out = host(lines)[1:]
    assert any(got.split() != bits([*py.targets, py.freq, *py.message]) for (_, _, _, py), got in zip(rec, out))
