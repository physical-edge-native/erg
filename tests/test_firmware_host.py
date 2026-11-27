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
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from erg import env, exactmath, gait, ppo, robot

ROOT = Path(__file__).resolve().parent.parent
FW = ROOT / "firmware" / "leg-esp32"
RAD_PER_STEP = 2.0 * 3.141592653589793 / 4096
ACC_PER_STEP = 0.122e-3 * 9.80665
GYRO_PER_STEP = 17.5e-3 * 3.141592653589793 / 180.0


def test_layout_header_is_current():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_firmware_layout.py"), "--check"], cwd=ROOT)
    assert r.returncode == 0, "run scripts/gen_firmware_layout.py"


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
