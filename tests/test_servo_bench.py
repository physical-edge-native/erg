"""The servo bench script and model fit (Issue #88): scripts/servo_bench.py, erg/bench.py.

Pass criteria:
- dry run (`scripts/servo_bench.py dry-run`, no hardware): the bench runs the short schedule against a simulated
  servo with known parameters (erg.bench.DRY_RUN_TRUTH) through the same packets as the real servo, writes the CSV
  log, and the fit recovers the parameters within erg.bench.FIT_TOLERANCE: the delay exactly (whole simulation
  steps of 2 ms), kp, armature, stall torque and no-load speed within 8 %, friction within 10 %, kd within
  0.05 N*m*s/rad (it is barely identifiable; see docs/design/2026-11-27-servo-model.md). The script exits with 0;
- the simulated servo follows the protocol: it answers ping, read and write, takes sync writes, and ignores packets
  with a bad checksum or for another id;
- the bench switches the torque off when a safety limit is passed (temperature here) and stops with an error;
- the logged commands land on the simulated servo: the simulated rig with the truth reproduces the logged positions
  within 1.5 steps of the position reading (rms), the reading's rounding plus the 2 ms simulation grid.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from erg import bench, sts

ROOT = Path(__file__).resolve().parent.parent


def test_dry_run_recovers_the_known_parameters(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "servo_bench.py"), "dry-run", "--out-dir",
                        str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads((tmp_path / "servo_bench_dry_run.json").read_text(encoding="utf-8"))
    assert set(out["check"]) == set(bench.FIT_TOLERANCE)
    assert all(v["ok"] for v in out["check"].values()), out["check"]
    cols, rig, meta = bench.read_csv(tmp_path / "servo_bench_dry_run.csv")
    assert rig == bench.Rig() and meta["simulated"] == "True"
    assert len(cols["t_read"]) > 1000


def test_simulated_servo_follows_the_protocol():
    port = bench.SimServoPort(bench.DRY_RUN_TRUTH, servo_id=3)
    port.write(sts.ping(3))
    assert port.read(6) == sts.reply(3, 0, [])
    port.write(sts.ping(4))  # another servo
    assert port.read(6) == b""
    bad = bytearray(sts.ping(3))
    bad[-1] ^= 1
    port.write(bytes(bad))
    assert port.read(6) == b""
    port.write(sts.read(3, sts.ADDR_P_GAIN, 3))
    assert sts.parse_reply(port.read(9), 3, 3) == (0, bytes([32, 32, 0]))
    port.write(sts.write(3, sts.ADDR_TORQUE_ENABLE, [1]))
    assert port.read(6) == sts.reply(3, 0, []) and port.torque
    port.write(sts.sync_write_goal([1, 3], [2048, 2048 + 200]))  # no reply to a sync write
    assert port.read(6) == b""
    port.sleep(1.0)
    port.write(sts.read(3, sts.ADDR_PRESENT_POSITION, 6))
    s = sts.parse_state(port.read(12), 3)
    assert 2248 - 20 <= s.position <= 2248  # reached, sagging a little under the load's weight (proportional loop)


def test_bench_switches_the_torque_off_at_a_limit():
    port = bench.SimServoPort(bench.DRY_RUN_TRUTH)
    port.regs[sts.ADDR_PRESENT_TEMPERATURE] = 70
    with pytest.raises(bench.BenchError, match="temperature"):
        bench.Bench(port).run(bench.schedule("short"), log=lambda s: None)
    assert not port.torque


def test_logged_commands_reach_the_simulated_servo():
    rig = bench.Rig()
    port = bench.SimServoPort(bench.DRY_RUN_TRUTH, rig)
    rows = bench.Bench(port).run(bench.schedule("short"), log=lambda s: None)
    cols = {c: np.array([r[c] for r in rows]) for c in bench.COLUMNS if c != "segment"}
    q, _, _ = bench.simulate(bench.DRY_RUN_TRUTH, cols, rig)
    rms = float(np.sqrt(np.mean((q - cols["position"]) ** 2)))
    assert rms < 1.5 * bench.RAD_PER_STEP, rms
    assert np.abs(cols["position"]).max() > 0.5  # the steps were made
