"""The parts list hardware/bom.csv (Issue #23).

Pass criteria:
- every row has the columns of the list, a stage of 1 or 2, a positive whole quantity, and a known role;
- without spares, stages 1 and 2 together give what the MuJoCo model needs: one servo per actuator (12),
  and per leg one microcontroller, one IMU, one foot switch and one servo bus adapter (4 each);
- stage 1 is exactly one leg (3 servos and one of each per-leg part), with at least one spare servo.
"""

import csv
from collections import Counter
from pathlib import Path

import mujoco

from erg import robot

BOM_PATH = Path(__file__).resolve().parent.parent / "hardware" / "bom.csv"

COLUMNS = [
    "stage", "part", "role", "quantity", "spare", "reference_price", "currency", "tax_included",
    "source", "second_source", "checked",
]
PER_LEG_ROLES = ("microcontroller", "imu", "foot_switch", "bus_adapter")
ROLES = ("servo",) + PER_LEG_ROLES


def read_bom():
    with open(BOM_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        return list(reader)


def counts(rows, stages, spare):
    c = Counter()
    for r in rows:
        if int(r["stage"]) in stages and (r["spare"] == "yes") == spare:
            c[r["role"]] += int(r["quantity"])
    return c


def test_rows_are_well_formed():
    rows = read_bom()
    assert rows
    for r in rows:
        assert r["stage"] in ("1", "2")
        assert r["role"] in ROLES
        assert int(r["quantity"]) > 0
        assert r["spare"] in ("yes", "no")
        assert r["tax_included"] in ("yes", "no")
        float(r["reference_price"])


def test_whole_robot_matches_the_model():
    m = robot.load()
    legs = len(robot.LEGS)
    imu_sites = [s for s in range(m.nsite) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s).endswith("_imu")]
    touch = [s for s in range(m.nsensor) if m.sensor_type[s] == mujoco.mjtSensor.mjSENS_TOUCH]
    assert len(imu_sites) == legs and len(touch) == legs

    c = counts(read_bom(), stages=(1, 2), spare=False)
    assert c["servo"] == m.nu == 12
    for role in PER_LEG_ROLES:
        assert c[role] == legs, role


def test_stage_1_is_one_leg_with_a_spare_servo():
    rows = read_bom()
    c = counts(rows, stages=(1,), spare=False)
    assert c["servo"] == len(robot.JOINTS_PER_LEG)
    for role in PER_LEG_ROLES:
        assert c[role] == 1, role
    assert counts(rows, stages=(1,), spare=True)["servo"] >= 1
