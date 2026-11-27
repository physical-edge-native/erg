"""The parts list hardware/bom.csv (Issue #23).

Pass criteria:
- every row has the columns of the list, a stage of 1 or 2, a positive whole quantity, and a known role;
- without spares, stages 1 and 2 together give what the MuJoCo model needs: one servo per actuator (12),
  and per leg one microcontroller, one IMU, one foot switch and one servo bus adapter (4 each);
- stage 1 is exactly one leg (3 servos and one of each per-leg part), with at least one spare servo;
- the parts to run the robot are listed (Issue #31): one 5 V regulator per microcontroller, a bench power supply in
  stage 1, and at least one battery and a charger;
- the power parts match the servo voltage (Issue #60): the servos are the 12 V class, and the battery's cell count
  keeps its voltage, full (4.2 V per cell) and nearly empty (3.5 V per cell), within the servo's and the servo bus
  adapter's input range; the charger handles that cell count;
- rows of stage "option" (parts not bought unless decided, such as foot force sensors) are not counted.
"""

import re

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
PER_LEG_ROLES = ("microcontroller", "imu", "foot_switch", "bus_adapter", "regulator")
POWER_ROLES = ("power_supply", "battery", "charger", "small_parts")
OPTION_ROLES = ("foot_force",)
ROLES = ("servo",) + PER_LEG_ROLES + POWER_ROLES + OPTION_ROLES
STAGES = ("1", "2", "option")
# Input ranges in volts: the Waveshare ST3215 (12 V class) servo 6-12.6 V; the Waveshare Bus Servo Adapter (A) is
# labelled 9-12.6 V. The robot's supply must lie in both.
SUPPLY_RANGE = (9.0, 12.6)
CELL_FULL, CELL_LOW = 4.2, 3.5


def read_bom():
    with open(BOM_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        return list(reader)


def counts(rows, stages, spare):
    c = Counter()
    for r in rows:
        if r["stage"] in {str(x) for x in stages} and (r["spare"] == "yes") == spare:
            c[r["role"]] += int(r["quantity"])
    return c


def test_rows_are_well_formed():
    rows = read_bom()
    assert rows
    for r in rows:
        assert r["stage"] in STAGES
        assert (r["stage"] == "option") == (r["role"] in OPTION_ROLES), r["part"]
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


def test_parts_to_run_the_robot_are_listed():
    rows = read_bom()
    whole = counts(rows, stages=(1, 2), spare=False)
    assert whole["regulator"] == whole["microcontroller"]
    assert counts(rows, stages=(1,), spare=False)["power_supply"] == 1
    assert whole["battery"] >= 1 and whole["charger"] >= 1


def cells(part):
    m = re.search(r"\b(\d)S\b", part)
    assert m, part
    return int(m.group(1))


def test_power_parts_match_the_servo_voltage():
    rows = [r for r in read_bom() if r["stage"] != "option"]
    for r in rows:
        if r["role"] == "servo":
            assert "12 V" in r["part"], r["part"]
    batteries = [r for r in rows if r["role"] == "battery"]
    for r in batteries:
        n = cells(r["part"])
        assert SUPPLY_RANGE[0] <= round(n * CELL_LOW, 6) and round(n * CELL_FULL, 6) <= SUPPLY_RANGE[1], r["part"]
    n = {cells(r["part"]) for r in batteries}
    assert len(n) == 1
    for r in rows:
        if r["role"] == "charger":
            assert re.search(r"2-3 cell|\b3S\b", r["part"]), r["part"]
