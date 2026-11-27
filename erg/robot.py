"""The erg four-legged robot in MuJoCo: model path, names, and the standing posture."""

from pathlib import Path

import mujoco
import numpy as np

from erg import design

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "quadruped.xml"

LEGS = ("FL", "FR", "HL", "HR")
JOINTS_PER_LEG = ("hip_abd", "hip_flex", "knee")
JOINT_NAMES = tuple(f"{leg}_{j}" for leg in LEGS for j in JOINTS_PER_LEG)

# The 12 V class serial bus servo of the redesign (erg/design.py, Issue #59; values to be checked on a real servo).
SERVO_STALL_TORQUE = design.SERVO_STALL_TORQUE  # N*m
SERVO_NO_LOAD_SPEED = design.SERVO_NO_LOAD_SPEED  # rad/s

# Control period of the design record: 20 ms, i.e. 10 simulation steps of 2 ms.
CONTROL_PERIOD = 0.02

# Standing posture, the same for every leg: feet straight under the hip flexion shafts, knees bent backwards.
STAND = np.array([0.0, *design.stand_angles()] * len(LEGS))


def load() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def reset_standing(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the robot in the standing posture, just above the floor, at rest, and command that posture."""
    mujoco.mj_resetData(model, data)
    data.qpos[7:] = STAND
    data.ctrl[:] = STAND
    mujoco.mj_forward(model, data)
