"""The erg four-legged robot in MuJoCo: model path, names, and the standing posture."""

from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "quadruped.xml"

LEGS = ("FL", "FR", "HL", "HR")
JOINTS_PER_LEG = ("hip_abd", "hip_flex", "knee")
JOINT_NAMES = tuple(f"{leg}_{j}" for leg in LEGS for j in JOINTS_PER_LEG)

# Feetech STS3215 at 7.4 V, from its datasheet (see models/quadruped.xml).
SERVO_STALL_TORQUE = 1.91  # N*m (19.5 kg*cm)
SERVO_NO_LOAD_SPEED = 5.51  # rad/s (0.19 s per 60 degrees)

# Control period of the design record: 20 ms, i.e. 10 simulation steps of 2 ms.
CONTROL_PERIOD = 0.02

# Standing posture, the same for every leg: feet under the hips, knees bent backwards.
STAND = np.array([0.0, 0.6, -1.2] * len(LEGS))


def load() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def reset_standing(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the robot in the standing posture, just above the floor, at rest, and command that posture."""
    mujoco.mj_resetData(model, data)
    data.qpos[7:] = STAND
    data.ctrl[:] = STAND
    mujoco.mj_forward(model, data)
