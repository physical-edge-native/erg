"""The MJCF model of the robot (Issue #4).

Pass criteria:
- the model loads; it has 12 hinge joints and 12 position actuators in the order of erg.robot.JOINT_NAMES;
- the actuators cannot exceed the servo's stall torque, and the total mass is inside the 1.2-2 kg of the design record;
- the model keeps to the subset of MJCF that converts to URDF (see the comment at the top of models/quadruped.xml);
- under a fixed standing command, the robot stands for 10 s: the torso stays above 0.10 m and tilts less than
  10 degrees at every control step, only the feet touch the floor, and no joint torque reaches the stall torque;
- the same run twice gives the same state bit for bit.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from erg import robot


def test_model_structure():
    m = robot.load()
    assert m.njnt == 13 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert all(t == mujoco.mjtJoint.mjJNT_HINGE for t in m.jnt_type[1:])
    names = tuple(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt))
    assert names == robot.JOINT_NAMES
    assert m.nu == 12
    for i in range(m.nu):
        assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == robot.JOINT_NAMES[i]
        assert m.actuator_trnid[i, 0] == i + 1
        assert m.actuator_forcelimited[i]
        assert np.allclose(m.actuator_forcerange[i], [-robot.SERVO_STALL_TORQUE, robot.SERVO_STALL_TORQUE])
        # the command range equals the joint range
        assert np.array_equal(m.actuator_ctrlrange[i], m.jnt_range[i + 1])
    assert np.allclose(m.dof_damping[6:], robot.SERVO_STALL_TORQUE / robot.SERVO_NO_LOAD_SPEED, rtol=1e-3)
    total = m.body_subtreemass[0]
    assert 1.2 <= total <= 2.0


def test_model_is_urdf_compatible():
    root = ET.parse(robot.MODEL_PATH).getroot()
    bodies = list(root.iter("body"))
    assert bodies and all(b.find("inertial") is not None for b in bodies)
    assert {g.get("type") for g in root.find("worldbody").iter("geom")} <= {"box", "capsule", "sphere", "plane"}
    for tag in ("tendon", "equality", "composite", "flexcomp", "skin"):
        assert root.find(f".//{tag}") is None, tag
    assert all(j.get("type", "hinge") == "hinge" for j in root.iter("joint"))
    # each body has at most one joint (URDF links have one parent joint)
    assert all(len(b.findall("joint")) + len(b.findall("freejoint")) <= 1 for b in bodies)


def stand(seconds=10.0):
    m = robot.load()
    d = mujoco.MjData(m)
    robot.reset_standing(m, d)
    substeps = round(robot.CONTROL_PERIOD / m.opt.timestep)
    torso = m.body("torso").id
    floor = m.geom("floor").id
    feet = {m.geom(f"{leg}_foot").id for leg in robot.LEGS}
    min_height, max_tilt, max_torque, others = np.inf, 0.0, 0.0, 0
    for _ in range(round(seconds / robot.CONTROL_PERIOD)):
        d.ctrl[:] = robot.STAND
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        min_height = min(min_height, d.xpos[torso, 2])
        up = d.xmat[torso].reshape(3, 3)[2, 2]  # z component of the torso's z axis
        max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(up, -1, 1))))
        max_torque = max(max_torque, np.abs(d.actuator_force).max())
        for c in d.contact[: d.ncon]:
            other = c.geom2 if c.geom1 == floor else c.geom1
            if floor in (c.geom1, c.geom2) and other not in feet:
                others += 1
    return d, min_height, max_tilt, max_torque, others


def test_stands_for_ten_seconds():
    d, min_height, max_tilt, max_torque, others = stand()
    assert min_height > 0.10, min_height
    assert max_tilt < 10.0, max_tilt
    assert others == 0
    assert max_torque < robot.SERVO_STALL_TORQUE, max_torque
    assert np.all(np.isfinite(d.qpos))


def test_standing_is_deterministic():
    a = stand(1.0)[0].qpos.copy()
    b = stand(1.0)[0].qpos.copy()
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))
