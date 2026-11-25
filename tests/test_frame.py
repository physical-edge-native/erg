"""The printed frame of one leg (Issue #25) and its agreement with the MuJoCo model.

Pass criteria:
- every printed part is a valid solid;
- the MuJoCo model has the frame's dimensions: hip flexion shaft HIP_DROP below the abduction shaft, thigh and
  shank lengths, foot radius (to 1e-12 m: the model is written in metres from the same millimetre values);
- the model's hip, thigh and shank bodies have the masses, centres of mass and inertias computed from the frame and
  servos. The model stores them rounded (4 significant digits for masses and inertias, 0.1 mm for centres of mass),
  so the tolerances are: mass 0.1 g, centre of mass 0.1 mm, principal moments of inertia 0.1 % (relative);
- the foot is where the frame puts it, for several joint angles (to 1e-6 mm: only rounding);
- no two parts of the leg overlap (overlap volume under 0.5 mm^3, i.e. only touching faces) at the joint limits of
  the model and at the standing posture. Excluded: a fully folded knee with the hip flexed beyond -1.0 rad, where
  the foot reaches the hip (reported in the design record).
"""

from itertools import combinations

import mujoco
import numpy as np
import pytest

from erg import frame, robot


def test_parts_are_valid_solids():
    for part in frame.parts():
        assert part.solid.is_valid, part.name
        assert 0.005 < part.mass < 0.1, (part.name, part.mass)


def test_model_has_the_frame_dimensions():
    m = robot.load()
    for leg in robot.LEGS:
        exact = dict(rtol=0, atol=1e-12)
        assert np.allclose(m.body(f"{leg}_thigh").pos, [0, 0, -frame.HIP_DROP / 1000], **exact)
        assert np.allclose(m.body(f"{leg}_shank").pos, [0, 0, -frame.THIGH_LENGTH / 1000], **exact)
        foot = m.geom(f"{leg}_foot")
        assert np.allclose(foot.pos, [0, 0, -frame.SHANK_LENGTH / 1000], **exact)
        assert np.isclose(foot.size[0], frame.FOOT_RADIUS / 1000, **exact)


def test_model_has_the_frame_masses():
    m = robot.load()
    computed = frame.body_inertials()
    for leg in robot.LEGS:
        mirror = 1 if leg.endswith("L") else -1  # right legs are mirror images: y of the centre of mass flips
        for body, (mass, com, full) in computed.items():
            b = m.body(f"{leg}_{body}")
            assert abs(b.mass[0] - mass) < 1e-4, (leg, body)
            assert np.allclose(b.ipos, [com[0], mirror * com[1], com[2]], atol=1e-4), (leg, body)
            ixx, iyy, izz, ixy, ixz, iyz = full
            principal = np.linalg.eigvalsh([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            assert np.allclose(np.sort(b.inertia), principal, rtol=1e-3, atol=0), (leg, body)


@pytest.mark.parametrize("pose", [(0.0, 0.0, 0.0), (0.0, 0.6, -1.2), (0.3, -0.7, -0.4), (-0.5, 1.6, -2.5)])
def test_foot_position_matches_the_model(pose):
    m = robot.load()
    d = mujoco.MjData(m)
    d.qpos[7:10] = pose  # front left leg
    mujoco.mj_kinematics(m, d)
    hip = m.body("FL_hip")
    foot_model = (d.geom("FL_foot").xpos - d.xpos[hip.id]) * 1000  # torso is not rotated
    foot_frame = np.array(tuple(frame.foot_position(*pose)))
    assert np.allclose(foot_model, foot_frame, atol=1e-6)


POSES = [
    (0.0, 0.0, 0.0),
    (0.0, 0.6, -1.2),
    (0.5, 0.0, -0.2),
    (-0.5, 0.0, -0.2),
    (0.0, 1.6, -2.5),
    (0.0, -1.0, -2.5),
    (0.0, -1.6, -0.2),
    (0.0, 1.6, -0.2),
    (0.5, -1.0, -2.5),
]


@pytest.mark.parametrize("pose", POSES)
def test_no_parts_overlap(pose):
    solids = frame.leg_solids(*pose)
    overlaps = [(a, b) for a, b in combinations(solids, 2) if (solids[a] & solids[b]).volume > 0.5]
    assert overlaps == []
