"""Layout-level code-CAD of body L1 (Issue #59) and its agreement with the MuJoCo model.

Pass criteria:
- every part is a valid solid;
- the printed masses used by erg/design.py (and so by the MuJoCo model) are those of the parts, to 1 g;
- no two parts overlap (overlap volume under 0.5 mm^3) at: the standing posture; the hand-made trot over one cycle;
  the hip flexion limits. The abduction limits (+-2.6 rad) swing the leg over the torso; there the thigh meets the
  torso, which the model's collisions stop, so they are checked only at the angles the stand-up sequence reaches;
- during the stand-up sequence (erg.righting, both push sides), at the joint angles the MuJoCo simulation reaches every
  0.5 s, a thigh may touch the torso while it rolls the body (the model has them collide): the overlap of the leg's
  parts with the torso and the abduction servo stays under 150 mm^3, the size of the soft contact and of the
  difference between the model's collision boxes and the printed parts (about 2 mm deep over a few square
  centimetres); all other pairs stay under 0.5 mm^3. Other combinations of large abduction and flexion are not checked here: the MuJoCo
  model has the thigh (its inner plate, the knee servo and the link) collide with the torso, so the robot cannot reach
  them in simulation, and the stand-up sequence was searched with those contacts.
"""

from itertools import combinations

import mujoco
import numpy as np
import pytest

from erg import cad_l1, design, gait, righting, robot

PAIRS_IN_CONTACT = {  # parts screwed or pinned to each other
    frozenset(p) for p in [("torso", "abd_servo"), ("shoulder", "hip_servo"), ("thigh", "knee_servo"),
                           ("thigh", "rod"), ("knee_servo", "rod"), ("thigh", "shank"), ("rod", "shank")]
}


def overlaps(pose):
    solids = {**cad_l1.body_solids(), **cad_l1.leg_solids(*pose)}
    return [(a, b) for a, b in combinations(solids, 2)
            if frozenset((a, b)) not in PAIRS_IN_CONTACT and (solids[a] & solids[b]).volume > 0.5]


def test_parts_are_valid_and_masses_match_the_design():
    parts = {"torso": cad_l1.torso(), "shoulder": cad_l1.shoulder(), "thigh": cad_l1.thigh(),
             "shank": cad_l1.shank(), "rod": cad_l1.crank_and_rod()}
    for name, s in parts.items():
        assert s.is_valid, name
    expected = {"torso": design.PLA_TORSO, "shoulder": design.SHOULDER, "thigh": design.THIGH_LINK,
                "shank": design.SHANK_LINK, "rod": design.PUSH_ROD}
    for name, m in expected.items():
        assert abs(cad_l1.mass(parts[name]) - m) < 0.001, (name, cad_l1.mass(parts[name]))


def poses():
    out = [tuple(robot.STAND[:3])]
    out += [tuple(gait.leg_targets(s, 0.04)) for s in np.linspace(0, 1, 8, endpoint=False)]
    out += [(0.0, f, -1.0) for f in design.FLEX_RANGE]
    return sorted({tuple(p) for p in np.round(out, 3).tolist()})


def reached_poses():
    """Front legs' joint angles reached in MuJoCo every 0.5 s of the stand-up sequence from the back (the right leg
    as its mirror image on the left)."""
    m = robot.load()
    out = []
    for side in (1, -1):
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[2], d.qpos[3:7] = 0.09, [0, 1, 0, 0]
        d.qpos[7:] = np.tile(righting.FOLD_LEG, 4)
        mujoco.mj_forward(m, d)
        moves = righting.sequence(np.tile(righting.FOLD_LEG, 4), side)
        for k in range(round(righting.duration(moves) / robot.CONTROL_PERIOD)):
            d.ctrl[:] = righting.targets_at(moves, k * robot.CONTROL_PERIOD)
            for _ in range(round(robot.CONTROL_PERIOD / m.opt.timestep)):
                mujoco.mj_step(m, d)
            if k % 25 == 0:
                q = d.qpos[7:]
                out += [tuple(q[0:3]), (-q[3], q[4], q[5])]
    return out


TOUCHING = {frozenset(p) for p in [("torso", "thigh"), ("torso", "knee_servo"), ("abd_servo", "thigh"),
                                   ("abd_servo", "knee_servo"), ("torso", "shank"), ("torso", "rod")]}


def test_stand_up_sequence_fits_the_parts():
    for pose in reached_poses():
        solids = {**cad_l1.body_solids(), **cad_l1.leg_solids(*pose)}
        for a, b in combinations(solids, 2):
            if frozenset((a, b)) in PAIRS_IN_CONTACT:
                continue
            limit = 150.0 if frozenset((a, b)) in TOUCHING else 0.5
            assert (solids[a] & solids[b]).volume < limit, (np.round(pose, 2), a, b)


@pytest.mark.parametrize("pose", poses())
def test_no_parts_overlap(pose):
    assert overlaps(pose) == []
