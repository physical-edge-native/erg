"""Layout-level code-CAD of body L1 (Issue #59) and its agreement with the MuJoCo model.

Pass criteria:
- every part is a valid solid;
- the printed masses used by erg/design.py (and so by the MuJoCo model) are those of the parts, to 1 g;
- no two parts overlap (overlap volume under 0.5 mm^3) at: the standing posture; the hand-made trot over one cycle;
  the abduction limits in the standing leg posture; the hip flexion limits; and every pose the stand-up sequence passes
  through (erg.righting, both push sides). Other combinations of large abduction and flexion are not checked here: the MuJoCo
  model has the thigh (its inner plate, the knee servo and the link) collide with the torso, so the robot cannot reach
  them in simulation, and the stand-up sequence was searched with those contacts.
"""

from itertools import combinations

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
    out += [(a, robot.STAND[1], robot.STAND[2]) for a in design.ABD_RANGE]
    out += [(0.0, f, -1.0) for f in design.FLEX_RANGE]
    for side in (1, -1):
        moves = righting.sequence(robot.STAND.copy(), side)
        for a, b, _ in moves:
            for t in np.linspace(0, 1, 4):
                q = a + (b - a) * t
                out.append(tuple(q[:3]))  # front-left leg
                out.append((-q[3], q[4], q[5]))  # front-right leg, as its mirror image on the left
    return sorted({tuple(p) for p in np.round(out, 3).tolist()})


@pytest.mark.parametrize("pose", poses())
def test_no_parts_overlap(pose):
    assert overlaps(pose) == []
