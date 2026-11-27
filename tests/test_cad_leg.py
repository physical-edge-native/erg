"""Print-ready parts of one leg of body L1 (Issue #72).

Pass criteria:
- every printed part is a valid solid;
- the printed masses of the leg are those used by erg/design.py (and so by the MuJoCo model), to 1 g: shoulder,
  thigh with its spacer, crank with rod, shank with foot. The torso with detailed cradles is lighter than the design's
  torso (the layout's side walls reached into the abduction servos and their horns, which the detailed torso opens);
  it is checked not to be heavier, and the design keeps the layout's value until the whole torso is detailed;
- every screw hole is where the servo's (or horn's, or crank's) pattern parameters put it: no material on the hole's
  axis, material just outside the hole's radius;
- no two parts overlap (overlap volume under 0.5 mm^3, as in tests/test_cad_l1.py) at the standing posture, over the
  hand-made trot, at the hip flexion limits and at the knee limits; parts screwed or pinned to each other are not
  checked against each other;
- during the stand-up sequence the limits of tests/test_cad_l1.py hold (150 mm^3 for the pairs that touch while the
  body rolls, 0.5 mm^3 for all others);
- the export writes a non-empty STEP and STL file per part, each part lying on the bed (z = 0) in its print
  orientation.
"""

from itertools import combinations

import numpy as np
import pytest
from build123d import Location, Plane, Pos, Rot

from erg import cad_l1, cad_leg, design
from test_cad_l1 import TOUCHING, poses, reached_poses

PAIRS_IN_CONTACT = {  # parts screwed or pinned to each other
    frozenset(p) for p in [("torso", "abd_servo"), ("abd_servo", "shoulder"), ("shoulder", "hip_servo"),
                           ("hip_servo", "thigh"), ("thigh", "knee_servo"), ("knee_servo", "crank"),
                           ("crank", "rod"), ("rod", "shank"), ("thigh", "shank"), ("shank", "foot"),
                           ("foot", "switch")]
}


@pytest.fixture(scope="module")
def parts():
    return cad_leg.print_parts()


def test_parts_are_valid(parts):
    for name, s in parts.items():
        assert s.is_valid, name


def test_masses_match_the_design(parts):
    m = {name: cad_leg.mass(s) for name, s in parts.items()}
    expected = {
        "shoulder": (m["shoulder"], design.SHOULDER),
        "thigh": (m["thigh"] + m["thigh_spacer"], design.THIGH_LINK),
        "crank and rod": (m["crank"] + m["rod"], design.PUSH_ROD),
        "shank and foot": (m["shank"] + m["foot"], design.SHANK_LINK),
    }
    for name, (got, want) in expected.items():
        assert abs(got - want) < 0.001, (name, got)
    assert cad_leg.mass(cad_leg.torso()) <= design.PLA_TORSO


def assert_hole(solid, loc: Location, x: float, y: float, z: float, d: float, axis: str = "z"):
    """In the frame `loc`: no material at (x, y, z) on the hole's axis, material at 0.5 mm outside its radius."""
    side = {"z": (1, 0, 0), "y": (1, 0, 0), "x": (0, 1, 0)}[axis]
    r = d / 2 + 0.5
    on_axis = (loc * Pos(x, y, z)).position
    assert not solid.is_inside(on_axis), (x, y, z)
    for s in (1, -1):
        beside = (loc * Pos(x + s * r * side[0], y + s * r * side[1], z + s * r * side[2])).position
        assert solid.is_inside(beside), (x, y, z, s)


def cap_holes(solid, loc: Location):
    mid = cad_leg.SERVO_FACE + 0.4 + 1.0  # inside the cap's plates
    for x, y in cad_leg.TOP_FRAME_HOLES:
        assert_hole(solid, loc, x, y, mid, cad_leg.M2_HOLE)
    for x, y in cad_leg.BOTTOM_FRAME_HOLES:
        assert_hole(solid, loc, x, y, -mid, cad_leg.M2_HOLE)


def horn_holes(solid, horn: Location, depth: float):
    o = cad_leg.HORN_HOLE_OFFSET
    for dx in (-o, o):
        for dy in (-o, o):
            assert_hole(solid, horn, dx, dy, depth, cad_leg.HORN_SCREW_HOLE)
    assert not solid.is_inside((horn * Pos(0, 0, depth)).position)


def test_screw_holes_follow_the_servo_patterns(parts):
    cap_holes(parts["torso_cradle_fl"], cad_l1.abd_servo_loc())
    cap_holes(parts["shoulder"], cad_l1.hip_servo_loc())
    cap_holes(parts["thigh"], cad_l1.knee_servo_loc())
    abd_horn = Location(Plane(origin=(cad_l1.X_FACE, cad_l1.HIP_Y, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)))
    horn_holes(parts["shoulder"], abd_horn, 2.0)
    hip_horn = Location(Plane(origin=(0, cad_leg.TH_HORN_Y, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0)))
    horn_holes(parts["thigh"], hip_horn, 2.5)
    horn_holes(parts["thigh_spacer"], hip_horn, 0.5)
    knee_horn = Location(Plane(origin=(0, cad_leg.CRANK_Y0, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0)))
    horn_holes(parts["crank"], knee_horn, 1.5)
    # ball studs at radius CRANK on both cranks, M2 pilot holes in the rod's ends
    assert_hole(parts["crank"], Location(), 0, cad_leg.CRANK_FACE_Y - 2.5, cad_leg.CRANK, cad_leg.M2_HOLE, "y")
    crank_tip = Rot(0, np.degrees(cad_leg.CRANK_PHASE), 0)
    assert_hole(parts["shank"], crank_tip, 0, cad_leg.SHANK_Y1 - 2.5, cad_leg.CRANK, cad_leg.M2_HOLE, "y")
    for z in (-cad_leg.BALL_LINK_REACH - 2, -cad_leg.ROD_LENGTH + cad_leg.BALL_LINK_REACH + 2):
        assert_hole(parts["rod"], Location(), 0, 0, z, cad_leg.M2_PILOT)
    # knee: the screw's hole through the thigh's hub, the sleeve's bore in the shank
    assert_hole(parts["thigh"], Location(), 0, cad_leg.THIGH_HUB_Y0 + 2, cad_leg.KNEE_Z, cad_leg.M3_HOLE, "y")
    assert_hole(parts["shank"], Location(), 0, cad_leg.SHANK_Y0 + 1, 0, cad_leg.SLEEVE_BORE, "y")


def overlaps(pose, limit=lambda a, b: 0.5):
    solids = {**cad_leg.body_solids(), **cad_leg.leg_solids(*pose)}
    return [(a, b, round((solids[a] & solids[b]).volume, 1)) for a, b in combinations(solids, 2)
            if frozenset((a, b)) not in PAIRS_IN_CONTACT and (solids[a] & solids[b]).volume > limit(a, b)]


KNEE_LIMITS = [(0.0, 0.85, k) for k in design.KNEE_RANGE]


@pytest.mark.parametrize("pose", poses() + KNEE_LIMITS)
def test_no_parts_overlap(pose):
    assert overlaps(pose) == []


def test_stand_up_sequence_fits_the_parts():
    limit = lambda a, b: 150.0 if frozenset((a, b)) in TOUCHING else 0.5  # noqa: E731
    for pose in sorted({tuple(np.round(p, 3)) for p in reached_poses()}):
        assert overlaps(pose, limit) == [], pose


def test_export_writes_every_part(tmp_path):
    out = cad_leg.export(tmp_path)
    assert set(out) == set(cad_leg.PRINT_ORIENTATION)
    for name, (step, stl, _) in out.items():
        assert step.stat().st_size > 1000 and stl.stat().st_size > 1000, name
    for name, solid in cad_leg.print_parts().items():
        assert abs(cad_leg.oriented(name, solid).bounding_box().min.Z) < 1e-6, name
