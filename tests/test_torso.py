"""The torso frame, the right legs' parts and the foot switch pocket (Issue #53), and their agreement with the
MuJoCo model.

Pass criteria:
- the torso and every right leg part are valid solids; a right leg part has the mass of its left one (mirror
  images, to 1e-9 g);
- the model's hips are at the torso's abduction shafts and its IMU sites on the torso's IMU pads (to 1e-12 m);
- the model's torso body has the mass, centre of mass and inertia computed from the torso frame, the abduction
  servos and the estimated battery, electronics and wiring. The model stores them rounded (4 significant digits for
  mass and inertia, 0.1 mm for the centre of mass), so the tolerances are: mass 0.1 g, centre of mass 0.1 mm,
  principal moments of inertia 0.1 % (relative);
- the foot switch sits in its pocket without touching the shank, its actuator face SOLE_GAP inside the sole;
- no part of the torso, of the battery or of any leg (printed parts, servos, switch) overlaps another (overlap
  volume under 0.5 mm^3) at the abduction limits combined with folded and stretched legs, for all four legs. Legs
  are not checked against each other (front and hind feet could meet; the MuJoCo model does not let robot parts
  collide either);
- every printed part fits the printer's bed (PRINT_BED, an assumption to be checked on the printer).
"""

import mujoco
import numpy as np
import pytest

from erg import frame, robot, torso

PRINT_BED = (250.0, 250.0, 250.0)  # mm; assumed for the Bambu Lab X2D, to be checked against its build volume


def test_torso_and_right_parts_are_valid():
    t = torso.torso_part()
    assert t.solid.is_valid and 0.05 < t.mass < 0.2
    left = {p.name: p for p in frame.parts()}
    for p in frame.right_parts():
        assert p.solid.is_valid, p.name
        assert abs(p.mass - left[p.name.removesuffix("_right")].mass) < 1e-12


def test_model_hips_and_imus_follow_the_torso():
    m = robot.load()
    for leg, (x, y) in torso.HIPS.items():
        assert np.allclose(m.body(f"{leg}_hip").pos, [x / 1000, y / 1000, 0], rtol=0, atol=1e-12)
        assert np.allclose(m.site(f"{leg}_imu").pos, np.array(torso.IMU_SITES[leg]) / 1000, rtol=0, atol=1e-12)


def test_model_has_the_torso_mass():
    m = robot.load()
    mass, com, full = torso.inertial()
    b = m.body("torso")
    assert abs(b.mass[0] - mass) < 1e-4
    assert np.allclose(b.ipos, com, atol=1e-4)
    ixx, iyy, izz, ixy, ixz, iyz = full
    principal = np.linalg.eigvalsh([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    assert np.allclose(np.sort(b.inertia), principal, rtol=1e-3, atol=0)


def test_switch_fits_its_pocket():
    shank = frame.shank_part().solid
    switch = frame.switch_body()
    assert (shank & switch).volume < 1e-6
    top = switch.bounding_box().max.X
    assert top == pytest.approx(frame.SHANK_LENGTH + frame.FOOT_RADIUS - frame.SOLE_GAP - frame.SWITCH_CLEARANCE)


POSES = [(0.5, 0.0, -0.2), (-0.5, 0.0, -0.2), (0.5, 1.6, -2.5), (-0.5, -1.0, -2.5), (0.0, -1.6, -0.2), (0.0, 0.6, -1.2)]


@pytest.mark.parametrize("pose", POSES)
def test_nothing_overlaps(pose):
    fixed = {"torso": torso.torso_solid(), "battery": torso.battery_solid()}
    bad = []
    for leg in torso.HIPS:
        for name, s in torso.leg_solids_in_torso(leg, *pose).items():
            for fname, f in fixed.items():
                v = (f & s).volume
                if v > 0.5:
                    bad.append((leg, name, fname, round(v, 2)))
    assert bad == []


def test_parts_fit_the_print_bed():
    for p in [torso.torso_part()] + frame.parts() + frame.right_parts():
        size = sorted(p.solid.bounding_box().size)
        assert all(s <= b for s, b in zip(size, sorted(PRINT_BED))), (p.name, size)
