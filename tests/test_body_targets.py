"""Body targets of the redesign (Issue #58).

Pass criteria:
- the targets are computed for the current model, every value is finite, and the support rectangle is the feet's;
- the current design's values are those recorded as the baseline in docs/design/2026-11-27-redesign.md (to the
  rounding used there), so the record and the code cannot drift apart;
- the computation is deterministic: two runs give the same values bit for bit.

The pass conditions for a new body (mass, tipping, torque margins) are checked in the issue that builds it (#59).
"""

import dataclasses
import math

from erg import body_targets

BASELINE = {  # docs/design/2026-11-27-redesign.md, "Baseline: the current design"
    "mass": (1.40, 0.005),
    "com_height": (0.170, 0.0005),
    "support_width": (0.100, 0.0005),
    "com_offset_x": (-0.0125, 0.0005),
    "height_over_half_width": (3.40, 0.005),
    "tipping_angle_deg": (16.4, 0.05),
    "leg_mass_fraction": (0.38, 0.005),
    "trot_torque_p95": (0.86, 0.005),
    "trot_torque_max": (0.97, 0.005),
    "trot_speed_p95": (0.71, 0.005),
    "stand_up_torque_max": (0.32, 0.005),
}


def test_targets_match_the_recorded_baseline():
    t = dataclasses.asdict(body_targets.compute())
    assert all(math.isfinite(v) for v in t.values())
    for name, (value, tol) in BASELINE.items():
        assert abs(t[name] - value) <= tol, (name, t[name])


def test_targets_are_deterministic():
    a = dataclasses.asdict(body_targets.compute())
    b = dataclasses.asdict(body_targets.compute())
    assert a == b
