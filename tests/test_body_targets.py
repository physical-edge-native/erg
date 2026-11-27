"""Body targets of the redesign (Issues #58, #59).

Pass criteria:
- the current model (body layout L1, #59) meets every target of docs/design/2026-11-27-redesign.md;
- the computation is deterministic: two runs give the same values bit for bit.

The first design's values are kept in the design record as the baseline (checked by this test until #59 replaced the
model).
"""

import dataclasses
import math

from erg import body_targets


def test_body_l1_meets_the_targets():
    t = dataclasses.asdict(body_targets.compute())
    assert all(math.isfinite(v) for v in t.values())
    missed = {name: t[name] for name in body_targets.TARGETS if not body_targets.meets(name, t[name])}
    assert missed == {}


def test_targets_are_deterministic():
    a = dataclasses.asdict(body_targets.compute())
    b = dataclasses.asdict(body_targets.compute())
    assert a == b


def test_servo_model_targets_are_reported():
    """With ServoParams.position_actuator() the targets are those of the model bit for bit; with the nominal servo
    model (Issue #70) they are finite and deterministic (their values are reported in
    docs/design/2026-11-27-servo-model.md, not checked against the targets here)."""
    from erg import servo

    assert body_targets.compute(servo=servo.ServoParams.position_actuator()) == body_targets.compute()
    a = dataclasses.asdict(body_targets.compute(servo=servo.ServoParams()))
    assert all(math.isfinite(v) for v in a.values())
    assert a == dataclasses.asdict(body_targets.compute(servo=servo.ServoParams()))
