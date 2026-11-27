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
