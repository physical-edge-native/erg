"""Print the body targets of the redesign (Issue #58) for the MuJoCo model, next to the targets of
docs/design/2026-11-27-redesign.md.

Usage: uv run python scripts/body_targets.py [--servo]

--servo: the joints follow their targets through the servo model of erg/servo.py (nominal parameters; Issue #70)
instead of the model's position actuators.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from erg import body_targets, servo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--servo", action="store_true", help="use the servo model of erg/servo.py (nominal parameters)")
    args = ap.parse_args()
    t = dataclasses.asdict(body_targets.compute(servo=servo.ServoParams() if args.servo else None))
    for name, value in t.items():
        line = f"{name:24s} {value:10.4f}"
        if name in body_targets.TARGETS:
            op, limit, unit = body_targets.TARGETS[name]
            ok = body_targets.meets(name, value)
            line += f"   target {op} {limit} {unit}   {'meets' if ok else 'MISSES'}"
        print(line)


if __name__ == "__main__":
    main()
