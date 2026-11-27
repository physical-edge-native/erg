"""Print the body targets of the redesign (Issue #58) for the MuJoCo model, next to the targets of
docs/design/2026-11-27-redesign.md.

Usage: uv run python scripts/body_targets.py
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from erg import body_targets  # noqa: E402

TARGETS = {
    "mass": ("<=", 1.2, "kg"),
    "height_over_half_width": ("<=", 1.5, ""),
    "tipping_angle_deg": (">=", 34.0, "deg"),
    "com_offset_x": ("abs<=", 0.005, "m"),
    "com_offset_y": ("abs<=", 0.005, "m"),
    "leg_mass_fraction": ("<=", 0.20, ""),
    "trot_torque_p95": ("<=", 0.5, "of stall torque"),
    "trot_torque_max": ("<=", 0.8, "of stall torque"),
    "trot_speed_p95": ("<=", 0.6, "of no-load speed"),
    "stand_up_torque_max": ("<=", 0.9, "of stall torque"),
}


def main() -> None:
    t = dataclasses.asdict(body_targets.compute())
    for name, value in t.items():
        line = f"{name:24s} {value:10.4f}"
        if name in TARGETS:
            op, limit, unit = TARGETS[name]
            ok = {"<=": value <= limit, ">=": value >= limit, "abs<=": abs(value) <= limit}[op]
            line += f"   target {op} {limit} {unit}   {'meets' if ok else 'MISSES'}"
        print(line)


if __name__ == "__main__":
    main()
