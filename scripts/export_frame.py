"""Write the printed parts (erg/frame.py, erg/torso.py) as STEP and STL files for printing (PLA, 100 % infill): the
torso, the left leg's parts, the right leg's parts (mirror images) and the bench stand. The whole robot needs the
torso once and the left and right leg parts twice each (Issue #53).

Usage: uv run python scripts/export_frame.py [output directory, default build/frame]

The files are not kept in the repository; they are made again from erg/frame.py whenever needed.
Check the servo dimensions on a real servo, with a test fit of one part, before printing the whole leg (see docs/design/2026-11-25-first-leg-frame.md),
and the foot switch's size with a test fit of one shank before printing all four (docs/design/2026-11-26-whole-robot-frame.md).
"""

import sys
from pathlib import Path

from build123d import export_step, export_stl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from erg import frame, torso  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/frame")
    out.mkdir(parents=True, exist_ok=True)
    for part in [torso.torso_part()] + frame.parts() + frame.right_parts():
        export_step(part.solid, str(out / f"{part.name}.step"))
        export_stl(part.solid, str(out / f"{part.name}.stl"), tolerance=0.01, angular_tolerance=0.1)
        print(f"{part.name}: {part.mass * 1000:.1f} g -> {out / part.name}.step, .stl")


if __name__ == "__main__":
    main()
