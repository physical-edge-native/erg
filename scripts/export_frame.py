"""Write the printed parts of one leg (erg/frame.py) as STEP and STL files for a printing service.

Usage: uv run python scripts/export_frame.py [output directory, default build/frame]

The files are not kept in the repository; they are made again from erg/frame.py whenever needed.
Check the servo dimensions on a real servo before ordering (see docs/design/2026-11-25-first-leg-frame.md).
"""

import sys
from pathlib import Path

from build123d import export_step, export_stl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from erg import frame  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/frame")
    out.mkdir(parents=True, exist_ok=True)
    for part in frame.parts():
        export_step(part.solid, str(out / f"{part.name}.step"))
        export_stl(part.solid, str(out / f"{part.name}.stl"), tolerance=0.01, angular_tolerance=0.1)
        print(f"{part.name}: {part.mass * 1000:.1f} g -> {out / part.name}.step, .stl")


if __name__ == "__main__":
    main()
