"""Write the printed parts of body L1 (erg/cad_l1.py) as STEP and STL files.

Usage: uv run python scripts/export_frame.py [output directory, default build/frame]

The files are not kept in the repository. The parts are layout-level (Issue #59): they fix the space each part takes
and its mass; screw holes, pins and the push rod's ends are added after a test print of one leg.
Parts are for the left legs; the right legs' are their mirror images (mirror them in the slicer, across the XZ plane).
"""

import sys
from pathlib import Path

from build123d import export_step, export_stl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from erg import cad_l1  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/frame")
    out.mkdir(parents=True, exist_ok=True)
    parts = {"torso": cad_l1.torso(), "shoulder": cad_l1.shoulder(), "thigh": cad_l1.thigh(), "shank": cad_l1.shank(),
             "crank_and_rod": cad_l1.crank_and_rod()}
    for name, solid in parts.items():
        export_step(solid, str(out / f"{name}.step"))
        export_stl(solid, str(out / f"{name}.stl"), tolerance=0.01, angular_tolerance=0.1)
        print(f"{name}: {cad_l1.mass(solid) * 1000:.1f} g -> {out / name}.step, .stl")


if __name__ == "__main__":
    main()
