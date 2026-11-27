"""Write the print-ready parts of one leg of body L1 (erg/cad_leg.py, Issue #72) as STEP and STL files.

Usage: uv run python scripts/export_leg.py [output directory, default build/leg]

Each part is written in its print orientation, lying on the bed (z = 0); the files are not kept in the repository.
The parts are for the front-left leg; the right legs' are their mirror images (mirror them in the slicer, across the
XZ plane). Print settings and the test-print checklist: docs/design/2026-11-27-leg-print.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from erg import cad_leg  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/leg")
    for name, (step, stl, mass) in cad_leg.export(out).items():
        print(f"{name}: {mass * 1000:.1f} g -> {step}, {stl.name}")


if __name__ == "__main__":
    main()
