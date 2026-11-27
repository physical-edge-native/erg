"""Write models/quadruped.xml from erg/design.py (body layout L1, Issue #59).

Usage: uv run python scripts/make_model.py [--check]
With --check, exit with status 1 if the committed model differs from what this script writes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from erg import design  # noqa: E402

MODEL = ROOT / "models" / "quadruped.xml"


def main() -> None:
    text = design.mjcf()
    if "--check" in sys.argv:
        sys.exit(0 if MODEL.read_text() == text else 1)
    MODEL.write_text(text)


if __name__ == "__main__":
    main()
