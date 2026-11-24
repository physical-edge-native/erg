"""Access to the pinned Dyne checkout in ``third_party/dyne``.

erg depends on Dyne wherever it can. Dyne's Python tools are scripts, not an installed package,
so this module puts Dyne's ``scripts/`` directory on ``sys.path`` and re-exports what erg uses:

- ``onnx_import.convert``: ONNX model -> ``.dynm`` bytes read by Dyne's ``mlp-actor``;
- ``onnx_check.Actor``: runs a WebAssembly solver with wasmtime;
- ``onnx_check.ref_eval``: Python evaluation in the same order of operations as ``mlp-actor``.

``build_mlp_actor()`` builds Dyne's ``mlp-actor`` for ``wasm32-unknown-unknown`` into erg's own
``target/`` directory, so the Dyne checkout stays clean.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DYNE = ROOT / "third_party" / "dyne"
TARGET = ROOT / "target" / "dyne"

if not (DYNE / "scripts" / "onnx_import.py").is_file():
    raise ImportError(
        "Dyne is missing in third_party/dyne; run `git submodule update --init`"
    )

sys.path.insert(0, str(DYNE / "scripts"))
import onnx_check  # noqa: E402
import onnx_import  # noqa: E402

Actor = onnx_check.Actor
ref_eval = onnx_check.ref_eval
convert = onnx_import.convert


def build_mlp_actor(small: bool = False) -> Path:
    """Build Dyne's mlp-actor as WebAssembly and return the path of the .wasm file.

    ``small=True`` builds the microcontroller variant (parameter limit 2048).
    """
    target_dir = TARGET / ("mlp-actor-small" if small else "mlp-actor")
    cmd = [
        "cargo", "build", "--release", "--quiet",
        "--target", "wasm32-unknown-unknown",
        "--target-dir", str(target_dir),
    ]
    if small:
        cmd += ["--features", "small"]
    subprocess.run(cmd, cwd=DYNE / "crates" / "mlp-actor", check=True, env=os.environ.copy())
    return target_dir / "wasm32-unknown-unknown" / "release" / "mlp_actor.wasm"
