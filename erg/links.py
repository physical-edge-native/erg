"""Messages between the legs, in Dyne's wire format (Issue #33).

The encoding is done by erg's WebAssembly module ``crates/erg-links`` (which uses Dyne's ``dyne-wire``), called with
wasmtime as Dyne's ``mlp-actor`` is. ``build()`` builds the module into erg's own ``target/`` directory.
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import wasmtime

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "crates" / "erg-links"
TARGET = ROOT / "target" / "erg-links"

N_VALUES = 11  # gravity direction 3, angular velocity 3, foot contact 1, learned vector 4
MSG_LEN = 24 + 8 + 8 * N_VALUES  # 120 bytes


def build() -> Path:
    """Build the WebAssembly module and return the path of the .wasm file."""
    cmd = [
        "cargo", "build", "--release", "--quiet", "--locked",
        "--target", "wasm32-unknown-unknown", "--target-dir", str(TARGET),
    ]
    subprocess.run(cmd, cwd=CRATE, check=True, env=os.environ.copy())
    return TARGET / "wasm32-unknown-unknown" / "release" / "erg_links.wasm"


class Links:
    """Encode and decode one leg's message with the WebAssembly module."""

    def __init__(self, path: str | Path | None = None):
        path = str(path or build())
        self.engine = wasmtime.Engine()
        self.store = wasmtime.Store(self.engine)
        module = wasmtime.Module.from_file(self.engine, path)
        self.ex = wasmtime.Instance(self.store, module, []).exports(self.store)
        self.mem = self.ex["memory"]
        self._values = self.ex["erg_values_buf"](self.store)
        self._msg = self.ex["erg_msg_buf"](self.store)
        self._cap = self.ex["erg_msg_cap"](self.store)

    def encode(self, src: int, step: int, t_us: int, values) -> bytes:
        v = np.ascontiguousarray(values, dtype="<f8")
        if v.shape != (N_VALUES,):
            raise ValueError(f"expected {N_VALUES} values, got shape {v.shape}")
        self.mem.write(self.store, v.tobytes(), self._values)
        n = self.ex["erg_encode"](self.store, src, step, t_us)
        if n < 0:
            raise ValueError(f"erg_encode failed ({n})")
        return bytes(self.mem.read(self.store, self._msg, self._msg + n))

    def decode(self, msg: bytes) -> tuple[int, int, int, np.ndarray]:
        """Returns (src, step, t_us, values)."""
        if len(msg) > self._cap:
            raise ValueError("message longer than the buffer")
        self.mem.write(self.store, msg, self._msg)
        r = self.ex["erg_decode"](self.store, len(msg))
        if r != N_VALUES:
            raise ValueError(f"erg_decode failed ({r})")
        raw = self.mem.read(self.store, self._values, self._values + 8 * N_VALUES)
        values = np.frombuffer(bytes(raw), dtype="<f8").copy()
        ex, st = self.ex, self.store
        return ex["erg_last_src"](st), ex["erg_last_step"](st), ex["erg_last_t_us"](st), values


def reference_bytes(src: int, step: int, t_us: int, values) -> bytes:
    """The same message built in Python from the layout written in Dyne's dyne-wire documentation (version 2,
    kind IFACE_BAND = 1, quantity inertial = 3, then count = 1 and per_point = 11 before the values). Used to check
    the module independently."""
    header = struct.pack("<BBBBIQQ", 2, 1, 3, 0, src, step, t_us)
    return header + struct.pack("<II", 1, N_VALUES) + np.ascontiguousarray(values, dtype="<f8").tobytes()
