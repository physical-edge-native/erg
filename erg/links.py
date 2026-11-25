"""Messages between the legs, in Dyne's wire format (Issue #33).

The encoding is done by erg's WebAssembly module ``crates/erg-links`` (which uses Dyne's ``dyne-wire``), called with
wasmtime as Dyne's ``mlp-actor`` is. ``build()`` builds the module into erg's own ``target/`` directory.
"""

from __future__ import annotations

import functools
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


@functools.cache
def build() -> Path:
    """Build the WebAssembly module (once per process) and return the path of the .wasm file."""
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


# --- Virtual network between the legs (Issue #34) ------------------------------------------------------------------

LINKS = {"none": 0, "neighbours": 1, "all": 2}
N_LEGS = 4


class Network:
    """Dyne's seeded virtual network between the legs, through the same WebAssembly module.

    ``links``: "none", "neighbours" (the ring front-left, front-right, hind-right, hind-left) or "all".
    ``loss``: probability that a message is lost. ``lat_us``, ``jitter_us``: base delay and jitter in microseconds
    (a uniform integer in 0..=2*jitter is added, as in Dyne). ``period_us``: the control period.
    """

    def __init__(self, links: str, loss: float = 0.0, lat_us: int = 0, jitter_us: int = 0, seed: int = 0,
                 period_us: int = 20_000, module: Links | None = None):
        self._m = module or Links()
        r = self._m.ex["erg_net_new"](self._m.store, LINKS[links], float(loss), lat_us, jitter_us, seed, period_us)
        if r != 0:
            raise ValueError(f"erg_net_new rejected the setting ({r})")
        self.links = links

    def targets(self, leg: int) -> list[int]:
        """The legs that `leg` sends to, in increasing order."""
        mask = self._m.ex["erg_net_targets"](self._m.store, leg)
        return [i for i in range(N_LEGS) if mask >> i & 1]

    def route(self, src: int, dst: int, step: int) -> int | None:
        """The control step at which a message sent at `step` from `src` to `dst` is delivered, or None if lost."""
        at = self._m.ex["erg_net_route"](self._m.store, src, dst, step)
        if at == -2:
            raise RuntimeError("no network")
        return None if at < 0 else at

    def step(self, step: int) -> list[tuple[int, int, int | None]]:
        """Route every message sent at control step `step`, in one call: a list of (source, destination, delivery
        step or None if lost), sources in leg order and destinations in increasing order (the same random numbers as
        calling route() in that order)."""
        n = self._m.ex["erg_net_step"](self._m.store, step)
        buf = self._m.ex["erg_net_step_buf"](self._m.store)
        raw = np.frombuffer(bytes(self._m.mem.read(self._m.store, buf, buf + 24 * n)), dtype="<i8").reshape(n, 3)
        return [(int(a), int(b), None if c < 0 else int(c)) for a, b, c in raw]
