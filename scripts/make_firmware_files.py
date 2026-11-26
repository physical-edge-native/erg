"""Make the files the leg firmware embeds (Issue #54): Dyne's mlp-actor, a per-leg policy and a check file.

Usage: uv run python scripts/make_firmware_files.py [--agent runs/.../agent.pt] [--rows 32]

- firmware/leg-esp32/main/embed/mlp_actor.wasm: Dyne's mlp-actor (erg.dyne.build_mlp_actor);
- firmware/leg-esp32/main/embed/policy.dynm: the per-leg actor of a trained agent (scripts/train.py --form split),
  or, without --agent, an untrained actor (seed 0) of the same shape, which is enough to measure time and check bits;
- firmware/leg-esp32/main/embed/check.bin: observations from the training environment and the actor's outputs for
  them, computed by Dyne's mlp-actor on the PC (wasmtime). The firmware runs the same observations at start and
  prints whether its outputs are bit for bit the same.

The files are not kept in the repository. Then: idf.py menuconfig (erg leg -> Embed ...), idf.py build flash monitor.
"""

import argparse
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from erg import dyne, ppo  # noqa: E402
from erg.env import ACT_SIZE, LEARNED, N_LEGS, OBS_SIZE, Env  # noqa: E402

OUT = ROOT / "firmware" / "leg-esp32" / "main" / "embed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=Path)
    ap.add_argument("--rows", type=int, default=32)
    a = ap.parse_args()
    torch.manual_seed(0)
    agent = ppo.Agent("split")
    if a.agent:
        agent.load_state_dict(torch.load(a.agent))
    OUT.mkdir(parents=True, exist_ok=True)
    wasm = dyne.build_mlp_actor()
    (OUT / "mlp_actor.wasm").write_bytes(wasm.read_bytes())
    with tempfile.TemporaryDirectory() as d:
        onnx = Path(d) / "policy.onnx"
        torch.onnx.export(agent.actor, torch.zeros(1, OBS_SIZE, dtype=torch.float64), str(onnx),
                          input_names=["x"], output_names=["y"], dynamic_axes={"x": {0: "N"}, "y": {0: "N"}},
                          opset_version=17, dynamo=False)
        data, n_in, layers, n_params, f32w = dyne.convert(str(onnx), "f64")
    (OUT / "policy.dynm").write_bytes(data)
    actor = dyne.Actor(str(wasm))
    if actor.load(data) != 0:
        raise SystemExit("mlp-actor refused the policy")
    env = Env(wire=False)
    rng = np.random.Generator(np.random.PCG64(54))
    obs = env.reset(0)
    rows = []
    while len(rows) < a.rows:
        rows += list(obs)
        act = np.concatenate([ppo.robot.STAND.reshape(N_LEGS, ACT_SIZE) + rng.uniform(-0.2, 0.2, (N_LEGS, ACT_SIZE)),
                              rng.uniform(-1, 1, (N_LEGS, LEARNED))], axis=1)
        obs = env.step(act)
    x = np.array(rows[: a.rows])
    y = actor.eval(x)
    blob = struct.pack("<III", len(x), x.shape[1], y.shape[1])
    for xi, yi in zip(x, y):
        blob += np.ascontiguousarray(xi, dtype="<f8").tobytes() + np.ascontiguousarray(yi, dtype="<f8").tobytes()
    (OUT / "check.bin").write_bytes(blob)
    print(f"{OUT}: mlp_actor.wasm, policy.dynm ({n_params} parameters), check.bin ({len(x)} rows)")


if __name__ == "__main__":
    main()
