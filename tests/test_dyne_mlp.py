"""A policy-shaped network trained-side in PyTorch runs in Dyne's mlp-actor (WebAssembly).

Path: PyTorch -> ONNX -> Dyne's onnx_import (.dynm) -> Dyne's mlp-actor run with wasmtime.

Pass criteria, stated in advance:
- ReLU network: the WebAssembly output equals Dyne's same-order Python evaluation **bit for bit**
  (only additions, multiplications and comparisons, so IEEE 754 fixes every bit);
- the same output agrees with PyTorch (float64) within 1e-12. PyTorch may sum in a different order,
  so bit equality is not expected; for a 3-layer net of width 32 with inputs in [-1, 1] the rounding
  differences are of order 1e-15, so 1e-12 leaves a wide margin while still catching a wrong weight.
"""

import numpy as np
import pytest
import torch

from erg import dyne

N_IN, WIDTH, N_OUT = 12, 32, 3  # e.g. one leg: joint angles/velocities, IMU, contact -> 3 joint targets


@pytest.fixture(scope="module")
def actors():
    return {small: dyne.Actor(str(dyne.build_mlp_actor(small=small))) for small in (False, True)}


def make_onnx(path):
    torch.manual_seed(20261124)
    net = torch.nn.Sequential(
        torch.nn.Linear(N_IN, WIDTH), torch.nn.ReLU(),
        torch.nn.Linear(WIDTH, WIDTH), torch.nn.ReLU(),
        torch.nn.Linear(WIDTH, N_OUT),
    ).double()
    torch.onnx.export(
        net, torch.zeros(1, N_IN, dtype=torch.float64), str(path),
        input_names=["x"], output_names=["y"],
        dynamic_axes={"x": {0: "N"}, "y": {0: "N"}}, opset_version=17, dynamo=False,
    )
    return net


@pytest.mark.parametrize("small", [False, True], ids=["full", "small"])
def test_policy_runs_in_dyne_mlp_actor(tmp_path, actors, small):
    net = make_onnx(tmp_path / "policy.onnx")
    data, n_in, layers, n_params, f32w = dyne.convert(str(tmp_path / "policy.onnx"), "f64")
    assert (n_in, len(layers), f32w) == (N_IN, 3, False)

    actor = actors[small]
    assert actor.load(data) == 0, "mlp-actor refused the model"

    x = np.random.default_rng(1).uniform(-1.0, 1.0, size=(64, N_IN))
    y_wasm = actor.eval(x)
    y_ref = np.array([dyne.ref_eval(layers, row, f32w) for row in x])
    with torch.no_grad():
        y_torch = net(torch.from_numpy(x)).numpy()

    assert np.array_equal(y_wasm.view(np.uint64), y_ref.view(np.uint64)), "not bit-exact"
    assert np.max(np.abs(y_wasm - y_torch)) <= 1e-12
