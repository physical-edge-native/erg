"""MuJoCo is installed and deterministic within one run of the same build.

Pass criterion: stepping the same model twice from the same state gives the same state bit for bit.
(Equality across operating systems is not expected from MuJoCo and is not tested here.)
"""

import mujoco
import numpy as np

XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.2">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def run(steps):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data.qpos.copy()


def test_mujoco_steps_deterministically():
    a, b = run(500), run(500)
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))
    assert 0.0 < a[2] < 0.2  # the box fell and rests on the floor
