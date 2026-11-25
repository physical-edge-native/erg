"""The fixed trot gait (Issue #32), run in MuJoCo on the current model.

Pass criteria, for forward speeds 0.1 and 0.23 m/s (the upper limit of the walking targets) and two start seeds,
over 10 s:
- the robot does not fall: the torso stays above 0.15 m (standing height about 0.22 m) and tilts less than 10 degrees;
- the mean forward speed is between 0.5 and 1.2 times the command. Grounds: the gait is open loop (it does not
  correct for slipping or for the servos lagging behind their targets), so the robot is expected to cover less
  ground than the feet's paths: on 2026-11-25 it reached 0.65-0.87 of the command. Below 0.5 it would no longer be
  a useful walk; above 1.2 something else than the gait would be moving the robot;
- no joint torque reaches the servo's stall torque (1.91 N m); on 2026-11-25 the largest was 1.82 N m.
"""

import numpy as np
import pytest

from erg import gait, robot
from erg.env import Env

SECONDS = 10.0


def walk(speed, seed):
    env = Env(central=True)
    env.reset(seed, (speed, 0.0, 0.0))
    torso = env.model.body("torso").id
    x0 = env.data.xpos[torso, 0]
    min_height, max_tilt, max_torque = np.inf, 0.0, 0.0
    steps = round(SECONDS / robot.CONTROL_PERIOD)
    for k in range(steps):
        env.step(gait.trot_targets(k * robot.CONTROL_PERIOD, speed))
        min_height = min(min_height, env.data.xpos[torso, 2])
        up = env.data.xmat[torso].reshape(3, 3)[2, 2]
        max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(up, -1, 1))))
        max_torque = max(max_torque, np.abs(env.data.actuator_force).max())
    mean_speed = (env.data.xpos[torso, 0] - x0) / SECONDS
    return mean_speed, min_height, max_tilt, max_torque


def test_standing_posture_is_the_gait_at_rest():
    assert np.allclose(gait.trot_targets(0.0, 0.0), robot.STAND, atol=1e-12)


@pytest.mark.parametrize("speed", [0.1, 0.23])
@pytest.mark.parametrize("seed", [0, 1])
def test_trot_walks_forward_without_falling(speed, seed):
    mean_speed, min_height, max_tilt, max_torque = walk(speed, seed)
    assert min_height > 0.15, min_height
    assert max_tilt < 10.0, max_tilt
    assert 0.5 * speed < mean_speed < 1.2 * speed, mean_speed
    assert max_torque < robot.SERVO_STALL_TORQUE, max_torque
