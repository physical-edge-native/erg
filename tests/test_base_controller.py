"""The base controller (Issue #71), against docs/design/2026-11-27-base-controller.md.

The pass conditions are checked here on seed 0 over 10 s (measured over 4-10 s), one command of each kind, so that
the suite stays short on the CI's three systems (each run takes about a second); scripts/base_controller.py runs the
full set (five seeds, 20 s, every command) and its output is recorded in the design record. The torque condition
is the one changed by the maintainer after the first run (design record): at 0.1 m/s.
"""

import inspect
import re

import numpy as np
import pytest

from erg import base_controller as bc
from erg import env as env_mod
from erg import links, reward, robot
from erg.servo import ServoParams

SECONDS = 10.0
WINDOW = 4.0


def test_neighbours_are_the_ring_of_the_virtual_network():
    net = links.Network("neighbours")
    for i, leg in enumerate(robot.LEGS):
        assert sorted(robot.LEGS[j] for j in net.targets(i)) == sorted(bc.NEIGHBOURS[leg])


def _obs_row(rng):
    return rng.uniform(-1.0, 1.0, size=env_mod.OBS_SIZE)


def _slot_start(receiver: str, sender: str) -> int:
    others = [x for x in robot.LEGS if x != receiver]
    return env_mod.LEG_SIZE + 3 + env_mod.N_LEGS + others.index(sender) * env_mod.MSG_SLOT


def test_leg_input_holds_only_own_values_and_neighbour_slots():
    rng = np.random.default_rng(0)
    for leg in robot.LEGS:
        row = _obs_row(rng)
        inp = bc.leg_input(row, leg)
        assert set(inp.neighbours) == set(bc.NEIGHBOURS[leg])
        diagonal = next(x for x in robot.LEGS if x != leg and x not in bc.NEIGHBOURS[leg])
        changed = row.copy()
        a = _slot_start(leg, diagonal)
        changed[a:a + env_mod.MSG_SLOT] = rng.uniform(-5.0, 5.0, size=env_mod.MSG_SLOT)
        assert bc.leg_input(changed, leg) == inp
        # every field of LegInput comes from this leg's own row: own sensors, phase, command, neighbour slots
        assert set(vars(inp)) == {"leg", "joint_pos", "acc", "gyro", "contact", "phase_cs", "command", "neighbours"}


def test_controller_output_ignores_the_diagonal_leg():
    rng = np.random.default_rng(1)
    for leg in robot.LEGS:
        row = _obs_row(rng)
        row[:3] = robot.STAND[:3]
        row[9:12] = (0.0, 0.0, 9.81)
        for other in bc.NEIGHBOURS[leg]:
            row[_slot_start(leg, other) + env_mod.MSG_VALUES] = 1.0  # age 1
        diagonal = next(x for x in robot.LEGS if x != leg and x not in bc.NEIGHBOURS[leg])
        changed = row.copy()
        a = _slot_start(leg, diagonal)
        changed[a:a + env_mod.MSG_SLOT] = rng.uniform(-5.0, 5.0, size=env_mod.MSG_SLOT)
        changed[a + env_mod.MSG_VALUES] = 1.0
        out1 = bc.LegController(leg).step(bc.leg_input(row, leg))
        out2 = bc.LegController(leg).step(bc.leg_input(changed, leg))
        assert out1 == out2


def test_a_controller_refuses_another_legs_input():
    inp = bc.leg_input(np.zeros(env_mod.OBS_SIZE), "FR")
    with pytest.raises(ValueError):
        bc.LegController("FL").step(inp)


def test_links_all_gives_the_same_walk_as_neighbours():
    """With links "all" the diagonal leg's messages reach every leg's observation; the controller must not use them,
    so the whole walk is the same bit for bit."""
    a = bc.run((0.2, 0.0, 0.1), 0, seconds=3.0, window=1.0, links="neighbours")
    b = bc.run((0.2, 0.0, 0.1), 0, seconds=3.0, window=1.0, links="all")
    assert np.array_equal(a.targets, b.targets)


def test_controller_code_uses_no_library_trigonometry():
    """For the bit-exact port to the firmware (#54): the controller's arithmetic uses erg.exactmath and sqrt only."""
    for f in (bc.LegController.step, bc.foot_position, bc.leg_ik, bc._blend, bc._glide, bc.leg_input,
              bc.gait.leg_angles):
        assert not re.search(r"\b(np|numpy|math)\.", inspect.getsource(f)), f.__name__


def test_kinematics_round_trip_and_standing_posture():
    for leg in robot.LEGS:
        q = (0.15, 0.3, -1.3)
        assert np.allclose(bc.leg_ik(leg, *bc.foot_position(leg, q)), q, atol=1e-12)
        x, y, z = bc.foot_position(leg, robot.STAND[:3])
        assert np.allclose(bc.leg_ik(leg, x, y, z), robot.STAND[:3], atol=1e-12)


def test_coupling_brings_the_clocks_to_a_trot():
    e = env_mod.Env(wire=False)
    e.reset(0, (0.1, 0.0, 0.0))
    e.phase = np.array([0.0, 0.3, 0.1, 0.6])  # FL, FR, HL, HR: not a trot
    obs = e.observe()
    ctrls = [bc.LegController(leg) for leg in robot.LEGS]
    for _ in range(250):
        outs = [c.step(bc.leg_input(obs[i], leg)) for i, (c, leg) in enumerate(zip(ctrls, robot.LEGS))]
        obs = e.step(np.array([list(o.targets) + list(o.message) for o in outs]), [o.freq for o in outs])
    d = lambda i, j: (e.phase[i] - e.phase[j]) % 1.0  # noqa: E731
    assert abs(d(0, 1) - 0.5) < 0.02 and abs(d(0, 2) - 0.5) < 0.02  # neighbours half a period apart
    assert min(d(0, 3), 1.0 - d(0, 3)) < 0.02  # diagonal legs together


def test_deterministic():
    a = bc.run((0.2, 0.0, 0.0), 0, seconds=3.0, window=1.0)
    b = bc.run((0.2, 0.0, 0.0), 0, seconds=3.0, window=1.0)
    assert np.array_equal(a.targets, b.targets)
    assert a.forward == b.forward


@pytest.mark.parametrize("command", [(0.2, 0.0, 0.0), (0.0, 0.0, -0.5), (0.0, 0.1, 0.0)])
def test_pass_conditions_seed_0(command):
    r = bc.run(command, 0, seconds=SECONDS, window=WINDOW)
    assert bc.check(command, r) == []


@pytest.mark.parametrize("direction", [1, -1])
def test_small_push_no_fall(direction):
    r = bc.run((0.2, 0.0, 0.0), 0, seconds=SECONDS, window=WINDOW, push=direction, push_time=6.0)
    assert not r.fell


def test_torque_condition():
    """Condition 6 as changed after the first run (design record): at 0.1 m/s. The torque at 0.15 and 0.2 m/s is
    reported by scripts/base_controller.py, not asserted."""
    r = bc.run(bc.TORQUE_COMMAND, 0, seconds=SECONDS, window=WINDOW)
    assert r.torque_p95 <= bc.TORQUE_P95


def test_stance_measurement_adds_the_rolling_foot():
    """Issue #94: the stance measurement is the foot centre's motion relative to the hip plus the foot radius times
    the shank's pitch rate (the foot rolls on the ground)."""
    leg, dt = "FL", bc.DT
    q0, q1 = (0.0, 0.30, -0.80), (0.0, 0.26, -0.78)
    c = bc.LegController(leg)
    inp = dict(leg=leg, acc=(0.0, 0.0, bc.G), gyro=(0.0, 0.0, 0.0), contact=True, phase_cs=(1.0, 0.0),
               command=(0.0, 0.0, 0.0), neighbours={})
    c.step(bc.LegInput(joint_pos=q0, **inp))
    c.step(bc.LegInput(joint_pos=q1, **inp))
    _, _, hx, _ = bc.leg_frame(leg)
    fx0, fx1 = bc.foot_position(leg, q0)[0] + hx, bc.foot_position(leg, q1)[0] + hx
    shank0, shank1 = q0[1] + q0[2], q1[1] + q1[2]
    v = -(fx1 - fx0) / dt + bc.ROLL_RADIUS * ((shank1 - shank0) / dt)
    assert c.vel[0] == pytest.approx(dt / (bc.VEL_TAU + dt) * v, rel=1e-12)
    assert bc.ROLL_RADIUS == pytest.approx(0.012)  # the foot's radius (erg.design.FOOT_RADIUS)


def mean_estimate(servo, seconds=10.0):
    e = env_mod.Env(servo=servo, randomize_servo=False)
    obs = e.reset(0, (0.1, 0.0, 0.0))
    ctrls = [bc.LegController(leg) for leg in bc.LEGS]
    est, true = [], []
    for k in range(round(seconds / bc.DT)):
        outs = [c.step(bc.leg_input(obs[i], leg)) for i, (c, leg) in enumerate(zip(ctrls, bc.LEGS))]
        obs = e.step(np.array([list(o.targets) + list(o.message) for o in outs]), freq=[o.freq for o in outs])
        if k * bc.DT >= WINDOW:
            est.append(np.mean([c.vel[0] for c in ctrls]))
            true.append(reward.heading_velocity(e.data)[0][0])
    return float(np.mean(est)), float(np.mean(true))


def test_velocity_estimate_with_the_servo_model(monkeypatch):
    """Issue #94: with the servo model the estimate is within 5 % of the torso's true forward speed at 0.1 m/s;
    without the rolling correction it read about 7 % low."""
    est, true = mean_estimate(ServoParams())
    assert abs(est - true) <= 0.05 * true
    monkeypatch.setattr(bc, "ROLL_RADIUS", 0.0)
    est0, true0 = mean_estimate(ServoParams())
    assert est0 < 0.95 * true0
