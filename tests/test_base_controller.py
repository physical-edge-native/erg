"""The base controller (Issue #71), against docs/design/2026-11-27-base-controller.md.

The pass conditions are checked here on seed 0 over 10 s (measured over 4-10 s), one command of each kind, so that
the suite stays short on the CI's three systems (each run takes about a second); scripts/base_controller.py runs the
full set (five seeds, 20 s, every command) and its output is recorded in the design record. The torque condition
fails with the current servo model (design record, Results): it is kept as a strict expected failure, so that the
test reports it, and turns red when it starts to pass and the record has to be updated.
"""

import inspect
import re

import numpy as np
import pytest

from erg import base_controller as bc
from erg import env as env_mod
from erg import links, robot

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


@pytest.mark.xfail(strict=True, reason="condition 6 fails with the current servo model (design record, Results)")
def test_torque_condition():
    r = bc.run((0.2, 0.0, 0.0), 0, seconds=SECONDS, window=WINDOW)
    assert r.torque_p95 <= bc.TORQUE_P95
