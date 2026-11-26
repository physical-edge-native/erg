"""The virtual network between the legs (Issue #34), on Dyne's seeded VirtNet.

Pass criteria:
- the link settings give the right legs: none; the ring front-left, front-right, hind-right, hind-left (each leg
  with two neighbours, never the diagonal leg); all;
- the same seed gives the same lost and delayed messages on every run (the full sequence of fates, compared exactly);
  another seed gives another sequence;
- a message is delivered one control step later at the earliest, and with a base delay of one control period and
  jitter it arrives within the steps that delay allows;
- the measured loss rate over N = 100,000 messages is within 5 standard deviations of the setting, the standard
  deviation of a binomial proportion, sqrt(p (1 - p) / N): about 0.0047 for p = 0.1 and 0.0072 for p = 0.3. A
  correct network fails this with a probability below 1e-6;
- routing one control step's messages in one call (Network.step) gives exactly the same fates as message by message;
- the time of routing one control step's messages in one call (all legs, "all" links: 12 messages) is measured and
  printed (pytest -s); it must stay below 1 ms, 5 % of the 20 ms control period. Measured on 2026-11-25 in a cloud
  container: see the pull request.
"""

import time

import numpy as np
import pytest

from erg import links

STEPS = 2000


def fates(seed, links_setting="all", loss=0.2, lat_us=10_000, jitter_us=15_000):
    net = links.Network(links_setting, loss, lat_us, jitter_us, seed)
    out = []
    for step in range(STEPS):
        for src in range(links.N_LEGS):
            for dst in net.targets(src):
                out.append(-1 if (at := net.route(src, dst, step)) is None else at - step)
    return out


def test_link_settings():
    none = links.Network("none")
    ring = links.Network("neighbours")
    every = links.Network("all")
    for leg in range(links.N_LEGS):
        assert none.targets(leg) == []
        assert every.targets(leg) == [i for i in range(links.N_LEGS) if i != leg]
    # FL=0, FR=1, HL=2, HR=3; ring FL-FR-HR-HL-FL
    assert [ring.targets(i) for i in range(4)] == [[1, 2], [0, 3], [0, 3], [1, 2]]


def test_same_seed_same_fates():
    a, b = fates(7), fates(7)
    assert a == b
    assert a != fates(8)


def test_one_call_per_step_gives_the_same_fates():
    net = links.Network("all", 0.2, 10_000, 15_000, 7)
    batched = [-1 if at is None else at - step for step in range(STEPS) for _, _, at in net.step(step)]
    assert batched == fates(7)


def test_delivery_steps():
    net = links.Network("all", 0.0, 0, 0, 1)
    assert all(net.route(0, 1, s) == s + 1 for s in range(100))
    # base delay one period (20 ms), jitter 0..2x5 ms: arrival 20-30 ms after sending -> step + 1 or step + 2
    delays = [d for d in fates(3, loss=0.0, lat_us=20_000, jitter_us=5_000)]
    assert set(delays) <= {1, 2} and {1, 2} <= set(delays)


@pytest.mark.parametrize("p", [0.1, 0.3])
def test_loss_rate(p):
    n = 100_000
    net = links.Network("all", p, 0, 0, 11)
    lost = sum(net.route(0, 1, s) is None for s in range(n))
    sd = np.sqrt(p * (1 - p) / n)
    assert abs(lost / n - p) < 5 * sd, (lost / n, p, sd)


def test_time_per_control_step():
    net = links.Network("all", 0.1, 10_000, 5_000, 5)
    steps = 2000
    t = time.perf_counter()
    for step in range(steps):
        net.step(step)
    per_step = (time.perf_counter() - t) / steps
    print(f"\nrouting 12 messages per control step: {per_step * 1e6:.1f} us")
    assert per_step < 1e-3
