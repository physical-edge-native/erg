"""The servo bench's serial-bus packets (Issue #88, erg/sts.py) against the leg firmware's (sts.c).

Pass criteria:
- ping, read, goal write and sync goal write packets made by erg.sts equal, byte for byte, those made by the
  firmware's C code compiled for the PC (the host fixture of tests/conftest.py), over the same cases as
  tests/test_firmware_host.py and a set of random ids, addresses and positions;
- replies are parsed to the same position, speed, load and error as the firmware's sts_parse_state, and bad replies
  are rejected with the same codes (-1 header, -2 id, -3 length, -4 checksum);
- the bench's longer read (position to current) parses the voltage, temperature and current at their addresses.
"""

import numpy as np
import pytest

from erg import sts


def test_packets_equal_the_firmware(host):
    rng = np.random.Generator(np.random.PCG64(88))
    cases = [("ping 1", sts.ping(1))]
    for _ in range(20):
        i, a, n, p = (int(rng.integers(0, 254)), int(rng.integers(0, 256)), int(rng.integers(1, 40)),
                      int(rng.integers(0, 4096)))
        cases += [(f"ping {i}", sts.ping(i)), (f"read {i} {a} {n}", sts.read(i, a, n)),
                  (f"goal {i} {p}", sts.write_goal(i, p))]
    for n in (1, 3, 12):
        ids = [int(x) for x in rng.integers(0, 254, n)]
        pos = [int(x) for x in rng.integers(0, 4096, n)]
        cases.append((f"sync {n} " + " ".join(f"{i} {p}" for i, p in zip(ids, pos)), sts.sync_write_goal(ids, pos)))
    out = host([c for c, _ in cases])
    assert bytes.fromhex(out[0]) == bytes.fromhex("ffff010201fb")  # Feetech's documented ping
    for (c, want), got in zip(cases, out):
        assert bytes.fromhex(got) == want, c


def test_replies_parse_as_the_firmware(host):
    rng = np.random.Generator(np.random.PCG64(87))
    replies = []
    for _ in range(30):
        i = int(rng.integers(0, 254))
        data = [int(x) for x in rng.integers(0, 256, 6)]
        replies.append((i, sts.reply(i, int(rng.integers(0, 256)), data)))
    good = sts.reply(1, 0, [10, 0, 20, 0, 30, 0])
    bad = [(1, good[:1] + b"\x00" + good[2:]), (2, good), (1, good[:-1] + bytes([good[-1] ^ 1])), (1, good[:-2])]
    out = host([f"parse {i} {r.hex()}" for i, r in replies + bad])
    for (i, r), got in zip(replies, out):
        s = sts.parse_state(r, i)
        assert list(map(int, got.split())) == [0, s.position, s.speed, s.load, s.error]
    for (i, r), got in zip(bad, out[len(replies):]):
        with pytest.raises(sts.ReplyError) as e:
            sts.parse_state(r, i)
        assert e.value.code == int(got.split()[0])


def test_bench_block():
    data = bytearray(sts.BENCH_BLOCK)
    data[0:2] = (3000).to_bytes(2, "little")
    data[2:4] = sts.to_sign_magnitude(-250, 15).to_bytes(2, "little")
    data[4:6] = sts.to_sign_magnitude(-512, 10).to_bytes(2, "little")
    data[sts.ADDR_PRESENT_VOLTAGE - sts.ADDR_PRESENT_POSITION] = 121
    data[sts.ADDR_PRESENT_TEMPERATURE - sts.ADDR_PRESENT_POSITION] = 41
    c = sts.ADDR_PRESENT_CURRENT - sts.ADDR_PRESENT_POSITION
    data[c:c + 2] = (77).to_bytes(2, "little")
    s = sts.parse_bench_block(sts.reply(5, 0x20, data), 5)
    assert (s.position, s.speed, s.load, s.error, s.voltage, s.temperature, s.current) == (3000, -250, -512, 0x20,
                                                                                         121, 41, 77)
