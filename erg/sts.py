"""Feetech STS serial bus protocol in Python, for the servo bench (Issue #88).

The same packets as the leg firmware's firmware/leg-esp32/main/sts.c (tests/test_sts.py checks them byte for byte
against the compiled C code):

    packet: 0xFF 0xFF id length instruction parameters... checksum
    length = number of parameters + 2; checksum = ~(id + length + instruction + parameters) & 0xFF
    reply:  0xFF 0xFF id length error data... checksum

Two-byte values are little-endian. Speed and load are sign-magnitude: bit 15 (speed) or bit 10 (load) is the sign.

The bench also reads the registers after the load (voltage, temperature, current) and writes single registers
(torque enable). Register addresses are from Feetech's STS memory table and, like the firmware's, are to be checked
on a real servo before relying on them (the bench run, Issue #87, checks them first: see the operator's guide in
docs/design/2026-11-27-servo-model.md).
"""

from dataclasses import dataclass

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_WRITE = 0x83
BROADCAST_ID = 0xFE

ADDR_P_GAIN = 0x15  # position loop P, D, I gains (one byte each), as set in the servo's EEPROM
ADDR_TORQUE_ENABLE = 0x28
ADDR_GOAL_POSITION = 0x2A  # goal position (2), goal time (2), goal speed (2)
ADDR_PRESENT_POSITION = 0x38  # present position (2), speed (2), load (2)
ADDR_PRESENT_VOLTAGE = 0x3E  # 0.1 V per unit
ADDR_PRESENT_TEMPERATURE = 0x3F  # degrees Celsius
ADDR_PRESENT_CURRENT = 0x45  # 2 bytes; 6.5 mA per unit (Feetech's table; unverified)
BENCH_BLOCK = ADDR_PRESENT_CURRENT + 2 - ADDR_PRESENT_POSITION  # 15 bytes: position to current
STEPS_PER_TURN = 4096
CURRENT_PER_UNIT = 0.0065  # A


def checksum(packet: bytes) -> int:
    """Over the id to the last parameter (skips the two 0xFF and the checksum itself)."""
    return ~sum(packet[2:-1]) & 0xFF


def _packet(id_: int, inst: int, params) -> bytes:
    params = bytes(params)
    if len(params) + 2 > 255:
        raise ValueError("too many parameters")
    p = bytearray([0xFF, 0xFF, id_, len(params) + 2, inst, *params, 0])
    p[-1] = checksum(p)
    return bytes(p)


def ping(id_: int) -> bytes:
    return _packet(id_, INST_PING, [])


def read(id_: int, addr: int, n: int) -> bytes:
    return _packet(id_, INST_READ, [addr, n])


def write(id_: int, addr: int, data) -> bytes:
    return _packet(id_, INST_WRITE, [addr, *data])


def _goal(position: int) -> list[int]:
    if not 0 <= position <= 0xFFFF:
        raise ValueError("position out of range")
    return [position & 0xFF, position >> 8, 0, 0, 0, 0]  # goal time and speed 0: move as fast as allowed


def write_goal(id_: int, position: int) -> bytes:
    """Goal position (steps), with goal time and speed 0 (sts_write_goal)."""
    return write(id_, ADDR_GOAL_POSITION, _goal(position))


def sync_write_goal(ids, positions) -> bytes:
    """One packet setting the goal positions of several servos (sts_sync_write_goal)."""
    params = [ADDR_GOAL_POSITION, 6]
    for i, p in zip(ids, positions, strict=True):
        params += [i, *_goal(p)]
    return _packet(BROADCAST_ID, INST_SYNC_WRITE, params)


def sign_magnitude(raw: int, sign_bit: int) -> int:
    mag = raw & ((1 << sign_bit) - 1)
    return -mag if raw & (1 << sign_bit) else mag


def to_sign_magnitude(value: int, sign_bit: int) -> int:
    mag = min(abs(int(value)), (1 << sign_bit) - 1)
    return mag | (1 << sign_bit) if value < 0 else mag


class ReplyError(Exception):
    """A reply with a bad header (-1), wrong id (-2), wrong length (-3) or bad checksum (-4); the code is that of
    sts_parse_state."""

    def __init__(self, code: int):
        super().__init__({-1: "short or bad header", -2: "wrong id", -3: "wrong length", -4: "bad checksum"}[code])
        self.code = code


def parse_reply(reply: bytes, id_: int, n: int) -> tuple[int, bytes]:
    """Error byte and the n data bytes of a reply to a read of n bytes (raises ReplyError, with the checks in the
    order of sts_parse_state)."""
    if len(reply) < 6 or reply[0] != 0xFF or reply[1] != 0xFF:
        raise ReplyError(-1)
    if reply[2] != id_:
        raise ReplyError(-2)
    if reply[3] != n + 2 or len(reply) != n + 6:
        raise ReplyError(-3)
    if checksum(reply) != reply[-1]:
        raise ReplyError(-4)
    return reply[4], bytes(reply[5:-1])


def reply(id_: int, error: int, data) -> bytes:
    """A servo's reply (for the simulated servo)."""
    return _packet(id_, error, data)


@dataclass
class State:
    position: int  # steps, 0..4095
    speed: int  # steps per second, signed
    load: int  # tenths of a percent of the maximum torque, signed
    error: int
    voltage: int | None = None  # 0.1 V
    temperature: int | None = None  # degrees Celsius
    current: int | None = None  # CURRENT_PER_UNIT, signed


def parse_state(reply_: bytes, id_: int) -> State:
    """The reply to read(id, ADDR_PRESENT_POSITION, 6), as sts_parse_state."""
    err, d = parse_reply(reply_, id_, 6)
    return State(d[0] | d[1] << 8, sign_magnitude(d[2] | d[3] << 8, 15), sign_magnitude(d[4] | d[5] << 8, 10), err)


def parse_bench_block(reply_: bytes, id_: int) -> State:
    """The reply to read(id, ADDR_PRESENT_POSITION, BENCH_BLOCK): position, speed, load, voltage, temperature and
    current."""
    err, d = parse_reply(reply_, id_, BENCH_BLOCK)
    v = ADDR_PRESENT_VOLTAGE - ADDR_PRESENT_POSITION
    c = ADDR_PRESENT_CURRENT - ADDR_PRESENT_POSITION
    return State(d[0] | d[1] << 8, sign_magnitude(d[2] | d[3] << 8, 15), sign_magnitude(d[4] | d[5] << 8, 10), err,
                 voltage=d[v], temperature=d[v + 1], current=sign_magnitude(d[c] | d[c + 1] << 8, 15))
