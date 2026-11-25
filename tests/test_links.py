"""Leg messages in Dyne's wire format (Issue #33).

Pass criteria:
- the bytes of a message made by erg's WebAssembly module (which uses Dyne's dyne-wire) equal, bit for bit, the same
  message built in Python from the layout in dyne-wire's documentation; the message is 120 bytes;
- decoding gives back the sender, step, time stamp and the values bit for bit, including -0.0, infinities, NaN and
  the smallest numbers;
- a message of another length or kind is rejected.
"""

import numpy as np
import pytest

from erg import links


@pytest.fixture(scope="module")
def lk():
    return links.Links()


def cases():
    rng = np.random.Generator(np.random.PCG64(33))
    out = [rng.normal(size=links.N_VALUES) for _ in range(20)]
    special = [0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324, -1.7976931348623157e308, 1.0, -1.0, 0.5, 1e-300]
    out.append(np.array(special))
    return out


def test_bytes_match_the_wire_format(lk):
    for k, values in enumerate(cases()):
        src, step, t_us = k % 4, 1000 + k, 20_000 * (1000 + k)
        msg = lk.encode(src, step, t_us, values)
        assert len(msg) == links.MSG_LEN == 120
        assert msg == links.reference_bytes(src, step, t_us, values)


def test_decoding_gives_back_the_values_bit_for_bit(lk):
    for k, values in enumerate(cases()):
        src, step, t_us, back = lk.decode(lk.encode(k % 4, 7 + k, 140_000 + k, values))
        assert (src, step, t_us) == (k % 4, 7 + k, 140_000 + k)
        assert np.array_equal(back.view(np.uint64), np.asarray(values, dtype="<f8").view(np.uint64))


def test_bad_messages_are_rejected(lk):
    msg = lk.encode(0, 1, 2, np.zeros(links.N_VALUES))
    with pytest.raises(ValueError):
        lk.decode(msg[:-8])  # truncated
    wrong_kind = bytes([msg[0], 0]) + msg[2:]  # IFACE_SCALAR header on a band payload
    with pytest.raises(ValueError):
        lk.decode(wrong_kind)
