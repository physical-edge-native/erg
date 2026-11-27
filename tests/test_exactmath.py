"""Sine, cosine, arctangent and arccosine from IEEE 754 correctly rounded operations only (Issue #54).

Pass criteria: within 4e-15 of Python's math library (which is not bit-exact across platforms) for sine and cosine
over [-20, 20] rad and for the arctangent of two numbers and the arccosine over their whole ranges; the trot gait's
targets with these functions within 1e-12 rad of those with the library functions. The bit-for-bit agreement with
the firmware's C version is checked in tests/test_firmware_host.py.
"""

import math

import numpy as np

from erg import exactmath as xm
from erg import gait


def test_sin_cos():
    for x in np.linspace(-20, 20, 40001):
        s, c = xm.sincos(float(x))
        assert abs(s - math.sin(x)) < 4e-15 and abs(c - math.cos(x)) < 4e-15


def test_atan2_acos():
    for y, x in np.random.default_rng(0).normal(size=(20000, 2)) * 3:
        assert abs(xm.atan2(y, x) - math.atan2(y, x)) < 4e-15
    for y, x in [(0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (-0.5, -2.0)]:
        assert abs(xm.atan2(y, x) - math.atan2(y, x)) < 4e-15
    for c in np.linspace(-1, 1, 20001):
        assert abs(xm.acos(float(c)) - math.acos(c)) < 4e-15


def test_gait_targets_match_the_library():
    def library(s, stride):
        x, z = gait.foot_path(s, stride)
        x = -stride / 2 + stride * ((s - 0.5) / 0.5) + gait.FOOT_X if s >= 0.5 else x
        z = -gait.STAND_DEPTH + gait.STEP_HEIGHT * math.sin(math.pi * (s - 0.5) / 0.5) if s >= 0.5 else z
        r2 = x * x + z * z
        cc = (r2 - gait.THIGH**2 - gait.SHANK**2) / (2 * gait.THIGH * gait.SHANK)
        knee = -math.acos(max(-1.0, min(1.0, cc)))
        flex = math.atan2(-x, -z) - math.atan2(gait.SHANK * math.sin(knee), gait.THIGH + gait.SHANK * math.cos(knee))
        return np.array([0.0, flex, knee])

    for s in np.linspace(0, 0.999, 500):
        for stride in (0.0, 0.02, -0.02, 0.04):
            assert np.max(np.abs(gait.leg_targets(float(s), stride) - library(float(s), stride))) < 1e-12
