"""Sine, cosine, arctangent and arccosine from additions, multiplications, divisions and square roots only
(Issue #54), so that the PC (Python), the WebAssembly runtime and the ESP32-S3 (C, firmware/leg-esp32/main/xmath.c)
compute the base gait and the gait phase bit for bit the same. These operations are correctly rounded by IEEE 754
on every platform; library functions such as math.sin are not, and differ between C libraries.

The C version repeats every operation in the same order; tests/test_firmware_host.py checks the bits.
tests/test_exactmath.py checks the accuracy against the library functions.
"""

from math import sqrt  # correctly rounded (IEEE 754), as sqrt in C

PI = 3.141592653589793
HALF_PI = 1.5707963267948966

# Taylor coefficients in z = x^2, highest power first, for |x| <= pi/4 after reduction:
# sin x = x + x z P(z), P = -1/3! + z/5! - ... ; cos x = 1 + z Q(z), Q = -1/2! + z/4! - ...
SIN_COEF = [-1.0 / 1307674368000.0, 1.0 / 6227020800.0, -1.0 / 39916800.0, 1.0 / 362880.0, -1.0 / 5040.0,
            1.0 / 120.0, -1.0 / 6.0]
COS_COEF = [1.0 / 20922789888000.0, -1.0 / 87178291200.0, 1.0 / 479001600.0, -1.0 / 3628800.0, 1.0 / 40320.0,
            -1.0 / 720.0, 1.0 / 24.0, -1.0 / 2.0]
# arctangent: x + x z R(z), R = -1/3 + z/5 - ... - z^10/23, highest power first
ATAN_COEF = [-1.0 / 23.0, 1.0 / 21.0, -1.0 / 19.0, 1.0 / 17.0, -1.0 / 15.0, 1.0 / 13.0, -1.0 / 11.0, 1.0 / 9.0,
             -1.0 / 7.0, 1.0 / 5.0, -1.0 / 3.0]


def _horner(coef: list, z: float) -> float:
    p = coef[0]
    for c in coef[1:]:
        p = p * z + c
    return p


def sincos(x: float) -> tuple[float, float]:
    """(sin x, cos x), for |x| below about 1e9."""
    q = float(int(x / HALF_PI + 0.5) if x >= -HALF_PI / 2 else -int(-x / HALF_PI + 0.5))  # nearest multiple
    r = x - q * HALF_PI
    z = r * r
    s = r + r * (z * _horner(SIN_COEF, z))
    c = 1.0 + z * _horner(COS_COEF, z)
    k = int(q) % 4
    if k == 0:
        return s, c
    if k == 1:
        return c, -s
    if k == 2:
        return -s, -c
    return -c, s


def sin(x: float) -> float:
    return sincos(x)[0]


def cos(x: float) -> float:
    return sincos(x)[1]


def atan(x: float) -> float:
    """Three halvings of the angle (atan x = 2 atan(x / (1 + sqrt(1 + x^2)))), then the Taylor series."""
    for _ in range(3):
        x = x / (1.0 + sqrt(1.0 + x * x))
    z = x * x
    return 8.0 * (x + x * (z * _horner(ATAN_COEF, z)))


def atan2(y: float, x: float) -> float:
    if x > 0.0:
        return atan(y / x)
    if x < 0.0:
        return atan(y / x) + (PI if y >= 0.0 else -PI)
    if y > 0.0:
        return HALF_PI
    if y < 0.0:
        return -HALF_PI
    return 0.0


def acos(c: float) -> float:
    return atan2(sqrt(1.0 - c * c), c)
