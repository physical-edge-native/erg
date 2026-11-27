#include "xmath.h"

#include <math.h>

#define XM_PI 3.141592653589793
#define XM_HALF_PI 1.5707963267948966

static const double SIN_COEF[] = {-1.0 / 1307674368000.0, 1.0 / 6227020800.0, -1.0 / 39916800.0, 1.0 / 362880.0,
                                  -1.0 / 5040.0, 1.0 / 120.0, -1.0 / 6.0};
static const double COS_COEF[] = {1.0 / 20922789888000.0, -1.0 / 87178291200.0, 1.0 / 479001600.0,
                                  -1.0 / 3628800.0, 1.0 / 40320.0, -1.0 / 720.0, 1.0 / 24.0, -1.0 / 2.0};
static const double ATAN_COEF[] = {-1.0 / 23.0, 1.0 / 21.0, -1.0 / 19.0, 1.0 / 17.0, -1.0 / 15.0, 1.0 / 13.0,
                                   -1.0 / 11.0, 1.0 / 9.0, -1.0 / 7.0, 1.0 / 5.0, -1.0 / 3.0};

static double horner(const double *coef, int n, double z)
{
    double p = coef[0];
    for (int i = 1; i < n; i++) p = p * z + coef[i];
    return p;
}

void xm_sincos(double x, double *s_out, double *c_out)
{
    double q = (x >= -XM_HALF_PI / 2) ? (double)(long long)(x / XM_HALF_PI + 0.5)
                                      : -(double)(long long)(-x / XM_HALF_PI + 0.5);
    double r = x - q * XM_HALF_PI;
    double z = r * r;
    double s = r + r * (z * horner(SIN_COEF, 7, z));
    double c = 1.0 + z * horner(COS_COEF, 8, z);
    int k = (int)((((long long)q % 4) + 4) % 4);
    switch (k) {
    case 0: *s_out = s; *c_out = c; break;
    case 1: *s_out = c; *c_out = -s; break;
    case 2: *s_out = -s; *c_out = -c; break;
    default: *s_out = -c; *c_out = s; break;
    }
}

static double xm_atan(double x)
{
    for (int i = 0; i < 3; i++) x = x / (1.0 + sqrt(1.0 + x * x));
    double z = x * x;
    return 8.0 * (x + x * (z * horner(ATAN_COEF, 11, z)));
}

double xm_atan2(double y, double x)
{
    if (x > 0.0) return xm_atan(y / x);
    if (x < 0.0) return xm_atan(y / x) + (y >= 0.0 ? XM_PI : -XM_PI);
    if (y > 0.0) return XM_HALF_PI;
    if (y < 0.0) return -XM_HALF_PI;
    return 0.0;
}

double xm_acos(double c)
{
    return xm_atan2(sqrt(1.0 - c * c), c);
}

double xm_mod1(double x)
{
    double r = fmod(x, 1.0);
    if (r != 0.0 && r < 0.0) r += 1.0;
    return r;
}
