/* Sine, cosine, arctangent and arccosine from correctly rounded operations only: the same operations in the same
 * order as erg/exactmath.py, so the firmware and the training environment compute the same bits (Issue #54). */
#pragma once

void xm_sincos(double x, double *s, double *c);
double xm_atan2(double y, double x);
double xm_acos(double c);
/* x modulo 1 with Python's sign convention (result in [0, 1) for a positive modulus) */
double xm_mod1(double x);
