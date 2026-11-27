# Leg firmware, stage 1: one leg on the bench

Date: 2026-11-26. Issue: #54. Decided by the maintainer where stated; the choices made with the implementation are
to be reviewed. This record was written with an AI assistant. Measurements on the hardware are added when the
parts arrive.

## Decided (2026-11-26)

| Point | Decision | Reason |
|---|---|---|
| Language and framework | C with ESP-IDF, reusing what Dyne's ESP32 firmware does (WAMR interpreter, same version range) | erg depends on Dyne where it can; Dyne already runs WebAssembly solvers on the ESP32 bit for bit with the PC. Rust with esp-hal was the other option |
| Base gait's trigonometry (2026-11-27, after #51 gave the controllers a base gait) | Sine, cosine, arctangent and arccosine written with additions, multiplications, divisions and square roots only (`erg/exactmath.py` in training, `main/xmath.c` in the firmware, the same operations in the same order) | These operations are correctly rounded on every platform, so the gait phase and the base gait are bit for bit the same in training and on the ESP32-S3; library functions are not. The other option was the library functions with a stated tolerance. Accuracy: within 4e-15 of the library over the ranges used; the trot's joint targets change by less than 1e-12 rad |

## Choices made with the implementation (to be reviewed)

| Point | Choice | Reason |
|---|---|---|
| Reuse of Dyne's firmware | The same WAMR component and version range, and the same way of calling a solver; no source file copied | Dyne's firmware is an application, not a reusable component. A proposal to Dyne to offer its WebAssembly loading as an ESP-IDF component is drafted for the maintainer |
| Pure parts apart | Servo protocol (`sts.c`) and observation and conversions (`leg.c`) have no ESP-IDF dependency and are tested on the PC bit for bit against Python | What can be checked without hardware is checked in CI |
| Observation layout | Generated from `erg/env.py` into `obs_layout.h`; a test fails if they differ | The firmware cannot drift from the training environment |
| Floating point | Double precision, no fused multiply-add (`-ffp-contract=off`) | The same bits on the PC and the ESP32-S3 (whose double precision is emulated in software, rounding every operation) |
| Servo goals | One sync-write packet for the three servos per period; states read one servo at a time | Reading one at a time gives the per-servo time asked for; sync read can be tried after the first measurements |
| Joint load | The servo's load reading (tenths of a percent of its maximum) / 1000 as the observation's torque / stall torque | The nearest reading the servo gives; its relation to the model's torque is to be measured |
| Settings | Pins, ids, leg index and calibration in `menuconfig` (the device's `sdkconfig`, not committed) | Device-specific values stay out of the public repository |

## What the bench measures (to be recorded here)

- Servo bus time for reading the three servos, and for the goal packet.
- Policy time in `mlp-actor` per evaluation on the ESP32-S3 (6,008 parameters since #51 gave each leg a gait clock: 67 → 48 → 48 → 8; the full variant of `mlp-actor`, up to 8,192 parameters; double precision emulated).
- The loop's total time and the jitter of its start against the 20 ms schedule.
- Whether the policy's outputs are bit for bit those of the PC (`check: bit-exact`).

Pass condition stated in advance: the loop's total time, at its maximum over 5 s, under 20 ms; outputs bit-exact.
