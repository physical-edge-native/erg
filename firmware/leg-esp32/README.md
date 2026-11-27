# Leg firmware for the ESP32-S3 (stage 1: one leg on the bench)

Issue #54. Runs one leg of the erg robot on a Seeed Studio XIAO ESP32-S3: three Feetech STS3215 servos on their
serial bus (through a Waveshare Bus Servo Adapter (A)), an LSM6DSOX IMU on I2C and the foot microswitch on a GPIO.
Every 20 ms it reads the sensors, builds the leg's observation in the order of the training environment
(`erg/env.py`, including the leg's gait phase), runs the leg's base controller (`erg/base_controller.py`, #85:
phase coupling, velocity estimate, speed integral, foot placement, attitude, inverse kinematics) and sends its three
joint goals; then it advances the leg's gait clock by the controller's frequency correction. Every two periods
(25 Hz) it runs the policy in Dyne's `mlp-actor` (WebAssembly, interpreted by WAMR as in Dyne's own ESP32 firmware),
whose six outputs adjust the base controller (frequency, foot-placement residuals, step height, gains; #82). The
adjustments are ramped linearly from the previous policy step's over the two periods, as in training since #86
(#95; `POLICY_INTERPOLATE` in `obs_layout.h`, 0 to hold them instead). Without a policy the base controller walks alone. It sets the knee servo's torque limit from the knee angle
(#81). It prints the time of each stage and the loop's jitter.

**Needs the hardware** to run. Without hardware, CI checks that it builds with ESP-IDF (`.github/workflows/ci.yml`,
job "firmware") and the parts that do not touch the hardware are tested on the PC bit for bit against Python
(`tests/test_firmware_host.py`: servo packets, conversions, observation, gait clock, base gait, joint targets,
knee torque limit, and the base controller with the policy's adjustments over recorded walks with the neighbours'
messages). The trigonometry uses arithmetic-only functions (`main/xmath.c`, the same as `erg/exactmath.py` used in
training), so these are bit for bit the same as in training.

| File | What |
|---|---|
| `main/sts.c` | Servo bus packets (ping, read, write, sync write) and replies |
| `main/leg.c` | Sensor values to the observation; gait clock and base gait; policy output to servo goals |
| `main/base_ctrl.c` | The base controller of one leg and the policy's adjustments (as `erg/base_controller.py`) |
| `main/xmath.c` | Sine, cosine, arctangent, arccosine from correctly rounded operations only (as `erg/exactmath.py`) |
| `main/obs_layout.h` | Observation layout and the constants of the base gait, the base controller and the knee limit, generated from Python by `scripts/gen_firmware_layout.py` (a retune only regenerates it) |
| `main/policy.c` | Dyne's `mlp-actor` in WAMR |
| `main/main.c` | Drivers, the 20 ms loop, timing, the start-up bit check |

## Build and flash (ESP-IDF 5.5)

```bash
uv run python scripts/make_firmware_files.py                # optional: mlp-actor, a policy and the check file
cd firmware/leg-esp32
idf.py set-target esp32s3
idf.py menuconfig                                           # "erg leg": pins, servo ids, leg index, calibration
idf.py build flash monitor
```

`make_firmware_files.py` without `--agent` embeds an untrained policy of the trained one's shape, enough to measure
time and check bits; with `--agent runs/.../agent.pt` it embeds a trained per-leg actor that adjusts the base
controller (`scripts/train.py --form split`, the default action "base"). Enable
"Embed Dyne's mlp-actor, the policy and the check file" in menuconfig to use them.

At start the firmware prints `check: bit-exact` if the policy's outputs on the ESP32-S3 equal those of Dyne's
`mlp-actor` on the PC for the same observations, and the policy's time per evaluation.

## Settings kept out of the repository

Pins, servo ids, the leg index and the calibration (servo position at joint angle 0 and direction of each joint)
are set in `menuconfig` and kept in the device's `sdkconfig`, which is not committed. This stage uses no Wi-Fi; no
network settings, keys or device addresses belong in the repository.

## Not yet done

- Messages between legs (ESP-NOW): a later issue. Until then the base controller hears no neighbour; its message
  values (velocity estimate and speed integral) are computed and kept for sending.
- The IMU's mounting (`s_mount` in `main.c`) is identity until the board's orientation on the torso is fixed.
- Register addresses and the sign conventions of the servo's speed and load are from Feetech's memory table, to be
  checked on a real servo.
