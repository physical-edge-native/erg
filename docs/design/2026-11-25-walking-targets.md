# Walking commands, test ground and pass conditions

Date: 2026-11-25. Issue: #27. Decided by the maintainer; this record was written with an AI assistant.

The comparison of split and central control ([2026-11-25-expected-differences.md](2026-11-25-expected-differences.md))
measures walking at commanded speeds. This record fixes, before any training, the commands, when a commanded speed
counts as reached, and the uneven ground used for evaluation. The values are in `erg/env.py`; `tests/test_env.py`
checks them.

## Decided (2026-11-25)

| Point | Decision |
|---|---|
| Commands | Forward speed in [-0.23, +0.23] m/s, sideways speed in [-0.12, +0.12] m/s, turning speed in [-0.8, +0.8] rad/s, each drawn uniformly, in both directions, from the episode's seed (`erg.env.COMMAND_MAX`, `draw_command`) |
| A commanded speed counts as reached | The robot walks 10 s without a fall, and its mean speed over those 10 s is within 20 % of the command (`REACH_SECONDS`, `REACH_TOLERANCE`) |
| Uneven test ground | Blocks from 0 to 2 cm high (about a tenth of the leg length), laid out from a seed; used only for evaluation, never in training (`TERRAIN_MAX_HEIGHT`). The ground itself is built in a later issue |
| Scope | Acrobatic motion (jumps, flips) stays outside erg: it needs actuators that control torque directly, which raise the cost of the robot by about five to twenty times, against erg's aim of a cheap robot (maintainer's decision, 2026-11-25). Actuators for an acrobatic quadruped at about ¥100,000 are proposed as a separate future task of the organization (physical-edge-native/pena-meta#9) |

## Why 0.23 m/s

The limit is set by the servos. Standing, the leg (hip flexion shaft to the sole) is about 0.16 m long.

- With the foot on the ground, walking at speed v turns the hip at about v / 0.16 m: 1.4 rad/s at 0.23 m/s. The servo
  is loaded then; at half its stall torque it turns at about 2.8 rad/s (its speed falls in a straight line from
  5.5 rad/s with no load to zero at stall).
- With the foot in the air, the leg is brought forward faster: if each foot is on the ground three quarters of the
  time (a slow walk), three times as fast, 4.3 rad/s. The leg in the air is hardly loaded, so the limit is the
  no-load speed, 5.5 rad/s.

| Option | Forward limit | Hip speed, foot on the ground | Hip speed, foot in the air | |
|---|---|---|---|---|
| A | 0.15 m/s | 0.9 rad/s | 2.8 rad/s | wide margin |
| **B** | **0.23 m/s** | **1.4 rad/s** | **4.3 rad/s** | **near the servo's limit. Chosen** |
| C | 0.33 m/s | 2.1 rad/s | 6.2 rad/s | above the no-load speed |

Near the limit is where delays between legs, the main weakness expected of split control, should matter most, so B
is the most informative choice that the servos can still follow. The sideways limit is about half of the forward
limit; turning at 0.8 rad/s moves the feet (about 0.1 m from the centre of the body) at about 0.08 m/s, below the
sideways limit.

In terms of the Froude number (speed squared over gravity times leg length, a common scale for comparing walking
across sizes), these speeds are slow walks: about 0.03 for 0.23 m/s. An earlier draft of the options derived them
from Froude numbers 0.1 to 0.5 with an arithmetic error; the corrected speeds for those Froude numbers (0.47 to
1.04 m/s) are beyond these servos.

## Other options considered

- Forward only (no backward walking): simpler, but commands in both directions are usual in training and cost
  nothing extra.
- Faster limits from the Froude number of ordinary animal walking (0.25): about 0.74 m/s, beyond the servos.
