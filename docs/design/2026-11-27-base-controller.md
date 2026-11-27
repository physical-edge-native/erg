# Base controller in MuJoCo: design and pass conditions

Date: 2026-11-27. Issue: #71. Builds on body L1 (#59) and the redesign record (2026-11-27-redesign.md, control
architecture item 1, and step 1 of "Separating causes when learning fails"). This record was written with an AI
assistant. The pass conditions below were written and committed **before** the code that measures them, as the
repository's rules ask; the results are added in a later section, with nothing in the conditions changed.

## What it is

A controller that walks the robot with no learning, the same for the per-leg and the central forms of control. It
runs on each leg, from that leg's own sensors (joint angles and speeds, the leg's IMU, foot contact) and the
messages of its two neighbours on the ring front-left, front-right, hind-right, hind-left (the ring of
`erg/links.py`). A learned policy later adjusts it (redesign record, item 2).

Per leg:

1. **Phase oscillator.** The leg's own gait clock (`erg/env.py`) is the oscillator. The controller sets the clock's
   frequency correction from its neighbours' phases, which arrive in their messages: the coupling pulls each
   neighbour towards half a period away, so the trot (diagonal legs together) is the stable solution on the ring.
2. **Foot placement** by the capture-point rule (Raibert's rule): the foot lands ahead of its neutral point by
   half the stance's travel plus k (v_estimated - v_commanded), with k = 1/ω, ω = sqrt(g / h), forward and sideways.
   With the centre of mass 0.142 m above the ground (standing, body L1), ω = 8.3 rad/s and 1/ω = 0.12 s.
3. **Body velocity estimate** from what a leg can know: while its foot is on the ground, the foot's velocity relative
   to the hip from the leg's own joint angles (forward kinematics, differenced over one control period), with the
   rotation of the body from the leg's gyroscope removed; in the air, the estimates of the neighbours whose feet
   are on the ground, from their messages; a low-pass filter on top. Each leg sends its estimate to its neighbours.
4. **Body attitude** by leg-length differences: roll and pitch from the leg's IMU (the accelerometer's gravity
   direction blended with the integrated gyroscope); each leg lengthens or shortens by the height error of its own
   hip.
5. **Swing-leg retraction**: near the end of the swing the foot moves backwards, so it meets the ground with less
   forward speed relative to it.
6. Inverse kinematics of the leg (abduction, then the two-link thigh and shank as in `erg/gait.py`, with the foot
   path centred at `FOOT_X` from the hip flexion shaft).

The arithmetic uses `erg/exactmath.py` (and square roots, which are correctly rounded), plain double-precision
operations and no library trigonometry, so that it can later be ported to the leg firmware bit for bit, as the base
gait was (#54).

Inputs a policy may adjust later, all zero by default: the clock's frequency correction, foot-placement residuals
forward and sideways, a change of step height, and changes of the placement and attitude gains.

The joint targets go to the model's current position actuators (stiffness 10 N·m/rad, torque limited to the stall
torque 2.94 N·m, damping stall torque / no-load speed). The servo model of #70 is not merged yet; the controller
outputs joint targets only, so the servo model can replace the actuators without changing it, and these conditions
are run again then.

## Pass conditions (stated before the run)

All in MuJoCo, body L1, `models/quadruped.xml`, per-leg mode of `erg/env.py` with messages between neighbours only,
through Dyne's wire format, with no loss and no delay beyond the one control period of the virtual network.
Each run starts standing with the joints offset by up to ±0.05 rad (uniform, the environment's reset noise) from
seeds 0–4, and lasts 20 s. Speeds are measured in the body's heading frame (the torso's velocity rotated by minus
its yaw angle), averaged over 5–20 s (the first 5 s let the robot start walking and the estimates settle). The yaw
rate is the change of the yaw angle over 5–20 s divided by 15 s.

A **fall** is the torso tilting more than 45° from upright, or its centre coming lower than 0.6 of its standing
height (0.15 m, so below 0.09 m), at any time of a run.

| # | Condition | Commands (forward m/s, sideways m/s, turning rad/s) | Pass when, for every seed |
|---|---|---|---|
| 1 | Forward speed tracked | (0.1, 0, 0), (0.2, 0, 0), (0.3, 0, 0) | mean forward speed within ±25 % of the command; mean sideways speed within ±0.05 m/s; mean yaw rate within ±0.1 rad/s |
| 2 | Turning followed | (0, 0, +0.5), (0, 0, -0.5), (0.15, 0, +0.5) | mean yaw rate of the command's sign and within ±50 % of it |
| 3 | Sideways followed | (0, +0.1, 0), (0, -0.1, 0) | mean sideways speed of the command's sign and within ±50 % of it; mean yaw rate within ±0.1 rad/s |
| 4 | No falls | all runs of conditions 1–3 | no fall |
| 5 | Small pushes | (0.2, 0, 0), with a sideways push at 10 s, to the left and to the right | no fall until the end of the run (20 s) |
| 6 | Torque | (0.2, 0, 0) | worst joint's 95th percentile of torque ÷ stall torque over 5–20 s ≤ 0.5 (the body target of the redesign record, now with the base controller) |
| 7 | Determinism | (0.2, 0, 0), seed 0 | two runs give the same joint targets bit for bit |
| 8 | Information rule | — | a leg's controller reads only its own sensors and its two neighbours' messages (below) |

The push is a force on the torso's centre of mass along the body's sideways axis, 3 N for 0.1 s: an impulse of
0.3 N·s, which changes the speed of the 1.2 kg robot by about 0.25 m/s sideways. That is "small": about the robot's
sideways command range (0.12 m/s) twice over, and well below what tips the standing robot over.

Reasons for the fractions:

- **±25 % forward.** The base controller has to bring the learned policy within reach of the command, not to track it
  sharply; the policy does that with its foot-placement residuals of about ±0.02 m per step (redesign record), which at
  half a period of 0.225 s per stance change the speed by up to about ±0.09 m/s, more than 25 % of 0.3 m/s
  (0.075 m/s). The learned policy's own target stays ±20 % (walking-targets record).
- **±50 % turning and sideways.** These come after forward speed in the requirements (R2 (2)) and the robot moves
  sideways only by abduction, which the trot uses little; the base controller has to move the right way at a useful
  rate, and the policy sharpens it.
- **0.6 of the standing height for a fall.** The torso's underside then sits about 0.07 m above the ground: the legs
  no longer hold it up.

What the check script reports for each command: mean forward and sideways speed (and the ratio to the command),
mean yaw rate, the largest tilt, the lowest torso height, falls, and the worst joint's 95th percentile and maximum of
torque ÷ stall torque.

The tests (`tests/test_base_controller.py`) run the same measures on seed 0 only and over 10 s (window 4–10 s), for
one command of each of conditions 1, 2, 3 and 5, so that the test suite stays short on the CI's three systems; the
full set above is run by `scripts/base_controller.py`, and its output is recorded below.

### The information rule (condition 8), how it is enforced

The controller of a leg is an object that gets, each control period, only a `LegInput`: that leg's own sensor values
and gait phase, the command, and the message slots of its two neighbours. The function that cuts this out of the
environment's observation keeps no reference to the environment, and takes the neighbours from a fixed table of the
ring. The tests check the table against Dyne's virtual network ("neighbours"), that changing the diagonal leg's
message or another leg's sensors does not change a leg's output, and that the whole run is the same bit for bit with
links "all" (where the diagonal leg's message would be readable) as with "neighbours".

## Results (2026-11-27)

Run with `uv run python scripts/base_controller.py` (five seeds, 20 s, Dyne's wire format) on the controller as
committed with this section. Nothing in the conditions above was changed after the run; condition 6 was changed
later by the maintainer, in a separate section below, with the first version and its failed result kept here.

| # | Condition | Result | Measured (over seeds 0–4) |
|---|---|---|---|
| 1 | Forward speed | pass | 0.1 m/s: 0.098–0.101 (0.98–1.01 of the command); 0.2 m/s: 0.195–0.202 (0.98–1.01); 0.3 m/s: 0.280–0.296 (0.93–0.99). Sideways drift ≤ 0.008 m/s, yaw drift ≤ 0.030 rad/s |
| 2 | Turning | pass | +0.5 rad/s: 0.374–0.376 (0.75); -0.5 rad/s: -0.381 to -0.382 (0.76); (0.15, 0, 0.5): 0.406–0.420 (0.81–0.84), forward 0.155–0.159 |
| 3 | Sideways | pass | +0.1 m/s: 0.102–0.106 (1.02–1.06); -0.1 m/s: -0.102 to -0.108; yaw drift ≤ 0.020 rad/s |
| 4 | No falls | pass | no fall in 40 runs; largest tilt 7.5°, lowest torso centre 0.143 m |
| 5 | Small pushes (0.3 N·s) | pass | no fall in 10 runs; largest tilt 4.9° |
| 6 | Torque at 0.2 m/s | **fail** | worst joint's 95th percentile 0.72–0.73 of the stall torque (limit 0.5); maximum 0.82–0.90 |
| 7 | Determinism | pass | the same joint targets bit for bit |
| 8 | Information rule | pass | `tests/test_base_controller.py` |

Torque at the other commands (worst joint's 95th percentile): 0.34–0.35 at 0.1 m/s, **1.00 at 0.3 m/s** (the servos
saturate: the speed is tracked, with no margin left), 0.31–0.32 turning on the spot, 0.60–0.62 for (0.15, 0, 0.5),
0.44–0.46 sideways.

Beyond the conditions (reported, not a condition; `--push-forces`): walking at 0.2 m/s, sideways pushes of 1.0 and
1.5 N·s gave no falls in 10 runs, 2.0 N·s gave 3 falls in 10, 2.5 N·s and more always fell.

### The failed condition: torque (a result, not loosened)

The hip flexion servo gives most of the torque (at 0.2 m/s, per joint: hip flexion 0.6–0.7, knee about 0.35,
abduction about 0.1). The model's servo turns its speed into a loss torque (joint damping = stall torque ÷ no-load
speed = 0.62 N·m·s/rad, a straight torque–speed line), and the swinging hip moves at about 3 rad/s at 0.2 m/s, so
about 1.9 N·m (0.6 of the stall torque) goes to the servo's own speed at the peak of each swing. This is set by the
speed of the body, not by the controller's details: changing the swing's speed profile, the retraction, the step
height, the period (±30 %), the stance fraction (0.45–0.6) and the attitude gain left the 95th percentile between
0.61 and 0.83 at 0.2 m/s.

The redesign record's target "≤ 0.5 at 0.2 m/s" was measured with the open-loop trot commanded at 0.2 m/s, which
walks at about 0.1 m/s (0.41 there). Walking at 0.2 m/s with the open-loop trot (commanded 0.35 m/s, reaching
0.19 m/s) gives 0.69, about what the base controller needs (0.72). So body L1 with this servo model meets the
torque target at about 0.1–0.15 m/s of actual speed, not at 0.2 m/s.

What I would change, for the maintainer to choose (it changes a body target of the redesign):

1. Run the conditions again with the servo model of #70 when it is merged: the loss torque above comes from the
   assumed straight torque–speed line of the seller's values, which the servo's identification may change.
2. Measure the body target at the robot's actual speed, and either accept a lower forward speed for the torque
   margin (for example 0.5 of the stall torque at 0.12 m/s), or keep 0.2 m/s with less margin.
3. If neither suffices: a faster servo (more no-load speed at the same stall torque), or a lighter swing (the swing's
   speed, not its mass, dominates here).

## Changed after the first run (maintainer, 2026-11-27)

**This change was made after the measurement above, having seen its result.** Condition 6 as first written (≤ 0.5 at
0.2 m/s) and its failure stay above as they were.

| # | Condition (changed) | Commands | Pass when, for every seed |
|---|---|---|---|
| 6 | Torque | (0.1, 0, 0) | worst joint's 95th percentile of torque ÷ stall torque over 5–20 s ≤ 0.5. At (0.15, 0, 0) and (0.2, 0, 0) the same value is reported only |

Decided by the maintainer: keep the 12 V ST3215 servo; the torque margin target (≤ 0.5 of the stall torque) applies
at walking speeds of 0.1–0.15 m/s, not 0.2 m/s. Reasons:

- The priority for the demonstration is stable walking, not speed.
- The redesign record's baseline of 0.41 was measured with the open-loop trot commanded at 0.2 m/s, which walks at
  about 0.1 m/s actual speed; the target was in effect set at about 0.1 m/s.
- The bench identification of the real servo (#70 and the servo identification of the redesign record) will revisit
  the torque–speed line this result depends on, and with it this condition.

Result with the changed condition (`scripts/base_controller.py`, five seeds, 20 s):

| Speed | Worst joint's 95th percentile ÷ stall torque | |
|---|---|---|
| 0.1 m/s | 0.34–0.35 | **pass** (condition 6) |
| 0.15 m/s | 0.52–0.55 | reported only |
| 0.2 m/s | 0.72–0.73 | reported only |

Context for the decision: servo candidates compared by the coordinating agent in MuJoCo with this controller
(seed 0, 12 s; position actuator model with each servo's stall torque and no-load speed; the servo masses in the
model were not changed). Worst joint's 95th percentile ÷ that servo's stall torque:

| Servo (seller's values) | 0.1 m/s | 0.15 m/s | 0.2 m/s | 0.3 m/s |
|---|---|---|---|---|
| ST3215 12 V (2.94 N·m, 4.72 rad/s; kept) | 0.35 | 0.52 | 0.73 | 1.00 |
| ST3215-HS (1.96 N·m, 11.1 rad/s) | 0.46 | – | 0.59 | fell |
| STS3250 (4.9 N·m, 7.87 rad/s) | 0.20 | – | 0.42 | 0.59 |

### Found while tuning (choices made with the implementation, to be reviewed)

The first two rows and the attitude estimate change the description of items 3 and 4 under "What it is", which was
written before the run; the code follows the rows below.

| Point | Choice | Reason |
|---|---|---|
| Speed feedback | An integral of the speed error (1 /s, bounded to ±0.15 m/s) added to the stance speed; no proportional term | With the model's soft servos (stiffness 10 N·m/rad) the open-loop stride covers about 0.6–0.7 of its length; the integral makes up the rest. A proportional term of 0.3 gave no better tracking and a little more torque; the first try, a proportional term of 1 with an integral of 2 /s, fell at 0.2 and 0.3 m/s |
| Velocity estimate and integral shared with the neighbours | Each period, a leg averages its estimate and its integral with its neighbours' (from their messages), then corrects the estimate by its own stance measurement | With a separate integral per leg, the left legs' and right legs' sideways integrals ran to opposite limits (±0.1 m/s), the legs pushed against each other, and the abduction torque rose from 0.06 to 0.25 of the stall torque. Averaging on the ring keeps them equal |
| Message contents | The four values erg.env reserves for learned values carry the velocity estimate (2) and the speed integral (2) | No other room in the message. When a policy sends learned values, the message needs more values (erg's own wire layout, `crates/erg-links`); to be decided with that work |
| Attitude estimate | Gyroscope integrated, pulled towards the accelerometer's gravity direction with a 1 s time constant, only when the reading is within ±20 % of g² | With 0.3 s and no gating the estimate was off by 0.1–0.2 rad while trotting (impacts and sway); with a gain of 1 the robot then fell at 0.2–0.3 m/s |
| Attitude gain | 0.5 of the hip's height error | 1 raised the torque and tilt; with 0.5, stepping in place under a steady torque of 0.4 N·m on the torso, the steady roll falls from 0.030 to 0.001 rad and the pitch from 0.034 to 0.011 rad |
| Turning | The stance feet's motion for the commanded turning speed, plus 0.5 times its error read on the leg's gyroscope | Without the gyroscope term the robot turned at 0.72 of the command; with it 0.75–0.76. The rest is lost to the feet slipping, which a policy may correct |
| Swing | Speed rising over the first quarter and falling over the last quarter of the forward move (peak 1.33 times the mean, against 1.57 for a half cosine); step height 12 mm as the hand-made trot; retraction 3 mm over the last fifth | Meant to ask less peak speed of the servos; in the torque it made little difference against a half cosine (within 0.02 of the stall torque). 6 mm of retraction cost about 0.02 more than 3 mm |
| Phase coupling | 0.5 of the frequency correction per unit of the sine of the phase error, summed over both neighbours; a neighbour's phase advanced by the age of its message | Clocks started at (0, 0.3, 0.1, 0.6) reach the trot within 5 s (test). Clocks started all equal stay equal: that state is a balance point of the coupling, but not a stable one; with the trot offsets of erg.env at reset it is not reached |

## Limits

- Simulation only, with the model's position actuators, not the servo model of #70; flat ground; no delay or loss
  in the virtual network beyond one control period.
- The velocity estimate assumes the stance foot does not slip.
- No stumble or early and late touch-down reflexes yet (redesign record, item 1); they come with steps and slopes
  (tier T2).
- Ported to the leg firmware (Issue #85, `firmware/leg-esp32/main/base_ctrl.c`) with the policy's adjustments: the
  same operations in the same order, with `xmath.c` for the trigonometry and its constants generated from this
  module into `obs_layout.h` (`scripts/gen_firmware_layout.py`), so a retune only regenerates them.
  `tests/test_firmware_host.py` checks the C version against this one bit for bit over recorded walks: four
  commands and seeds, with and without the policy's adjustments (held two control periods), with the neighbours'
  messages and, in one, 30 % message loss (missing and aged messages). On the robot the legs do not exchange
  messages yet (ESP-NOW is a later issue), so each leg's controller runs without its neighbours until then.
