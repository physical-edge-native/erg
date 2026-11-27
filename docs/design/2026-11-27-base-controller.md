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

## Results

To be added after the run, with nothing above changed.
