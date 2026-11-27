# Learning setup: a policy on the base controller, with the servo model and smoothness terms

Date: 2026-11-27. Issue: #82. Builds on the base controller (#71), the servo model (#70) and body L1 (#59), and on
the redesign record (2026-11-27-redesign.md: control architecture items 2, 3, 4, 5, and step 3 of "Separating causes
when learning fails"). This record was written with an AI assistant. The pass conditions below were written and
committed **before** the training code and before any training run; results are added in a later section, with
nothing in the conditions changed.

## Decided by the maintainer (2026-11-27)

- Training commands at walking speeds of 0–0.15 m/s forward; the torque margin target (≤ 0.5 of the stall torque) at
  0.1–0.15 m/s.
- The policy adjusts the base controller (frequency, foot-placement residuals of about ±0.02 m, step height, gains);
  a zero action is the base controller exactly.
- Two forms with the same training budget in environment steps: **per-leg** (one network shared by the four legs,
  each leg seeing its own sensors and its two neighbours' messages through Dyne's wire format, as now) and
  **central** (one network for all legs, with the same outputs per leg).
- The servo model of #70 with a randomised delay.
- Smoothness: penalties on the first and second differences of the actions, and a Lipschitz (CAPS-type) term.
- Body-height and posture terms.
- The comparison is fair (same body, sensors, budget); the per-leg form is never claimed to be better overall.

## What was measured before writing the conditions

Step 1 of "Separating causes" with the servo model, which the base controller's record left for later: the base
controller alone, 12 s from seeds 0–2, measured over 4–12 s (scratch run, not a pass condition):

| Servo | Command | Forward speed | Worst joint's 95th percentile of torque ÷ stall | Mean \|second difference\| of joint targets (rad) |
|---|---|---|---|---|
| model's position actuator | 0.05 / 0.1 / 0.15 m/s | 0.055 / 0.097–0.100 / 0.150–0.154 | 0.31 / 0.34–0.35 / 0.51–0.52 | 0.0056 / 0.0078–0.0084 / 0.0128 |
| servo model, nominal (#70) | 0.05 / 0.1 / 0.15 m/s | 0.055 / 0.112–0.114 / 0.169–0.172 | 0.66–0.67 / 0.64–0.66 / 0.74 | 0.0065 / 0.0073–0.0075 / 0.011 |
| servo model, randomised | 0.05 / 0.1 / 0.15 m/s | 0.054–0.055 / 0.110–0.112 / 0.171–0.178 | 0.61–0.72 / 0.67–0.74 / 0.76–0.87 | 0.0060–0.0067 / 0.0074–0.0075 / 0.0102–0.0107 |

No falls. With the servo model the base controller **misses the torque target at 0.1 m/s** (0.64–0.74 against 0.5)
and overshoots the speed by 10–19 %. That is the starting point of the learned policy, and the reason the torque
term below is ten times its weight in the training record of 2026-11-26.

## Control and action mapping

Each control period (20 ms) every leg runs the base controller (`erg/base_controller.py`) from its own sensors and
its two neighbours' messages, in both forms (the base controller is the same for both: on the robot it runs on each
leg's microcontroller in both forms). The environment is the per-leg environment of `erg/env.py` with links between
neighbours for both forms, so that the base controller gets its messages; only the policy differs.

The policy acts every **2 control periods (40 ms, 25 Hz**; redesign record: "about 25 Hz"), and its action is held
for both periods. Per leg it gives 6 values; each is clipped to [-1, 1] and scaled into the base controller's
`Adjust`:

| Output | Adjusts | Scale (per unit) | Range |
|---|---|---|---|
| 1 | frequency correction of the leg's clock (added to the coupling) | 0.3 | clock frequency ±15 % (FREQ_RANGE 0.5) |
| 2 | foot-placement residual, forward | 0.02 m | ±0.02 m |
| 3 | foot-placement residual, sideways | 0.02 m | ±0.02 m |
| 4 | step height change | 0.006 m | 12 ± 6 mm |
| 5 | relative change of the placement (capture-point) gain | 0.5 | gain × 0.5–1.5 |
| 6 | relative change of the attitude gain | 0.5 | gain × 0.5–1.5 |

A zero action gives `Adjust()` with every field exactly 0.0, which is the base controller exactly (tested bit for
bit). The actor's last layer starts at zero weights and biases, so training starts from the base controller.

- **Per-leg form**: one actor shared by the four legs; each leg's input is its row of the per-leg observation of
  `erg/env.py` unchanged (67 values: own sensors, gait phase, command, leg identifier, three message slots of which
  only the two neighbours' are filled). Output 6. The message values the leg sends are the base controller's
  (velocity estimate and speed integral, which use the four learned slots of the message); the policy sends no
  values of its own in this setup.
- **Central form**: one actor on the central observation of `erg/env.py` unchanged (75 values: every leg's sensors
  and gait phase, the command). Output 4 × 6 = 24 in the order of the legs.
- Actor sizes: ReLU 48 × 48 for both, 5,910 parameters per leg and 7,176 central (under the 8,192 limit of Dyne's
  full mlp-actor). Same critic for both (ReLU 128 × 128 on the central observation, training only).

The observations of both forms are those of the earlier records, unchanged: the per-leg policy sees its neighbours'
messages, which carry the base controller's estimates; the central policy sees every leg's sensors but no
messages. This is the information each form has on the robot.

## Servo model and randomisation

`erg.servo.ServoParams()` (nominal values of #70) with `randomize_servo=True` in training: per episode, delay drawn
uniformly in 0–40 ms, kp, kd and friction each × U(0.8, 1.2) (the ranges of #70). The same for both forms.

## Commands

Per episode, drawn uniformly: forward 0–0.15 m/s, sideways −0.05 to +0.05 m/s, turning −0.3 to +0.3 rad/s. No
curriculum (the base controller already walks at these commands). Episodes last 10 s (500 control periods, 250
policy steps) or end at a fall (`erg.reward.fallen`: torso below 0.12 m or tilted more than 45°).

## Reward (per policy step; the per-period terms are averaged over its 2 control periods)

| Term | Weight | Form |
|---|---|---|
| Speed tracking | 1.0 | exp(−\|v_xy − c_xy\|² / 0.05²), v in the torso's heading frame (m/s) |
| Turning tracking | 0.5 | exp(−(ω_z − c_turn)² / 0.2²) (rad/s) |
| Body height | −100 per m² | (torso height − 0.15 m)² |
| Posture | −1.0 | roll² + pitch² of the torso, as the squared horizontal part of the gravity direction in the torso frame |
| Effort | −0.5 | mean over joints of (torque / stall torque)² |
| Action rate (first difference) | −0.05 | mean over outputs of (a_t − a_{t−1})² (clipped actions) |
| Action second difference | −0.02 | mean over outputs of (a_t − 2 a_{t−1} + a_{t−2})² |
| Fall | 0 | the episode ends |

The step's total is clipped below at zero, as in the training record (legged_gym). The tracking widths are narrower
than in the earlier record (0.25 m/s, 0.5 rad/s) because the base controller already walks near the command; the
widths of the earlier record were set to make a policy start walking from standing still. No air-time term: the
base controller sets the steps.

## Smoothness in the policy loss (CAPS)

Added to PPO's loss, on the actor's mean output μ (Mysore et al., 2021, "Regularizing action policies for smooth
control with reinforcement learning"):

- temporal: λ_T · mean (μ(s_t) − μ(s_{t+1}))², λ_T = 0.1, over consecutive policy steps of the same episode;
- spatial (a local Lipschitz penalty): λ_S · mean (μ(s) − μ(s + ε))², ε ~ N(0, 0.01²) on every observation value,
  λ_S = 1.0.

## PPO settings and budget

As in the training record, except where stated: 8 environments, 64 policy steps (128 control periods) per
environment between updates, 4 epochs of 4 minibatches, discount 0.99 per policy step, GAE λ 0.95, clip 0.2,
learning rate 3e-4, value coefficient 0.5, gradient norm 1.0 clipped separately for the actor and the critic
(added with the implementation, before any training: with one norm for both, the critic's large gradients early
in training, value losses of about 50, shrink the actor's step), entropy bonus 0.0 (changed from 0.01: the policy
starts at the base controller and only needs small adjustments), initial log standard deviation −1.6 (about 0.2 of
an output's unit range: ±0.004 m of foot placement).

**Budget: 1,000,000 environment steps (control periods of 20 ms) per run, the same for both forms; seeds 0, 1, 2
per form** (six runs). This is a first training in a cloud session (four CPU threads, about two hours in all), not
the main comparison (ten seeds per form, longer runs, on the maintainer's PC). The wall-clock time is reported.

## Evaluation

Each trained policy (its mean action) and the base controller alone are run the same way: the servo model
randomised from evaluation seeds 900001–900005 (not used by any training episode), from standing, 20 s per run,
measured over 5–20 s as in the base controller's record (speeds in the heading frame; yaw rate from the change of
yaw). Commands: (0.05, 0, 0), (0.1, 0, 0), (0.15, 0, 0), (0.1, 0, +0.3), (0.1, 0, −0.3).

## Pass conditions (stated before training)

For a form to pass a condition, **every** one of its three trained policies must pass it on every evaluation seed.

| # | Condition | Commands | Pass when |
|---|---|---|---|
| L1 | Forward speed tracked | (0.05, 0, 0), (0.1, 0, 0), (0.15, 0, 0) | mean forward speed within ±20 % of the command (the walking-targets record's tolerance); sideways drift within ±0.03 m/s; yaw drift within ±0.1 rad/s |
| L2 | Turning followed | (0.1, 0, ±0.3) | mean yaw rate within ±30 % of the command; forward speed within ±30 % of 0.1 m/s |
| L3 | No falls | all 25 runs per policy | no fall (base controller record: tilt > 45° or torso centre below 0.09 m) |
| L4 | Smoothness | (0.1, 0, 0) | mean over joints and time of \|second difference of the joint targets\| (rad, per control period) ≤ 1.2 × the base controller alone's value on the same evaluation seed |
| L5 | Torque margin | (0.1, 0, 0) | worst joint's 95th percentile of torque ÷ stall torque ≤ 0.5 |
| L6 | Determinism | — | training twice with the same seed gives the same weights bit for bit (tests) |
| L7 | Zero action | — | the zero action gives the base controller's joint targets bit for bit (tests) |

Reported next to the conditions, not conditions: the same measures for the base controller alone (it is expected to
fail L5, above); the torque at 0.15 m/s (the maintainer's range 0.1–0.15 m/s; reported only, since the base
controller is at 0.74–0.87 there); and the mean absolute relative tracking error over the five commands, learned
against the base controller alone. A form's result is these values; that one form passes more conditions than the
other in this small first training is a measured difference under these settings, not a claim about the forms in
general.

Reasons for the values:

- **±20 % forward** is the tolerance of a reached commanded speed in the walking-targets record.
- **±30 % turning**: sharper than the base controller's ±50 %, as the policy is meant to sharpen it, but turning comes
  after forward speed in the requirements (R2).
- **1.2 × the base controller's second difference**: the learned adjustments must not make the joint targets
  noticeably rougher than the hand-written controller (the jitter of #51's policies was the reason for the
  redesign); 20 % leaves room for the adjustments themselves.
- **Torque ≤ 0.5 at 0.1 m/s**: the maintainer's target (base controller record, condition 6 as changed).

## Tests (CI)

`tests/test_learning_setup.py`: zero action equals the base controller bit for bit (both forms, with the servo
model); action clipping and scales; each reward term on hand-made cases (height, posture, first and second
differences); the CAPS terms are zero for a constant actor; deterministic training of both forms; a tiny training run
of both forms through `scripts/train.py` that checks only the pipeline (well under a minute).
