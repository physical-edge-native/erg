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
differences); the CAPS terms are zero for a constant actor; deterministic training of both forms (`tests/test_ppo.py`); a tiny training run
of both forms through `scripts/train.py` that checks only the pipeline (well under a minute).

## Results of the first training (2026-11-27)

Trained with the code as committed with this record (`uv run python scripts/train.py --form {split,central}
--steps 1000000 --n-envs 8 --seed {0,1,2}`), the six runs at once on a four-thread cloud machine: 1,000,448
environment steps each, 1,708–1,819 s of wall-clock per run (about 31 minutes for all six); per step about 1.6 ms of
environment (base controller, servo model, MuJoCo) and 0.17 ms (per-leg) or 0.11 ms (central) of update. No fall in
training. Evaluated with `uv run python scripts/evaluate_learning.py` (175 runs of 20 s, 76 s on four processes).
Weights are not committed (about 260 kB per run with the critic); the runs are reproducible bit for bit from the
commands above.

Nothing in the conditions was changed after the run.

| # | Condition | Base controller alone | Per-leg, seeds 0 / 1 / 2 | Central, seeds 0 / 1 / 2 |
|---|---|---|---|---|
| L1 | Forward speed ±20 % | pass | pass / pass / pass | pass / pass / pass |
| L2 | Turning ±30 % | pass | pass / pass / **fail** | pass / pass / pass |
| L3 | No falls | pass | pass / pass / pass | pass / pass / pass |
| L4 | Smoothness ≤ 1.2 × base | (1.0) | **fail** / **fail** / **fail** | **fail** / **fail** / **fail** |
| L5 | Torque p95 ≤ 0.5 at 0.1 m/s | **fail** | **fail** / **fail** / **fail** | **fail** / **fail** / **fail** |
| L6 | Determinism | — | pass (tests) | pass (tests) |
| L7 | Zero action = base controller | — | pass (tests) | pass (tests) |

**Per form: both forms fail L4 and L5; the per-leg form also fails L2 (one policy, one seed).** Neither form passes
the conditions in this first training.

Measured values (ranges over the five evaluation seeds; torque is the worst joint's 95th percentile ÷ stall torque;
smoothness is the policy's mean |second difference| of the joint targets ÷ the base controller's on the same seed):

| Policy | Forward at 0.05 / 0.1 / 0.15 m/s | Yaw rate at (0.1, 0, ±0.3) | Torque at 0.1 m/s (seeds ≤ 0.5) | Torque at 0.15 m/s | Smoothness ratio at 0.1 m/s | Mean relative tracking error |
|---|---|---|---|---|---|---|
| Base controller | 0.050–0.056 / 0.108–0.112 / 0.163–0.176 | +0.27–0.31 / −0.24 to −0.29 | 0.56–0.70 (0 of 5) | 0.73–0.82 | 1.00 | 0.096 |
| Per-leg 0 | 0.054–0.055 / 0.111–0.113 / 0.164–0.173 | +0.23–0.30 / −0.21 to −0.24 | 0.57–0.73 (0 of 5) | 0.61–0.75 | 1.57–1.71 | 0.133 |
| Per-leg 1 | 0.052–0.054 / 0.106–0.108 / 0.163–0.168 | +0.23–0.31 / −0.24 to −0.29 | 0.47–0.61 (1 of 5) | 0.62–0.70 | 1.48–1.64 | 0.092 |
| Per-leg 2 | 0.054–0.055 / 0.108–0.110 / 0.166–0.173 | +0.26–0.27 / −0.20 to −0.26 | 0.49–0.63 (1 of 5) | 0.61–0.67 | 1.39–1.54 | 0.126 |
| Central 0 | 0.054–0.055 / 0.109–0.112 / 0.166–0.171 | +0.23–0.28 / −0.26 to −0.32 | 0.47–0.55 (2 of 5) | 0.63–0.69 | 1.80–1.90 | 0.101 |
| Central 1 | 0.052–0.055 / 0.110–0.113 / 0.164–0.173 | +0.22–0.26 / −0.27 to −0.31 | 0.49–0.53 (1 of 5) | 0.61–0.70 | 2.09–2.28 | 0.109 |
| Central 2 | 0.056–0.057 / 0.110–0.113 / 0.164–0.172 | +0.24–0.28 / −0.25 to −0.29 | 0.46–0.54 (4 of 5) | 0.60–0.72 | 1.65–1.78 | 0.112 |

The L2 failure: per-leg seed 2 at (0.1, 0, −0.3) on evaluation seed 900005 turned at −0.203 rad/s (limit −0.21).

What the learned adjustments did, against the base controller alone (step 3 of "Separating causes" measured
against step 1, under the same servo randomisation):

- **Torque went down, not enough.** At 0.1 m/s the worst joint's 95th percentile fell from 0.56–0.70 to 0.46–0.73
  (per-leg) and 0.46–0.55 (central); at 0.15 m/s from 0.73–0.82 to 0.60–0.75. The central policies lowered it more
  and more evenly in this run; 7 of 15 central evaluations and 2 of 15 per-leg evaluations at 0.1 m/s were at or
  under 0.5.
- **Tracking did not improve.** The base controller overshoots by 8–12 % at 0.1 m/s with the servo model; the
  policies kept the overshoot (forward 0.106–0.113 m/s) and their mean relative tracking error (0.09–0.13) is about
  the base controller's (0.096) or worse.
- **The joint targets became rougher**, 1.4–2.3 times the base controller's mean second difference, the central
  policies more than the per-leg ones in this run.
- No falls, as with the base controller.

These are differences measured on three seeds per form with one million steps each, in a first setup that misses
its own conditions; they are not a claim about the forms in general, and the main comparison (ten seeds per form,
longer runs) is still to be made.

## Notes after the run (not changes to the conditions)

- **Smoothness (L4) is probably dominated by the hold of the action.** The policy's adjustments are held for two
  control periods and then jump; a jump of a foot-placement residual or of the step height is a step in the swing
  foot's path, which shows directly in the second difference of the joint targets. The penalties on the actions'
  differences (weights 0.05 and 0.02) and the CAPS terms (temporal 0.1, spatial 1.0; their unweighted values rose during
  training to 0.02–0.07 and about 1e-5) were too weak to keep the adjustments still. Options for the maintainer, one at a time
  as the redesign record asks: interpolate the adjustments over the policy period (or a first-order filter in the
  base controller); raise the weights of the first and second differences or of the temporal CAPS term; or run the
  policy every control period (50 Hz).
- **Torque (L5)**: the base controller with the servo model misses the target by itself (0.56–0.70 at 0.1 m/s), so
  the policy has to remove up to a third of the torque with ±0.02 m, ±15 % frequency and ±6 mm of step height. The
  effort term (0.5 × mean squared torque ratio) is a mean over joints and time, while the condition is the worst
  joint's 95th percentile; a term on the hip flexion's peaks, or the servo's armature and gain from the bench
  identification (the servo model's record: the armature is the largest single cause), may matter more than the
  policy.
- **Tracking**: the speed term's width (0.05 m/s) gives almost no gradient for the base controller's overshoot of
  about 0.01 m/s (exp(−0.04) = 0.96), so the policy had little reason to correct it; the overshoot is better removed
  in the base controller itself (its speed integral with the servo model's delay), which is a change to #71's
  controller rather than to the learning setup.
- The logged mean reward per update jumps between about 0.7 and 2 because the time-limit bootstrap (the critic's
  value added to the last reward of an episode) is included in it; it is not a measure of progress.

## Issue #86, step 1: the base controller's speed overshoot with the servo model (2026-11-27)

Issue #86 asks, one change at a time on the same seeds: (1) retune the base controller's speed integral for the servo
model's overshoot and check #71's pass conditions with and without the servo model; (2) interpolate the policy's
adjustments over the 40 ms policy period; (3) only if L4 still fails, stronger smoothness and CAPS weights. #82's
pass conditions above are unchanged. Written with an AI assistant.

`scripts/base_controller.py` gains `--servo none|nominal|random` to run #71's conditions with the servo model.

**Result of step 1: no change to the base controller is adopted.** Measured (randomised servo, seeds 900001–900003,
20 s, the measures of the evaluation above):

| Speed integral (1/s; 1.0 now) | Forward at 0.05 / 0.1 / 0.15 m/s |
|---|---|
| 0 (none) | 0.049–0.051 / 0.104–0.108 / 0.144–0.158 |
| 0.3 | 0.051–0.056 / 0.111–0.112 / 0.158–0.169 |
| 1.0 | 0.050–0.054 / 0.108–0.110 / 0.163–0.172 |
| 3.0 | 0.052–0.054 / 0.108–0.110 / 0.162–0.167 |

The gain of the integral does not change the overshoot: the integral already brings the controller's **velocity
estimate** to the command (seed 900001 at 0.1 m/s: estimate 0.101 m/s, true 0.108 m/s; with the model's position
actuator 0.099 against 0.102). The estimate reads low because of the touch-down: in the first tenth of the gait cycle
after touch-down, the stance measurement (the foot's motion relative to the hip) is about 0.14 m/s below the torso's
speed (the foot is still settling and slides), and close to it in mid-stance. What was tried on the estimate and the
touch-down, and why it was not kept:

- Measuring only in mid-stance (cycle phase 0.15–0.45): the overshoot rose to 24–30 % at 0.1 m/s, since the body is
  slower in mid-stance than on average. Leaving out only the first 0.05 or 0.1 of the cycle changed little.
- More swing-leg retraction (10 mm over the last 0.4 of the swing, against 3 mm over the last 0.2), so that the foot
  meets the ground moving less: with the servo model the overshoot fell to 5–8 % at 0.1 m/s and 6–9 % at 0.15 m/s
  and the torque at 0.1 m/s to 0.56–0.63; but with the model's position actuator it broke #71's conditions (0.2 m/s
  seed 0 yaw drift −0.145 rad/s; 0.3 m/s: 0.50–0.92 of the command, yaw drifts up to 0.59 rad/s, one fall). 6 and
  15 mm, and 10 mm with a speed integral of 2 /s, gave overshoots of 2–10 % with the servo model (not run on #71's
  conditions).

#71's conditions with the unchanged controller (`scripts/base_controller.py --servo ...`, seeds 0–4, 20 s):

| Servo | Result |
|---|---|
| none (the model's position actuators) | all conditions pass, as recorded in #71 |
| nominal | fail: torque at 0.1 m/s 0.64–0.65 (condition 6); 0.2 m/s 0.240–0.260 (seeds 1 and 2 beyond +25 %); 0.3 m/s falls in 5 of 5 |
| randomised | fail: torque at 0.1 m/s 0.65–0.73; 0.3 m/s falls in 5 of 5; 0.2 m/s 0.228–0.247 (within 25 %) |

Turning, sideways, pushes at 0.2 m/s and determinism pass with the servo model too. Torque at 0.15 m/s (reported
only): 0.74–0.78 nominal, 0.77–0.90 randomised. So the base controller with the servo model does not meet #71's
conditions at 0.3 m/s and for torque; that holds before any learning and is outside the training commands
(0–0.15 m/s). A better velocity estimate (for example the touch-down slip from the IMU, or a stride-averaged
measurement) is left for a later issue; it changes #71's controller and its conditions have to be re-run.

Since the base controller is unchanged, the policies of step 1 are those of the first training above; nothing was
re-trained.

## Issue #86, step 2: adjustments ramped over the policy period (2026-11-27)

Change: each leg's adjustments go linearly from the previous policy step's to the new one over the two control
periods of a policy step (the first period gets the mean of the two, the second the new action), instead of being
held for both (`erg.ppo.period_actions`, `Config.interpolate`, on by default for the base-controller mapping). A zero
action is still the base controller bit for bit; evaluation applies the same ramp (tested bit for bit against
training). Everything else as in the first training: seeds 0–2, 1,000,448 environment steps per run (1,693–1,797 s
each, the six in parallel, about 30 minutes), the same evaluation.

| Policy | Forward at 0.05 / 0.1 / 0.15 m/s | Yaw rate at (0.1, 0, ±0.3) | Torque at 0.1 m/s (seeds ≤ 0.5) | Torque at 0.15 m/s | Smoothness ratio at 0.1 m/s | Mean relative tracking error |
|---|---|---|---|---|---|---|
| Base controller | 0.050–0.056 / 0.108–0.112 / 0.163–0.176 | +0.27–0.31 / −0.29 to −0.24 | 0.56–0.70 (0 of 5) | 0.73–0.82 | 1.00 | 0.096 |
| Per-leg 0 | 0.055–0.056 / 0.106–0.112 / 0.167–0.172 | +0.25–0.31 / −0.27 to −0.22 | 0.51–0.61 (0 of 5) | 0.57–0.72 | 1.11–1.27 | 0.124 |
| Per-leg 1 | 0.054–0.058 / 0.109–0.112 / 0.165–0.170 | +0.25–0.32 / −0.30 to −0.24 | 0.54–0.64 (0 of 5) | 0.55–0.71 | 1.28–1.50 | 0.112 |
| Per-leg 2 | 0.057–0.059 / 0.112–0.115 / 0.167–0.169 | +0.27–0.33 / −0.28 to −0.20 | 0.57–0.74 (0 of 5) | 0.62–0.78 | 1.31–1.52 | 0.144 |
| Central 0 | 0.053–0.055 / 0.108–0.110 / 0.163–0.165 | +0.22–0.30 / −0.32 to −0.27 | 0.47–0.61 (1 of 5) | 0.60–0.74 | 1.19–1.38 | 0.091 |
| Central 1 | 0.053–0.054 / 0.108–0.110 / 0.162–0.168 | +0.29–0.33 / −0.30 to −0.25 | 0.48–0.62 (1 of 5) | 0.61–0.66 | 1.09–1.23 | 0.085 |
| Central 2 | 0.056 / 0.111–0.113 / 0.168–0.175 | +0.24–0.27 / −0.30 to −0.25 | 0.47–0.59 (1 of 5) | 0.64–0.74 | 1.17–1.34 | 0.121 |

| # | Base controller | Per-leg 0 / 1 / 2 | Central 0 / 1 / 2 |
|---|---|---|---|
| L1 | pass | pass / pass / pass | pass / pass / pass |
| L2 | pass | pass / pass / **fail** | pass / pass / pass |
| L3 | pass | pass / pass / pass | pass / pass / pass |
| L4 | — | **fail** / **fail** / **fail** | **fail** / **fail** / **fail** |
| L5 | **fail** | **fail** / **fail** / **fail** | **fail** / **fail** / **fail** |

The ramp lowered the smoothness ratio from 1.39–1.71 to 1.11–1.52 (per-leg) and from 1.65–2.28 to 1.09–1.38
(central), but every policy is above 1.2 on at least one evaluation seed, so L4 still fails and step 3 follows. The
per-leg seed 2 policy again turns at −0.203 rad/s on evaluation seed 900005 (L2). Torque at 0.1 m/s: 0.51–0.74
per-leg (none of 15 evaluations at or under 0.5; 2 of 15 in the first training) and 0.47–0.62 central (3 of 15;
7 of 15 before). Tracking error 0.112–0.144 per-leg and 0.085–0.121 central, against 0.096 for the base controller
alone.

## Issue #86, step 3: ten times the smoothness and CAPS weights (2026-11-27)

Change (on top of step 2's ramp): penalties on the first and second differences of the actions 0.5 and 0.2 (0.05 and
0.02 before), CAPS temporal 1.0 and spatial 10.0 (0.1 and 1.0 before). Everything else as in step 2: seeds 0–2,
1,000,448 environment steps per run (1,640–1,784 s each, the six in parallel, about 30 minutes), the same evaluation.

| Policy | Forward at 0.05 / 0.1 / 0.15 m/s | Yaw rate at (0.1, 0, ±0.3) | Torque at 0.1 m/s (seeds ≤ 0.5) | Torque at 0.15 m/s | Smoothness ratio at 0.1 m/s | Mean relative tracking error |
|---|---|---|---|---|---|---|
| Base controller | 0.050–0.056 / 0.108–0.112 / 0.163–0.176 | +0.27–0.31 / −0.29 to −0.24 | 0.56–0.70 (0 of 5) | 0.73–0.82 | 1.00 | 0.096 |
| Per-leg 0 | 0.054–0.056 / 0.110–0.111 / 0.160–0.172 | +0.23–0.30 / −0.26 to −0.23 | 0.48–0.57 (1 of 5) | 0.59–0.68 | 1.05–1.13 | 0.124 |
| Per-leg 1 | 0.054–0.055 / 0.109–0.111 / 0.164–0.169 | +0.26–0.31 / −0.32 to −0.25 | 0.55–0.71 (0 of 5) | 0.62–0.77 | 1.01–1.13 | 0.095 |
| Per-leg 2 | 0.053–0.055 / 0.108–0.111 / 0.163–0.171 | +0.26–0.31 / −0.30 to −0.22 | 0.49–0.64 (1 of 5) | 0.62–0.73 | 1.04–1.18 | 0.099 |
| Central 0 | 0.054–0.055 / 0.109–0.112 / 0.164–0.170 | +0.27–0.30 / −0.30 to −0.23 | 0.53–0.65 (0 of 5) | 0.67–0.75 | 1.06–1.12 | 0.091 |
| Central 1 | 0.051–0.055 / 0.109–0.110 / 0.163–0.174 | +0.27–0.29 / −0.27 to −0.22 | 0.44–0.50 (5 of 5) | 0.54–0.69 | 0.98–1.03 | 0.097 |
| Central 2 | 0.054–0.055 / 0.109–0.111 / 0.165–0.174 | +0.26–0.30 / −0.32 to −0.26 | 0.46–0.57 (1 of 5) | 0.62–0.73 | 1.05–1.19 | 0.092 |

| # | Base controller | Per-leg 0 / 1 / 2 | Central 0 / 1 / 2 |
|---|---|---|---|
| L1 | pass | pass / pass / pass | pass / pass / pass |
| L2 | pass | pass / pass / pass | pass / pass / pass |
| L3 | pass | pass / pass / pass | pass / pass / pass |
| L4 | — | pass / pass / pass | pass / pass / pass |
| L5 | **fail** | **fail** / **fail** / **fail** | **fail** / pass / **fail** |

**Per form: both forms now pass L1–L4 and fail L5** (a form passes only if all three of its policies pass on every
evaluation seed). One central policy (seed 1) passes every condition; no per-leg policy does. At 0.1 m/s, 2 of 15
per-leg evaluations and 6 of 15 central evaluations have torque at or under 0.5 (0.48–0.71 per-leg, 0.44–0.65
central; the base controller alone 0.56–0.70). Smoothness ratios 1.01–1.18 per-leg and 0.98–1.19 central (limit 1.2).
Tracking error 0.095–0.124 per-leg and 0.091–0.097 central, against 0.096 for the base controller alone: the
policies still keep its overshoot (forward 0.108–0.112 m/s at 0.1 m/s).

Over the three steps of #86, measured on three seeds per form with one million steps each: smoothness came within
the limit only with the ramp and the stronger weights together; the torque margin at 0.1 m/s, which the base
controller with the servo model misses by itself, remains the failing condition for both forms. As before, these
are differences measured in this small setup, not a claim about either form in general.

Total for #86: three sets of measurements of the base controller (about 15 minutes on four threads) and two
trainings of six runs (about 30 minutes each on four threads, about 6 CPU-hours in all), plus three evaluations of
about 80 s each.
