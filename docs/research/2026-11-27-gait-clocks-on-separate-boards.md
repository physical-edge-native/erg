# Keeping the legs' gait clocks in step on separate boards

Date: 2026-11-27. Issue: #102. Written by an AI assistant during a scheduled run; not yet reviewed by the maintainer.

Since #51 each leg's controller keeps its own gait clock: a phase in [0, 1) that starts at the trot offsets and
advances once per control step by `CONTROL_PERIOD / PERIOD × (1 + FREQ_RANGE × f)`, where `f` in [−1, 1] is the
policy's frequency correction (`erg/env.py`). No clock reads another; legs keep step only through their messages,
which carry the sender's phase. In simulation the four legs step together from the same instant. On the robot each
leg's ESP32-S3 runs its own 20 ms loop (`firmware/leg-esp32/main/main.c`, timed with `esp_timer`) on its own crystal,
and starts when it is powered. This note works out how far that can differ from the simulation and collects what is
published on keeping separate boards in step. It changes no design decision.

**Kinds of statements.** Figures marked *worked out* are computed in this note from erg's own constants
(`CONTROL_PERIOD` = 0.02 s, `PERIOD` = 0.35 s on main, 0.45 s in the redesign's pull requests; `FREQ_RANGE` = 0.5)
and from assumed crystal tolerances. Figures marked *published* come from the cited sources, as read in abstracts,
listings and excerpts (full texts not read). Figures marked *secondary* come from forums or third-party guides.

## 1. Three ways the boards differ from the simulation

### 1.1 Crystal drift: negligible

- Assumption (*secondary*): a 40 MHz crystal for 2.4 GHz Wi-Fi is typically specified at ±10 ppm initial tolerance
  and about ±25 ppm in total over temperature and ageing (TI application note
  [SWRA680](https://www.ti.com/lit/swra680) for a comparable Wi-Fi chip; Espressif's ESP32 datasheet, as discussed
  on the [Espressif forum](https://esp32.com/viewtopic.php?p=155141), states the frequency but no stability). The
  ESP32-S3's `esp_timer` runs from that crystal.
- *Worked out*: two boards at the extremes differ by at most about 50 ppm. Because each clock advances once per
  control step, a board whose 20 ms is 50 ppm longer advances its phase 50 ppm more slowly in wall time. With
  `PERIOD` = 0.35 s the phase difference grows by 50 × 10⁻⁶ / 0.35 s ≈ **1.4 × 10⁻⁴ cycles per second**, about
  0.009 cycles (3 ms of a 350 ms gait cycle) per minute.
- The frequency correction can move a clock by up to `FREQ_RANGE` / `PERIOD` ≈ 1.4 cycles per second, about
  10,000 times the drift. Drift is well within what the policy can correct through its messages, and much smaller
  than the phase noise the policy already handles.

### 1.2 Loop offsets: a fraction of a step, and a slow slide

- Each board starts its 20 ms schedule when its own loop starts, so the four schedules are offset from each other by
  some δ between 0 and 20 ms, fixed after start-up.
- *Worked out*: a message sent at board A's step and arriving about 1 ms later (the ESP-NOW round trip of about
  1.2 ms measured by the Open Dynamic Robot Initiative, see the
  [distributed hardware note](2026-11-27-distributed-hardware-and-link-delays.md)) is read at board B's next step.
  The age of the information when B uses it is therefore anywhere from about 1 ms to about 21 ms, depending on δ, but
  it is always read **one step later**, as erg's simulation assumes by default.
- *Worked out*: drift makes δ slide. At 50 ppm, δ moves by 1 µs per step and passes through a whole 20 ms period in
  about 20 ms / 50 ppm = 400 s. Whenever B's step passes A's (every few minutes for some pair of legs), there is a
  short time in which a message arrives just before or just after B reads, so it may be read with **zero or two
  steps** of delay, or two messages may arrive in one step and none in the next. The receiving leg already handles
  this through the message age it observes (`AGE_CAP`), and the simulation's jitter (`jitter_us`) can produce the
  same pattern if set to the order of a step.
- The gait phase itself is not affected by δ beyond the phase step it represents: δ / `PERIOD` ≤ 20 / 350 ≈ 0.06
  cycles, the same order as one step of the clock.

### 1.3 Start-up: the largest difference

- In simulation the four clocks start at exactly the trot offsets. On the robot each board starts its clock when its
  firmware reaches the control loop. Boards powered from one switch still boot at slightly different times (Wi-Fi
  and radio start-up, sensor checks), and a board that resets alone restarts its clock at its start offset while the
  others keep running.
- *Worked out*: a start-up spread of 0.1 s is already 0.1 / 0.35 ≈ 0.29 cycles of phase error, far larger than drift
  (1.1) or loop offsets (1.2). At the full frequency correction, the policy could close it in about 0.2 s, if it has
  learned to; but training never shows it such errors, because every episode starts with the clocks exactly in step.

## 2. Published ways to keep separate boards in step

| Method | Idea | Published accuracy | Source |
|---|---|---|---|
| Reference Broadcast Synchronization (RBS) | one node broadcasts; receivers compare the times at which each of them received the same broadcast, which removes the sender's delay | microseconds; reported clearly better than NTP under similar conditions | Elson, Girod, Estrin, OSDI 2002, [doi:10.1145/844128.844143](https://doi.org/10.1145/844128.844143) (*published*) |
| Flooding Time Synchronization Protocol (FTSP) | a reference node floods time-stamped messages; time stamps are taken in the radio driver; each node fits a line to estimate offset and drift | average 1.6 µs, maximum 6.1 µs per hop on MICA2 sensor boards (from the authors' slides) | Maróti, Kusy, Simon, Lédeczi, SenSys 2004, [doi:10.1145/1031495.1031501](https://doi.org/10.1145/1031495.1031501) (*published*; figures from [slides](https://www.math.u-szeged.hu/~mmaroti/pdf/talks/2004%20The%20flooding%20time%20synchronization%20protocol.pdf)) |
| OpenWiFiSync on ESP32 | broadcast reference over Wi-Fi on ESP32 hardware | within about ±30 µs | [arXiv:2511.14457](https://arxiv.org/abs/2511.14457) (*published*, abstract only) |
| ESP-NOW time module (Espressif) | an initiator broadcasts its time; responders adjust their clocks | no figure given | [Espressif ESP-NOW documentation](https://docs.espressif.com/projects/esp-now/en/latest/esp32/api-reference/time/espnow_time.html) |
| Dyne's time synchronization | each board exchanges time stamps with a time server on a PC (as NTP does), keeps the samples with the shortest round trip, fits offset and drift over the last 32 | in simulation of the C code: 14 µs at the largest with small jitter, 481 µs with a long tail of delays; not yet measured on boards | `third_party/dyne/docs/2026-11-11-timesync-ota.md` (Dyne's own record) |

Dyne's English hardware guide adds that its split computations do not need synchronized clocks, because their steps
are aligned by the exchange with neighbours (`third_party/dyne/docs/en/hardware.md`). erg's legs are in the same
position for the steps; the gait clocks are what is new.

## 3. What this suggests for erg (for discussion, not decided)

1. **Clock synchronization to microseconds is not needed.** One millisecond of clock error is 0.003 cycles of gait
   phase. The methods above are far more accurate than erg needs, and Dyne's needs a PC time server, which the robot
   does not carry.
2. **Start-up needs a decision.** The largest gap between simulation and robot is how the clocks start. Options:
   - (a) a start message: the boards wait until they hear all their neighbours, then one leg (or the remote control)
     broadcasts "start at step n", and every leg starts its clock at its trot offset at that step;
   - (b) train with random initial phase errors (and an occasional reset of one leg's clock), so the policy learns to
     pull the clocks into step from its messages;
   - (c) both: (a) for a clean start, (b) for a leg that resets while walking (expected advantage 4, a controller
     leaving and rejoining).
   Option (b) or (c) changes the training conditions and would have to be written in the training design record
   before training; option (a) is firmware only. Recommended for discussion: (c).
3. **Loop offsets and their slide** are covered by the simulation's message delay and jitter if the jitter is set
   to the order of one step for the robustness runs (#41, #45). A random offset per leg, fixed per episode, would
   model 1.2 more closely; whether to add it is a choice for the training issue.
4. **Measurement on the robot.** The bench log can record each leg's phase and step counter with a time stamp, to
   measure the start-up spread and the phase differences while walking. Phase differences between legs, not clock
   errors, are what the comparison needs.

## Not covered yet

- Insect-inspired rule-based coordination (Walknet) and graph-network policies, still open from the first note.
- A primary source for ESP-NOW delay distributions (still open from the hardware note).
