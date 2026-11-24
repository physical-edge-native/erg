# The four-legged robot, and what we compare

Date: 2026-11-24. Issue: #2. Decided by the maintainer; this record was written with an AI assistant.
Background research (parts, prices, prior work) is kept in the maintainer's private notes.

## Decided (2026-11-24)

### (a) Legs: three joints per leg, Feetech STS3215 serial bus servos

- 12 joints in total (hip abduction, hip flexion, knee on each leg).
- Feetech STS3215, 7.4 V class: about 19.5 kg·cm stall torque at 7.4 V, about 0.19 s per 60°, 55 g.
  Each servo reports position (12-bit, magnetic encoder), speed, load, voltage, current and temperature
  over a half-duplex serial bus at up to 1 Mbps.
- About ¥3,245 per servo in Japan (about $18 abroad). The whole robot, without frame, wiring and charger,
  comes to about ¥52,000–60,000; mass about 1.2–2 kg.

Why:

- A controller that sees only its own leg needs its own joint angles and loads; servos without feedback leave
  it almost nothing to observe (work on hobby-servo quadrupeds without encoders had to drop joint angles from
  the policy input).
- Three joints per leg is what every quadruped with a learned policy transferred to hardware at scale uses;
  two joints cannot step sideways.
- The STS3215 has public tools for identifying its friction and a MuJoCo model (BAM, Open Duck Mini), which
  matters because a wrong servo model is a known cause of failed transfer.
- Cost is measured, not only minimised: we first compare on a body that works, then measure what is lost on
  cheaper builds.

Other options considered: two joints per leg with the same servo (about ¥39,000–47,000; no sidestep, less to
observe per leg); three joints with the 13 g Feetech SCS0009 (about ¥33,000; little torque margin, no friction
model); two joints with SG90 servos without feedback (about ¥12,000; nothing for a per-leg controller to see).

### (b) One microcontroller and one IMU per leg, no body computer

- Each leg has its own ESP32-class microcontroller driving its three servos on its own bus, and its own IMU
  mounted where the leg meets the body, so that all four IMUs measure the motion of the body.
- No computer sits in the body.

Why: no central node (each leg is a peer, as in the Physical-Edge Native Architecture); every per-leg
controller in prior work sees the body's orientation; with four IMUs, legs can average their estimates
through messages and carry on when one leg stops. Dyne's `imu-fusion-actor` and its 2026-11-21 hardware
record of several IMUs on one board apply here.

Other options considered: a fifth microcontroller in the body that measures orientation and sends it to all
legs (one IMU, but a star whose centre stops all legs); an IMU on one leg only (cheapest, but that leg becomes
special).

### (c) Which legs talk to which: a parameter, compared in stages

In simulation, the links between legs are a setting, and we compare, on the same body and budget:

1. no messages between legs (coordination only through the body);
2. neighbours only: each leg with the leg beside it and the leg in front of or behind it
   (a ring: front-left – front-right – hind-right – hind-left – front-left);
3. all four legs with each other;
4. one central policy that sees everything.

On the built robot the default is **neighbours only**, following the architecture's rule that regions
exchange values only with neighbours.

Why: where the differences appear along this ladder is itself a result. Prior work found per-leg control with
both neighbours' information as good as or better than central control, but assumed perfect communication;
erg adds message delay and loss.

## Also decided (2026-11-24, proposed and confirmed by the maintainer)

| Point | Decision | Reason |
|---|---|---|
| (d) Message from a leg to its neighbours, once per control step | its body-orientation estimate (gravity direction, 3 values) and angular velocity (3), its foot contact (1), and a short learned vector (4 values); 11 values as one Dyne band message (24-byte header + 88 bytes = 112 bytes) | fits in one ESP-NOW packet (up to 250 bytes); the learned vector lets legs share what they need without fixing it by hand |
| (e) Sensing, without cameras | per joint: position, speed, load; per leg: IMU (accelerometer and gyroscope), foot contact by a microswitch (about ¥300); servo load as a second contact signal | the cheapest reliable contact signal; force pads cost about 10 times more |
| Control period | 20 ms (50 Hz) | reading three servos takes well under 1 ms at 1 Mbps (to be measured); ESP-NOW round trips are a few milliseconds |
| The central policy on hardware | one microcontroller with all 12 servos on its bus, no radio | this is how a conventional small robot is built; it is the honest baseline, with no message delay |
| Link between leg controllers | decided after measuring on boards; candidates: ESP-NOW (direct between devices, no access point), Wi-Fi UDP (what Dyne uses now), CAN (wired, fixed delay) | in simulation, delay and loss are parameters from Dyne's seeded virtual network, so the choice does not block #4 and #5 |
| Microcontroller | XIAO ESP32S3 for now (about ¥1,456; single-precision floating-point hardware); first measure the inference time of Dyne's `mlp-actor` on the ESP32 boards already used for Dyne | Dyne's `mlp-actor` computes in double precision, which every ESP32 does in software, and runs interpreted; if it does not fit in 20 ms, propose single-precision variants to Dyne (Dyne issue #71) |
| Policy network | small ReLU networks, the same weights shared by the four legs (with a leg identifier as input) | ReLU keeps the WebAssembly output bit-exact with the Python reference; sharing weights is common in prior work and cuts training cost |

## What we measure (f), and how results are accepted

All numbers are reported per configuration of (c), with the same training budget (environment steps) and at
least 5 training seeds each; we report the mean and the spread, never the best seed alone.

- **Walking**: speed-tracking error at commanded forward, sideways and turning speeds; falls per minute;
  the same on uneven ground not seen in training.
- **Faults**: message loss (0–30 %), message delay (0–2 control steps, with jitter), one leg's controller
  stopping (its leg holds its last position) and coming back, following Dyne's leave/rejoin rules.
- **Computation**: parameters and multiply-adds per controller; inference time measured on the
  microcontroller.
- **Communication**: bytes per control step per leg and for the whole robot.
- **Training**: environment steps and CPU hours until a set walking speed is first reached.

Acceptance of the tools themselves (not of a "winner"):

- a controller's WebAssembly output equals the Python reference bit for bit (ReLU networks);
- the same seed gives identical simulation outputs on every run;
- tolerances for comparing simulation with hardware are written down before the hardware runs, with their
  grounds.

## (g) Cost

- Parts cost of the robot, recorded from actual purchases (the receipts themselves stay private).
- CPU hours to train, on the stated machine (22 threads).
- Computation per control step on the microcontroller.
- A target price for the robot is **not set for the first robot**; the maintainer sets it after the actual
  cost of the first build is measured.
