# Prior work: control split over several microcontrollers, and measured link delays

Date: 2026-11-27. Issue: #100. Written by an AI assistant during a scheduled run; not yet reviewed by the maintainer.

The first research note ([2026-11-27-decentralized-legged-control.md](2026-11-27-decentralized-legged-control.md))
covered how control can be split in software. This note covers how existing robots and boards split control over
**several microcontrollers**, and what delays their links between boards show where they were measured. It is input
for the delays and losses erg assumes in simulation (0, 1 or 2 control steps of 20 ms, loss 0–30 %; #41, #45) and
for choosing the link between leg boards (ESP-NOW, Wi-Fi UDP or CAN; not decided). It changes no design decision.

**How far the statements were checked.** Figures marked *primary* come from the project's own repository or paper
(as seen in listings and excerpts; the full texts were not read). Figures marked *secondary* come from articles,
forums or project logs and should not be cited in a design record until checked against a primary source.

## Terms

- **Round-trip time (RTT)**: time from sending a message to receiving the reply, including the software on both ends.
  A one-way delay is roughly half of it, if both directions are alike.
- **ESP-NOW**: Espressif's connectionless protocol that sends short frames (up to 250 bytes) directly between ESP32
  boards over the Wi-Fi radio, without an access point.
- **CAN**: a wired bus common in vehicles and robots; classic CAN runs at up to 1 Mbit/s with 8-byte frames.

## 1. Robots and boards

### Open Dynamic Robot Initiative (Solo quadruped): one ESP32 "master board" for all joints

Grimminger et al., "An Open Torque-Controlled Modular Robot Architecture for Legged Locomotion Research", IEEE
Robotics and Automation Letters 2020, [arXiv:1910.00093](https://arxiv.org/abs/1910.00093); board repository:
[open-dynamic-robot-initiative/master-board](https://github.com/open-dynamic-robot-initiative/master-board).

- Structure: the joint motor drivers connect to one ESP32-based master board, which links to an off-board computer
  that runs the controller at 1 kHz.
- Measured (*primary*): about **0.2 ms** round trip over 100 Mbit/s Ethernet for a 127-byte message, including
  driver and operating-system latency; about **1.2 ms** round trip over raw ESP-NOW (Wi-Fi), as given in the
  repository's README.
- Relevance: the closest published figure for the radio erg is likely to use, on the same family of chips. A round
  trip of about 1 ms is small against erg's 20 ms period, so one control step of delay (the default in erg's
  simulation) is a cautious assumption for the delay itself. The figure is for one link between two boards in a
  laboratory; erg's four legs share one radio channel, which is where losses and occasional long delays would come
  from. The ODRI board runs the radio at much higher rates than erg needs, which suggests ESP-NOW has room for
  erg's 50 messages per second per leg.

### MIT Mini Cheetah: one CAN bus per leg

Katz, Di Carlo, Kim, "Mini Cheetah: A Platform for Pushing the Limits of Dynamic Quadruped Control", ICRA 2019
([doi:10.1109/ICRA.2019.8793865](https://doi.org/10.1109/ICRA.2019.8793865)); interface firmware
([SPIne](https://os.mbed.com/users/benkatz/code/SPIne/)).

- Structure: the main computer talks over SPI to interface boards that drive four CAN buses, one per leg, with the
  three actuators of that leg on it. Each actuator closes its own current and joint loops at high rate.
- Figures (*secondary*): one CAN bus per leg because a single bus cannot carry twelve actuators at the desired rate;
  a bus ceiling of roughly 4 kHz divided by the number of actuators on it is quoted from Katz's thesis.
- Relevance: the same reason erg gives each leg its own servo bus. The split here is of the **bus**, not of the
  **policy** (one computer still runs the whole-body controller). It supports erg's expected advantage 3 (reading
  three servos per bus allows a shorter period than twelve on one bus) as a design rule used in practice, not as
  a measured comparison.

### StarlETH: CAN at 400 Hz

Hutter et al., StarlETH (2013); figure quoted in a 2019 survey of legged-robot buses
([SciTePress 2019](https://www.scitepress.org/Papers/2019/79782/79782.pdf), *secondary*): a CAN bus and the
high-level controller both at 400 Hz. The survey notes that bus latency reduces the stability margin of a control
loop, and that loop rates are limited when control is centralized.

### Hexapod with one microcontroller per leg (2011)

[SciTePress 2011](https://www.scitepress.org/Papers/2011/34048/34048.pdf) (*secondary*, from the abstract): six leg
boards, each running the position loops of its three motors, linked over Ethernet/UDP to a PC with real-time Linux
that generates the gait. As in the Mini Cheetah, the legs' boards run low-level loops only; the gait is central.

### Modular robots with one oscillator per module: YaMoR

Sproewitz, Moeckel, Maye, Ijspeert, "Learning to move in modular robots using central pattern generators and online
optimization", International Journal of Robotics Research 27(3–4), 2008
([doi:10.1177/0278364907088401](https://doi.org/10.1177/0278364907088401);
[EPFL copy](https://infoscience.epfl.ch/record/118460)).

- Structure: one oscillator per module, coupled to the others over **Bluetooth** (a custom scatternet protocol), so
  the rhythm itself is computed on separate boards and kept in step by radio.
- Relevance: the closest structural precedent for erg's split form, where each leg keeps its own gait clock (#51)
  and corrects it from its neighbours' messages. No delay figure was found in the parts read. The note on
  oscillators in the first research note applies: coupling through the body can carry part of the coordination
  when messages are late.

## 2. ESP-NOW losses and delays outside laboratories

- Espressif's developer blog ([reliability of ESP-NOW](https://developer.espressif.com/blog/reliability-esp-now/),
  community-written, *secondary*): ranges up to about 200 m with low loss in open space; loss depends strongly on
  the building.
- A road-vehicle field study (MDPI *Technologies*, 2026,
  [link](https://www.mdpi.com/2227-7080/14/6/344), *secondary* as read from the abstract): average delays from
  about 10 ms to about 115 ms and losses from 2 % to 60 % as speed rose from 10 to 70 km/h. Not comparable with
  boards a few centimetres apart on one robot, but it shows how far delays stretch when links degrade.
- An Espressif forum thread (*secondary*): losses in a room with 13 devices were attributed to channel congestion.

Relevance: the losses erg simulates (up to 30 %) are within what degraded links show; for four boards a few
centimetres apart they are likely pessimistic, except when the 2.4 GHz channel is crowded (other Wi-Fi networks,
Bluetooth). Delays longer than 2 control steps (40 ms) appear in degraded links, so a test at a larger delay might
be worth adding to the robustness measurements, as a check rather than a claim.

## What this suggests for erg (for discussion, not decided)

1. **Delay assumptions.** The one ESP-NOW figure found (about 1.2 ms round trip between two ESP32s) is far below
   one 20 ms control step. erg's default of one step of delay mostly reflects the control schedule (a message sent
   in one step is read in the next), not the radio. Dyne's measurement on boards (dyne#167) should report the
   distribution of delays, not only the mean, because a long tail would decide how often a message misses its step.
2. **Loss and congestion.** The literature points to channel congestion, not distance, as the likely cause of
   losses on a small robot. The bench test of the link could be run once in a quiet room and once next to a busy
   Wi-Fi network.
3. **What is new in erg.** In every robot found here, the boards on the legs run low-level loops only and one
   computer runs the whole-body controller (Mini Cheetah, the 2011 hexapod, Solo). YaMoR runs the rhythm on the
   modules, but not a learned policy. Running a learned whole-body policy split over the legs' own microcontrollers,
   exchanging values only with neighbours, was not found in this search. This is an observation from a short search,
   not a claim of novelty; a wider search is needed before saying so anywhere public.
4. **Wired alternative.** CAN at 1 Mbit/s between the leg boards would remove radio losses at the cost of wiring,
   which bears on expected advantage 7 (simpler wiring). If the radio link turns out unreliable, that trade-off is a
   decision for the maintainer.

## Not covered yet

- Measured ESP-NOW latency distributions in peer-reviewed work (none found in this search).
- Time synchronization between leg boards (each leg keeps its own gait clock; how far the clocks drift).
- Insect-inspired rule-based coordination (Walknet) and graph-network policies, still open from the first note.
