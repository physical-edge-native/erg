# Prior work: decentralized (per-leg) versus central control of legged robots

Date: 2026-11-27. Issue: #73. Written by an AI assistant during a scheduled run; not yet reviewed by the maintainer.

This note collects earlier work that splits the control of a legged body into per-leg (or per-joint) units and,
where it was done, compares that with one central controller. It is input for erg's training and comparison
issues (#41 size of the learned vector in leg messages, #43 training both forms with the same budget) and for the
list of expected differences ([`docs/design/2026-11-25-expected-differences.md`](../design/2026-11-25-expected-differences.md)).
It changes no design decision.

**How far the statements were checked.** Each summary below was taken from the paper's abstract, its publisher or
arXiv listing, and short excerpts; the full texts were not read for this note. Numbers, network sizes and
experimental details must be checked in the papers before any of them is cited in a design record.

## Terms

- **Central policy**: one neural network that reads all sensors of the robot and outputs commands for all joints.
- **Decentralized (split) control**: several controllers, each commanding one leg (or one joint) from its own
  sensors, possibly with values from some other legs.
- **Modular policy with shared weights**: one small network, copied for every leg or joint, so all copies have the
  same parameters.
- **Central pattern generator (CPG)**: a set of coupled oscillators producing rhythmic leg motion.

## 1. Learned per-leg policies compared with a central policy

### Schilling, Konen, Ohl, Korthals (IROS 2020) — hexapod, simulation

[arXiv:2005.11164](https://arxiv.org/abs/2005.11164). Six local control modules, one per leg, each a neural network
trained with deep reinforcement learning. Each module's input is the local information of its own leg and of its two
neighbouring legs. Compared with a central controller on the same simulated hexapod, the abstract reports that the
decentralized version learned faster and that the spread of results over runs was narrower, while the central
version got stuck in poor solutions more often.

### Schilling, Melnik, Ohl, Ritter, Hammer (Neural Networks 144, 2021) — hexapod, simulation

[doi:10.1016/j.neunet.2021.09.017](https://doi.org/10.1016/j.neunet.2021.09.017). The journal follow-up compares
several architectures between fully central and fully local. The abstract reports that an intermediate form
(decentralized, each leg seeing only its own and its two neighbours' information) did well, and that decentralized
forms learned more robustly and generalized better to unseen conditions.

**Relevance to erg.** This is the closest earlier comparison to erg's: per-leg policies with neighbour information
against a central policy, same body. erg's default links (each leg with its neighbours on the ring) are the same
idea on four legs. Differences from erg: a hexapod (statically stable gaits are easier with six legs), simulation
only, and no real message channel (neighbour values arrive without loss or delay). erg adds lost and late messages,
the microcontroller's inference limits and the real robot.

### Zai El Amri, Hermes, Schilling (LOD 2022) — quadruped, simulation

[arXiv:2210.08003](https://arxiv.org/abs/2210.08003),
[doi:10.1007/978-3-031-25891-6_20](https://doi.org/10.1007/978-3-031-25891-6_20). Four per-leg policies on a
simulated four-legged agent, with a hierarchical layer above them for navigation. The abstract reports faster
learning than central designs.

**Relevance to erg.** The same group's result carried to four legs, which matters because a trot on four legs is
dynamically, not statically, stable; but still in simulation, with no communication limits.

### Liu, Guo, Lin, Ma, Zhu, Li — MASQ (arXiv 2024; ICML 2025 workshop) — quadruped, real robot

[arXiv:2408.13759](https://arxiv.org/abs/2408.13759). Each leg of one quadruped is treated as an agent in
cooperative multi-agent reinforcement learning, with a critic that sees the whole robot during training. The
abstract reports faster convergence than a single-agent baseline and more robust walking on the real robot on
several surfaces (grass, rocks, flat ground, rubber track).

**Relevance to erg.** The only work found here that deploys per-leg learned policies on a real quadruped. Two
points to check in the full text before comparing: whether each leg's actor uses only its own (and neighbours')
observations at run time, and whether the per-leg actors run on separate computers. If the actors run on one
computer with all observations available, the result is about learning, not about splitting the computation as erg
does. A central critic during training with per-leg actors at run time is an option erg's training design could
adopt without breaking the rule that each leg's controller sees only its neighbours while running (to be decided in
the training issue, not here).

## 2. Learned modular policies with message passing

### Huang, Mordatch, Pathak — Shared Modular Policies (ICML 2020)

[arXiv:2007.04976](https://arxiv.org/abs/2007.04976). One small network, shared by every actuator, controls its own
joint from its local sensors and exchanges learned messages with neighbouring modules. A single such policy walked
more than 20 different planar bodies (hoppers, bipeds, quadrupeds). The authors note that coordination of the whole
body emerges from the messages alone, trained only by the reinforcement-learning reward.

**Relevance to erg.** erg's leg messages already carry a learned vector chosen by the sending leg's policy (#35);
this work is the main precedent for that. Its messages travel within one computer, every step, without loss. #41
asks what the size of erg's learned vector buys when messages can be lost or late; this paper gives no answer to
that question.

### Kurin, Igl, Rocktäschel, Boehmer, Whiteson — Amorpheus (ICLR 2021)

[arXiv:2010.01856](https://arxiv.org/abs/2010.01856). Ablations of graph-based modular policies showed that the
body's graph (which module talks to which) did not help their performance; a transformer that lets every module
attend to every other did better.

**Relevance to erg.** A caution: in that setting, restricting communication to physical neighbours was a cost, not
a benefit. erg's restriction to neighbours comes from the hardware (one radio hop, small messages), not from an
expected gain in learning. This supports keeping "all links" as a setting in simulation, so the cost of
neighbour-only links can be measured (as the expected-differences record already plans).

### Guo, Jiang et al. — DEMOS (arXiv 2023)

[arXiv:2306.17411](https://arxiv.org/abs/2306.17411). Learns which parts of the body can be controlled separately
instead of fixing the split by hand, keeping the couplings that whole-body motion needs. Tested on simulated
quadrupeds and humanoids; the abstract reports robustness to motor failures and transfer to new tasks.

**Relevance to erg.** erg fixes the split per leg because each leg has its own microcontroller; a learned split is
not applicable to the hardware. The reported robustness to motor failure is close to erg's expected advantage 1
(keeps walking when one leg's controller stops), and could be a point of comparison for how that is measured.

## 3. Decentralized oscillators with local feedback (no learning)

### Owaki, Kano, Nagasawa, Tero, Ishiguro (J. R. Soc. Interface, 2013)

[doi:10.1098/rsif.2012.0669](https://doi.org/10.1098/rsif.2012.0669). A minimal quadruped robot with one oscillator
per leg and **no** connections between the legs' controllers; each oscillator uses only its own leg's load
(force) feedback. Coordination between the legs arises through the body's mechanics. The robot adapted its gait to
changes in weight distribution and speed. A follow-up
([Owaki and Ishiguro, Sci. Rep. 2017](https://doi.org/10.1038/s41598-017-00348-9)) reports gait transitions from
walk to trot to gallop with the same mechanism.

**Relevance to erg.** This is the strongest precedent for erg's "no links" setting: with load feedback, legs can
coordinate through the body alone. It suggests that erg's per-leg gait clocks (#51) could use the leg's own load
or contact to adjust their phase, and that the "no links" baseline need not be a straw man. It also shows what the
servos must provide: the STS3215-class servos report load, which is the signal these controllers use.

## What this suggests for erg (for discussion, not decided)

1. **Expected advantages.** Two earlier simulation results (Schilling et al. 2020, 2021; Zai El Amri et al. 2022)
   report faster or more reliable learning for per-leg policies. erg's expected-differences record does not list
   learning speed or reliability over seeds, because training compute was moved to gauss. A difference in the
   spread over seeds at the same budget could still be reported in #43 as an observation, without being a
   pre-registered claim. Whether to add it as an expected difference is the maintainer's call, and must be written
   before #43 runs.
2. **Fair baselines.** Amorpheus is a reminder that the central policy (or an all-links split) can win where the
   body's structure is imposed on the communication. erg already plans the "all links" setting in simulation;
   it should be kept in #43 and #45.
3. **Lost and late messages** are not studied in any of the learned works found here. This is where erg's
   measurements (#41, #45) would add something new, with Dyne's virtual network.
4. **Load feedback in the base gait.** The oscillator work suggests trying load- or contact-driven phase
   adjustment in the per-leg gait clocks; the base controller issue (#71) is the place to consider it.

## Not covered yet

- Hardware-distributed control in other projects (one microcontroller per leg with a field bus or radio) and its
  measured latencies.
- Insect-inspired rule-based coordination (Cruse's Walknet) and its comparison with learned controllers.
- Graph neural network policies for legged robots (NerveNet, Wang et al. 2018) in more detail.
